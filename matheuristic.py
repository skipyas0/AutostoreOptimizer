from time import sleep
import argparse
import json
import os
import pickle
import random

from docplex.cp.solution import CpoSolveResult
from loguru import logger

import cp_model as docplex_model
import ortools_cp_model as ortools_model
import cpp_cp_model
from autostore_heuristic import BinEvent, Solution, validate_solution
from freezing_utils import (
    FreezeManager,
    apply_partial_starting_point,
    create_partial_starting_point,
    generate_freeze_constraints,
    get_optimized_set,
)
from instance import Instance
from jaccard_similarity import build_similarity_matrix
from matheuristic_plots import LoguruStream, Status, VisualLogger
from neighborhood_selection import (
    strategy_random_orders,
    strategy_similar_orders,
)
from precalculate_instances_for_lns import check_if_precalculated, precalculate_config

random.seed(42)


class MockVarSol:
    def __init__(self, name, present, start=None, end=None, val=None):
        self.name = name
        self.present = present
        self._start = start
        self._end = end
        self._val = val

    def get_name(self):
        return self.name

    def get_value(self):
        return self

    def is_present(self):
        return self.present

    def get_start(self):
        return self._start

    def get_end(self):
        return self._end


class ORToolsSolution:
    def __init__(self, src, handles):
        self.var_map = {}
        self.handles = handles

        if hasattr(src, "Proto"):  # is a mdl (initial solution hints)
            hints = src.Proto().solution_hint
            for idx, var_idx in enumerate(hints.vars):
                self.var_map[var_idx] = hints.values[idx]
        else:  # is a solver (extracted active solution)
            for var in handles["all_intervals_flat"].values():
                self.var_map[var.pres.Index()] = src.Value(var.pres)
                if src.Value(var.pres):
                    self.var_map[var.start.Index()] = src.Value(var.start)
                    self.var_map[var.end.Index()] = src.Value(var.end)

    def Value(self, var):
        return self.var_map.get(var.Index(), 0)

    def BooleanValue(self, var):
        return bool(self.var_map.get(var.Index(), 0))

    def get_all_var_solutions(self):
        sols = []
        for var in self.handles["all_intervals_flat"].values():
            sols.append(self.get_var_solution(var))
        return sols

    def get_var_solution(self, var):
        pres = self.BooleanValue(var.pres)
        # return a MockVarSol which has get_var() == var, but wait, MockVarSol currently expects name!
        if pres:
            vs = MockVarSol(
                var.name, True, start=self.Value(var.start), end=self.Value(var.end)
            )
        else:
            vs = MockVarSol(var.name, False)
        # Inject the var object so get_var() works if it exists
        vs._var = var
        # override get_var to return the actual var
        vs.get_var = lambda: var
        return vs


def cp_sol_to_solution(sol, handles):
    def iv_present(x):
        if x is None:
            return False
        if hasattr(sol, "BooleanValue"):
            return sol.BooleanValue(x.pres)
        vs = sol.get_var_solution(x)
        return (vs is not None) and vs.is_present()

    def iv_start(x):
        if hasattr(sol, "Value"):
            return sol.Value(x.start)
        return sol.get_var_solution(x).get_start()

    def iv_end(x):
        if hasattr(sol, "Value"):
            return sol.Value(x.end)
        return sol.get_var_solution(x).get_end()

    I_os_lane = handles["I_os_lane"]
    I_os = handles["I_os"]
    P = handles["P"]
    F = handles["F"]
    R = handles["R"]
    B = handles["B"]
    U = handles["U"]
    orders_req = handles["orders_req"]
    S, L, K, O = handles["S"], handles["L"], handles["K"], handles["O"]

    makespan = 0
    for o in O:
        ends = [iv_end(I_os[(o, s)]) for s in S if iv_present(I_os[(o, s)])]
        if ends:
            makespan = max(makespan, max(ends))

    order_assignments = {}
    for o in O:
        s_sel = next((s for s in S if iv_present(I_os[(o, s)])), None)
        if s_sel is not None:
            ln_sel = next(ln for ln in L if iv_present(I_os_lane[(o, s_sel, ln)]))
            st, en = iv_start(I_os[(o, s_sel)]), iv_end(I_os[(o, s_sel)])
            order_assignments[o] = (s_sel, ln_sel, st, en)

    bin_events = {s: [] for s in S}
    total_moves = 0

    pick_events = {}

    for s in S:
        for k in K:
            Uk = U[k]
            for e in range(Uk):
                if (s, k, e) in B and iv_present(B[(s, k, e)]):
                    total_moves += 2

                    fs, fe = iv_start(F[(s, k, e)]), iv_end(F[(s, k, e)])
                    bs, be = iv_start(B[(s, k, e)]), iv_end(B[(s, k, e)])
                    rs, re = iv_start(R[(s, k, e)]), iv_end(R[(s, k, e)])

                    orders_served = []
                    for o in O:
                        if (
                            k in orders_req[o]
                            and (o, s, k, e) in P
                            and iv_present(P[(o, s, k, e)])
                        ):
                            ps, pe = (
                                iv_start(P[(o, s, k, e)]),
                                iv_end(P[(o, s, k, e)]),
                            )
                            orders_served.append(o)
                            pick_events[(o, s, k)] = (ps, pe)

                    be_obj = BinEvent(
                        sku=k,
                        copy_id=e,
                        fetch_start=fs,
                        fetch_end=fe,
                        presence_start=bs,
                        presence_end=be,
                        return_start=rs,
                        return_end=re,
                        orders_served=orders_served,
                    )
                    bin_events[s].append(be_obj)

    return Solution(
        order_assignments=order_assignments,
        bin_events=bin_events,
        makespan=makespan,
        total_moves=total_moves,
        feasible=True,
        pick_events=pick_events,
    )


def sort_variables(mdl, sp, backend="docplex", handles=None):
    if backend == "ortools":

        def get_start_time(var):
            if sp.BooleanValue(var.pres):
                return sp.Value(var.start)
            return float("inf")

        interval_vars = list(handles["all_intervals_flat"].values())
        sorted_vars = sorted(interval_vars, key=lambda v: (get_start_time(v), v.name))
        var_to_idx = {var: idx for idx, var in enumerate(sorted_vars)}
        return var_to_idx, len(var_to_idx)
        
    if backend == "cpp":
        def get_start_time(var):
            var_sol = sp.get_var_solution(var)
            if var_sol is not None and var_sol.is_present():
                return var_sol.get_start()
            return float("inf")

        interval_vars = list(handles["all_intervals_flat"].values())
        sorted_vars = sorted(interval_vars, key=lambda v: (get_start_time(v), v.get_name()))
        var_to_idx = {var: idx for idx, var in enumerate(sorted_vars)}
        return var_to_idx, len(var_to_idx)

    def get_start_time(var):
        var_sol = sp.get_var_solution(var)
        if var_sol is not None and var_sol.is_present():
            return var_sol.get_start()
        # Push absent variables to the very top (or bottom) of the y-axis
        return float("inf")

    # 1. Filter only interval variables
    interval_vars = [
        var for var in mdl.get_all_variables() if "Interval" in var.type.name
    ]

    # 2. Sort chronologically by start time in the warmstart (sp),
    # using the variable name as a secondary tie-breaker for stability.
    sorted_vars = sorted(interval_vars, key=lambda v: (get_start_time(v), v.get_name()))

    # 3. Create the mapping based on the sorted order
    var_to_idx = {var: idx for idx, var in enumerate(sorted_vars)}
    num_variables = len(var_to_idx)
    return var_to_idx, num_variables


def load_instance(instance_config=None, instance_folder=None):
    if instance_folder is None and instance_config is None:
        raise ValueError("Provide at least one non-None parameter")

    if instance_folder is None:
        instance_folder = check_if_precalculated(instance_config, with_removal=True)
        if instance_folder is None:
            print(
                f"Precalculating... this will take a while ({instance_config['time_limit']})"
            )
            instance_folder = precalculate_config(instance_config)

    elif instance_config is None:
        with open(f"{instance_folder}/config.json", "r") as f:
            instance_config = json.load(f)

    if "experiments" not in os.listdir(instance_folder):
        raise ValueError("Provided instance folder is incomplete")
    else:
        instance = Instance.from_pickle(f"{instance_folder}/instance.pkl")

    with open(f"{instance_folder}/heuristic_solution.pkl", "rb") as f:
        heur_sol = pickle.load(f)

    with open(f"{instance_folder}/cp_intermediate_records.pkl", "rb") as f:
        cp_records = pickle.load(f)
    return instance, heur_sol, cp_records, instance_folder, instance_config


def prepare_model_docplex(config, instance, heur_sol):

    mdl, handles = docplex_model.build_model(
        instance,
        add_symmetry_breaking=config["symmetry_breaking"],
        horizon=config["horizon"],
    )

    starting_point = docplex_model.inject_warmstart(
        heur_sol, heur_sol.pick_events, mdl, handles
    )
    return mdl, handles, starting_point


def prepare_model_ortools(config, instance, heur_sol):
    mdl, handles = ortools_model.build_model(
        instance,
        rt_return=instance.rt_ret,
        add_symmetry_breaking=config["symmetry_breaking"],
        horizon=config["horizon"],
        move_cap=config["movecap"],
    )
    ortools_model.inject_warmstart(heur_sol, heur_sol.pick_events, mdl, handles)
    starting_point = ORToolsSolution(mdl, handles)
    return mdl, handles, starting_point


class MockCpoStartingPoint:
    def __init__(self):
        self.state_dict = {}

    def add_interval_var_solution(self, var, presence=False, start=None, end=None, size=None):
        name = var.get_name()
        d = {"present": 1 if presence else 0}
        if start is not None:
            d["start"] = start
        if end is not None:
            d["end"] = end
        if size is not None:
            d["size"] = size
        self.state_dict[name] = d
        
    def add_integer_var_solution(self, var, value):
        pass


class MockCpoModel:
    def create_empty_solution(self):
        return MockCpoStartingPoint()


def prepare_model_cpp(config, instance, heur_sol):
    mdl, handles = cpp_cp_model.build_model(
        instance,
        add_symmetry_breaking=config["symmetry_breaking"],
        horizon=config["horizon"],
    )
    mock_mdl = MockCpoModel()
    mock_sp = docplex_model.inject_warmstart(
        heur_sol, heur_sol.pick_events, mock_mdl, handles
    )
    
    warm_names, warm_p, warm_s, warm_e = [], [], [], []
    for name, state in mock_sp.state_dict.items():
        warm_names.append(name)
        warm_p.append(state.get("present", 0))
        warm_s.append(state.get("start", 0))
        warm_e.append(state.get("end", 0))
        
    mdl.apply_warm_start(warm_names, warm_p, warm_s, warm_e)
    return mdl, handles, cpp_cp_model.CppSolveResult({
        "status": "Feasible", 
        "objective": heur_sol.makespan, 
        "var_solutions": mock_sp.state_dict
    })


class Solver:
    def __init__(self, experiment_config, instance_config=None, instance_path=None):
        (
            self.instance,
            self.heur_sol,
            self.cp_records,
            instance_path,
            instance_config,
        ) = load_instance(instance_config, instance_path)
        self.instance_config = instance_config
        self.vlg = VisualLogger(instance_path, instance_config, experiment_config)
        self.severity = 1
        self.stagnation_count = 0
        self.experiment_config = experiment_config

        for run in range(self.experiment_config["runs"]):
            self.run_solver()
        self.vlg.save_experiment()

    def eps_greedy_acceptance(self, sol_object, solve_status=None):
        """
        Perform eps-greedy acceptance on incumbent solution:
            - Always update current solution if the incumbent is strictly better
            - Otherwise update with a fixed probability: config['eps_greedy_prob']

        Handles best/current solution updates, stagnation count and logging to VLG
        """

        if self.experiment_config["backend"] == "ortools":
            from ortools.sat.python import cp_model

            if solve_status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                if self.experiment_config["improvement_constr"]:
                    self.stagnation_count += 1
                self.vlg.add_stat_to_current("statuses", Status.Unknown)
                self.vlg.add_stat_to_current("best", self.best_result)
                self.vlg.add_stat_to_current("current", self.current_result)
                return
            incumbent = sol_object.ObjectiveValue()
        else:
            # solver didn't find any solution
            if sol_object is None or sol_object.get_solve_status() in [
                "Unknown",
                "Infeasible",
            ]:
                if self.experiment_config["improvement_constr"]:
                    self.stagnation_count += 1
                self.vlg.add_stat_to_current("statuses", Status.Unknown)
                self.vlg.add_stat_to_current("best", self.best_result)
                self.vlg.add_stat_to_current("current", self.current_result)
                return
            solve_status = sol_object.get_solve_status()
            incumbent = sol_object.get_objective_value()

        if incumbent < self.current_result:
            # result is better than current: auto-accept
            accept = True

            if incumbent < self.best_result:
                if self.experiment_config["backend"] == "ortools":
                    status = (
                        Status.Optimal_New_Best
                        if solve_status == cp_model.OPTIMAL
                        else Status.Feasible_New_Best
                    )
                    self.best_solution = ORToolsSolution(sol_object, self.handles)
                else:
                    status = (
                        Status.Optimal_New_Best
                        if solve_status == "Optimal"
                        else Status.Feasible_New_Best
                    )
                    self.best_solution = sol_object

                # update best-so-far
                self.best_result = incumbent
            else:
                if self.experiment_config["backend"] == "ortools":
                    status = (
                        Status.Optimal_Improve
                        if solve_status == cp_model.OPTIMAL
                        else Status.Feasible_Improve
                    )
                else:
                    status = (
                        Status.Optimal_Improve
                        if solve_status == "Optimal"
                        else Status.Feasible_Improve
                    )
        else:
            # result is worse or equal to current: use eps
            accept = random.random() < self.experiment_config["eps_greedy_prob"]

            if incumbent > self.current_result + 1e-6:
                # it shouldn't be possible to get a worse result if the solver achieved optimum
                # if self.experiment_config["backend"] != "ortools":
                    # assert solve_status == "Feasible"
                status = Status.Feasible_Degradation

            elif incumbent >= self.current_result:
                if self.experiment_config["backend"] == "ortools":
                    status = (
                        Status.Optimal_No_Improve
                        if solve_status == cp_model.OPTIMAL
                        else Status.Feasible_No_Improve
                    )
                else:
                    status = (
                        Status.Optimal_No_Improve
                        if solve_status == "Optimal"
                        else Status.Feasible_No_Improve
                    )

        # increase stagnation count if solver achieved optimum but stagnated
        if status in [
            Status.Optimal_No_Improve,
            Status.Optimal_Improve,
        ]:
            self.stagnation_count += 1
        # reset stagnation count if new best was achieved
        elif status in [Status.Optimal_New_Best, Status.Feasible_New_Best]:
            self.stagnation_count = 0

        # eps greedy acceptance
        if accept:
            self.current_result = int(incumbent)
            if self.experiment_config["backend"] == "ortools":
                self.current_solution = ORToolsSolution(sol_object, self.handles)
            else:
                self.current_solution = sol_object

        # log current results and status
        self.vlg.add_stat_to_current("statuses", status)
        self.vlg.add_stat_to_current("best", self.best_result)
        self.vlg.add_stat_to_current("current", self.current_result)

        return status

    def get_freeze_sets(self, to_optimize, old_freeze_constraints, strategy_results):
        if self.experiment_config["delta_freezing"]:
            to_add, to_remove = self.freeze_manager.apply_delta_freezing(
                self.current_solution,
                to_optimize,
                self.handles.get("all_intervals_flat"),
            )
            freeze_constraints = self.freeze_manager.active_constraints
            logger.info(
                f"Delta Freezing: +{len(to_add)} constraints, -{len(to_remove)} constraints. Frozen: {len(freeze_constraints)}"
            )
        else:
            to_add = generate_freeze_constraints(
                self.mdl,
                self.current_solution,
                self.handles,
                strategy_results,
            )
            to_remove = old_freeze_constraints
            freeze_constraints = to_add
        return to_add, to_remove, freeze_constraints

    def get_starting_point(self, to_optimize):
        if self.experiment_config["backend"] in ("ortools", "cpp"):
            return None

        if self.experiment_config["starting_point_for_frozen"]:
            starting_point = self.current_solution
            if isinstance(self.current_solution, CpoSolveResult):
                starting_point = starting_point.get_solution()
        else:
            starting_point = create_partial_starting_point(
                self.mdl, self.current_solution, to_optimize
            )
        return starting_point

    def run_solver(self):
        # run reset logic
        if self.experiment_config["backend"] == "ortools":
            self.mdl, self.handles, sp = prepare_model_ortools(
                self.instance_config, self.instance, self.heur_sol
            )
        elif self.experiment_config["backend"] == "docplex":
            self.mdl, self.handles, sp = prepare_model_docplex(
                self.instance_config, self.instance, self.heur_sol
            )
        elif self.experiment_config["backend"] == "cpp":
            
            self.mdl, self.handles, sp = prepare_model_cpp(
                self.instance_config, self.instance, self.heur_sol
            )
        
        if self.experiment_config["delta_freezing"] and self.experiment_config["backend"] != "cpp":
            self.freeze_manager = FreezeManager(
                self.mdl,
                self.experiment_config["backend"],
                self.instance_config["horizon"],
            )

        self.weighted_jaccard_matrix, _ = build_similarity_matrix(
            self.handles["O"],
            self.handles["orders_req"],
            self.handles["rt"],
            self.handles["rt_return"],
            normalize=False,
        )
        strategies = [
            lambda sev: (
                strategy_random_orders(
                    self.handles, self.current_solution, p=0.1 * sev
                ),
                "random_orders",
            ),
            lambda sev: (
                strategy_similar_orders(
                    self.handles,
                    self.current_solution,
                    0.1 * sev,
                    self.weighted_jaccard_matrix,
                ),
                "similar_orders",
            ),
            # lambda sev: (
            #     strategy_random_skus(self.handles, self.current_solution, p=0.1 * sev),
            #     "random_skus",
            # ),
            # lambda sev: (
            #     combine_strategies(
            #         strategy_random_orders(
            #             self.handles, self.current_solution, p=0.05 * sev
            #         ),
            #         strategy_random_skus(
            #             self.handles, self.current_solution, p=0.05 * sev
            #         ),
            #     ),
            #     "random_orders_and_skus",
            # ),
            # lambda sev: (
            #     strategy_random_lanes(self.handles, self.current_solution, 1),
            #     "random_lanes",
            # ),
            # lambda sev: (
            #     strategy_single_timeslice(
            #         self.handles, self.current_solution, 100 * sev
            #     ),
            #     "single_timeslice",
            # ),
            # lambda sev: (
            #     strategy_multi_timeslice(
            #         self.handles, self.current_solution, 2, 100 * sev
            #     ),
            #     "double_timeslice",
            # ),
            # lambda sev: (
            #     strategy_multi_timeslice(
            #         self.handles, self.current_solution, 3, 100 * sev
            #     ),
            #     "triple_timeslice",
            # ),
        ]

        # prepare variable index sorted look-up for unfrozen set logging
        var_to_idx, num_variables = sort_variables(
            self.mdl, sp, self.experiment_config["backend"], self.handles
        )

        self.severity = 1
        self.stagnation_count = 0
        self.best_result = self.heur_sol.makespan
        self.best_solution = sp
        self.current_solution = self.best_solution
        self.current_result = self.best_result

        self.obj_constraint = None
        self.cpp_frozen_vars = set()

        old_freeze_constraints = []
        lg = LoguruStream()
        # start iterations
        self.vlg.log_run_start(num_variables, var_to_idx, sp)
        for i in range(self.experiment_config["iters"]):
            
            self.vlg.log_iteration_start(i)
            if (
                self.stagnation_count >= self.experiment_config["stagnation_th"]
                and self.severity < self.experiment_config["max_severity"]
            ):
                self.severity += self.experiment_config["severity_step"]
                self.vlg.add_stat_to_current("severity_increases", 1)
                self.stagnation_count = 0

            self.vlg.time_from_here()
            strat_idx = random.randint(0, len(strategies) - 1)
            strategy_results, strat_name = strategies[strat_idx](self.severity)
            self.vlg.log_strategy(strat_name, strat_idx)

            # prepare model for next solve (OR-Tools only, must happen before get_optimized_set)
            if self.experiment_config["backend"] == "ortools":
                # For OR-Tools, rebuild the model from scratch to avoid protobuf corruption
                import ortools_cp_model

                horizon_val = self.instance_config.get("horizon", 0)
                self.mdl, self.handles = ortools_cp_model.build_model(
                    self.instance,
                    rt_return=self.instance.rt_ret,
                    add_symmetry_breaking=self.instance_config["symmetry_breaking"],
                    horizon=horizon_val,
                    move_cap=self.instance_config.get("movecap", None),
                )

                # We need to recreate the FreezeManager because it holds a reference to the old model
                self.freeze_manager = FreezeManager(
                    self.mdl, backend="ortools", horizon=horizon_val
                )

            # create to_optimize set with graph traversal
            to_optimize = get_optimized_set(
                self.handles, self.current_solution, strategy_results
            )

            # generate freeze constraints from the to_optimize set
            if self.experiment_config["backend"] == "cpp":
                to_optimize_names = {v.get_name() for v in to_optimize}
                
                frozen_names, frozen_p, frozen_s, frozen_e = [], [], [], []
                warm_names, warm_p, warm_s, warm_e = [], [], [], []
                unfrozen_names = []
                new_frozen_vars = set()
                
                for name, vs in self.current_solution.var_sols.items():
                    p = 1 if vs.get("present") else 0
                    s = vs.get("start", 0) if p else 0
                    e = vs.get("end", 0) if p else 0
                    
                    if name in to_optimize_names:
                        warm_names.append(name)
                        warm_p.append(p)
                        warm_s.append(s)
                        warm_e.append(e)
                    
                    if name not in to_optimize_names:
                        new_frozen_vars.add(name)
                        if name not in self.cpp_frozen_vars:
                            frozen_names.append(name)
                            frozen_p.append(p)
                            frozen_s.append(s)
                            frozen_e.append(e)

                for name in self.cpp_frozen_vars:
                    if name not in new_frozen_vars:
                        unfrozen_names.append(name)
                        
                self.cpp_frozen_vars = new_frozen_vars

                self.vlg.add_stat_to_current(
                "constr_generate_time", self.vlg.time_diff_with_overwrite()
                )

                self.mdl.apply_delta_freezing(frozen_names, frozen_p, frozen_s, frozen_e, unfrozen_names)
                self.mdl.apply_warm_start(warm_names, warm_p, warm_s, warm_e)
                
                to_add, to_remove, freeze_constraints = [], [], []
            else:
                to_add, to_remove, freeze_constraints = self.get_freeze_sets(
                    to_optimize, old_freeze_constraints, strategy_results
                )

                self.vlg.add_stat_to_current(
                    "constr_generate_time", self.vlg.time_diff_with_overwrite()
                )

            # create starting point for the next iteration
            starting_point = self.get_starting_point(to_optimize)

            if self.experiment_config["backend"] == "ortools":
                # Apply freeze sets as constraints directly to the new model
                self.freeze_manager.apply_delta_freezing(
                    self.current_solution,
                    to_optimize,
                    self.handles["all_intervals_flat"],
                )

                apply_partial_starting_point(
                    self.mdl, self.current_solution, to_optimize
                )
                self.vlg.log_freeze_constr(to_optimize, var_to_idx, strat_idx)

                self.vlg.time_from_here()
                if not hasattr(self, "solver"):
                    from ortools.sat.python import cp_model

                    self.solver = cp_model.CpSolver()
                    self.solver.parameters.max_time_in_seconds = self.experiment_config[
                        "iter_time_limit"
                    ]
                    # self.solver.parameters.num_search_workers = 1
                    # self.solver.parameters.cp_model_presolve = False
                    # self.solver.parameters.linearization_level = 0
                    # self.solver.parameters.search_branching = cp_model.FIXED_SEARCH

                # Force improvement
                # if self.experiment_config["improvement_constr"]:
                #     self.mdl.add(
                #         self.handles["makespan_var"] <= self.current_result - 1
                #     )

                status = self.solver.Solve(self.mdl)
                self.vlg.log_solve_time(self.solver, status)
                self.eps_greedy_acceptance(self.solver, status)
            elif self.experiment_config["backend"] == "docplex":
                # set iteration starting point
                self.mdl.set_starting_point(starting_point)
                # add and remove freeze constraints
                self.mdl.remove(to_remove)
                self.mdl.add(to_add)
                old_freeze_constraints = freeze_constraints
                self.vlg.log_freeze_constr(to_optimize, var_to_idx, strat_idx)

                # Force improvement
                if self.experiment_config["improvement_constr"]:
                    if self.obj_constraint is not None:
                        self.mdl.remove(self.obj_constraint)
                    self.obj_constraint = self.mdl.add(
                        self.handles["makespan"] <= self.current_result - 1
                    )

                # run solve on partially frozen model
                self.vlg.time_from_here()
                sol = self.mdl.solve(
                    TimeLimit=self.experiment_config["iter_time_limit"],
                    LogVerbosity="Quiet",
                    Workers=self.experiment_config["workers"],
                    Presolve=self.experiment_config["presolve"],
                    SearchType=self.experiment_config["search_type"],
                    # log_output=lg,
                )
                self.vlg.log_solve_time(sol)
            elif self.experiment_config["backend"] == "cpp":
                self.vlg.log_freeze_constr(to_optimize, var_to_idx, strat_idx)
                
                impr_makespan = -1
                if self.experiment_config["improvement_constr"]:
                    impr_makespan = self.current_result - 1
                    
                self.vlg.time_from_here()
                result_dict = self.mdl.solve(
                    self.experiment_config["iter_time_limit"],
                    impr_makespan,
                    self.experiment_config["workers"],
                    self.experiment_config["presolve"],
                    self.experiment_config["search_type"]
                )
                sol = cpp_cp_model.CppSolveResult(result_dict)
                self.vlg.log_solve_time(sol)
                self.eps_greedy_acceptance(sol, sol.get_solve_status())

            # run eps-greedy acceptance (already done for ortools and cpp)
            if self.experiment_config["backend"] == "docplex":
                self.eps_greedy_acceptance(sol, sol.get_solve_status())

            # log iteration end in VisualLogger
            self.vlg.log_iteration()

        # End of LNS loop: Validate the best solution found
        logger.info("Converting best solution for validation...")
        final_solution = cp_sol_to_solution(self.best_solution, self.handles)

        logger.info("Validating final solution...")
        try:
            violations = validate_solution(
                final_solution,
                self.instance,
                horizon=self.instance_config["horizon"],
                move_cap=self.instance_config["movecap"],
            )
        except KeyError:
            logger.error("Config doesn't have horizon or movecap field.")

        if violations:
            logger.error(f"Validation FAILED! {len(violations)} violations found:")
            for v in violations[:10]:
                logger.error(f" - {v}")
        else:
            logger.info("Validation PASSED successfully.")

        self.vlg.log_run_end(self.best_solution, self.handles)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run matheuristic experiments.")
    parser.add_argument(
        "instance_name",
        type=str,
        help="Name of the precalculated instance folder under precalculated_instances/",
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="docplex",
        help="Optimizer for the reparing CP model. Options: docplex, ortools, cpp",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of runs (default: 10)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=300,
        help="Number of iterations per run (default: 300)",
    )
    parser.add_argument(
        "--stagnation-th",
        type=int,
        default=50,
        dest="stagnation_th",
        help="Stagnation threshold (default: 50)",
    )
    parser.add_argument(
        "--severity-step",
        type=int,
        default=1,
        dest="severity_step",
        help="Severity increment step (default: 1)",
    )
    parser.add_argument(
        "--max-severity",
        type=int,
        default=5,
        dest="max_severity",
        help="Upper bound for severity (default: 5)",
    )
    parser.add_argument(
        "--iter-time-limit",
        type=float,
        default=2.0,
        dest="iter_time_limit",
        help="Per-iteration time limit in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of workers for CP Optimizer",
    )
    parser.add_argument(
        "--presolve",
        type=str,
        default="Auto",
        choices=["Auto", "On", "Off"],
        help="Presolve setting for CP Optimizer",
    )
    parser.add_argument(
        "--search-type",
        type=str,
        default="Auto",
        choices=["Auto", "DepthFirst", "Restart", "MultiPoint"],
        help="Search type for CP Optimizer",
    )

    parser.add_argument(
        "--delta-freezing",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="delta_freezing",
        help="Use delta freezing for updating the freeze constraints (default: True)",
    )

    parser.add_argument(
        "--improvement-constr",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="improvement_constr",
        help="Add improvement of the makespan over current as a new constraint in each iteration (default: True)",
    )

    parser.add_argument(
        "--starting-point-for-frozen",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="starting_point_for_frozen",
        help="Initialize the starting point for frozen variables too (default: True)",
    )

    parser.add_argument(
        "--eps-greedy-prob",
        type=float,
        default=0.2,
        dest="eps_greedy_prob",
        help="Epsilon-greedy acceptance probability (default: 0.2)",
    )
    args = parser.parse_args()

    experiment_config = {
        "runs": args.runs,
        "iters": args.iters,
        "stagnation_th": args.stagnation_th,
        "severity_step": args.severity_step,
        "max_severity": args.max_severity,
        "iter_time_limit": args.iter_time_limit,
        "delta_freezing": args.delta_freezing,
        "starting_point_for_frozen": args.starting_point_for_frozen,
        "eps_greedy_prob": args.eps_greedy_prob,
        "backend": args.backend,
        "improvement_constr": args.improvement_constr,
        "presolve": args.presolve,
        "workers": args.workers,
        "search_type": args.search_type
    }
    solver = Solver(
        experiment_config,
        instance_path=f"precalculated_instances/{args.instance_name}",
    )
