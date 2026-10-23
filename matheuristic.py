import argparse
import json
import os
import pickle
import random
import time

import tqdm
from docplex.cp.solution import CpoSolveResult
from docplex.cp.solver.cpo_callback import EVENT_SOLUTION, CpoCallback
from loguru import logger

import cp_model as docplex_model
import cpp_cp_model
from autostore_heuristic import validate_solution
from cp_model_utils import (
    CppSolveResult,
    cp_sol_to_solution,
    sort_variables,
)
from freezing_utils import (
    FreezeManager,
    create_partial_starting_point,
    generate_freeze_constraints,
    get_optimized_set,
)
from instance import Instance
from jaccard_similarity import build_similarity_matrix
from matheuristic_plots import Status, VisualLogger
from neighborhood_selection import (
    StrategyManager,
)
from precalculate_instances_for_lns import check_if_precalculated, precalculate_config

random.seed(42)


class IterationProgressCallback(CpoCallback):
    """Records every improving solution found while solving a subproblem.

    Used with the docplex backend to capture, per iteration, *when* each
    (strictly improving) solution was discovered and the KPIs it achieved. This
    information drives the choice of the iteration time limit and a target on the
    number of discovered solutions. The callback must be ``reset`` before each
    solve.

    Records look like::

        {"time": <seconds since solve start>, "best": <objective>, "kpis": {<kpi>: <value>}}
    """

    def __init__(self):
        super().__init__()
        self.records = []
        self.best_obj = None
        self._start_time = None

    def reset(self):
        self.records = []
        self.best_obj = None
        self._start_time = time.perf_counter()

    def invoke(self, solver, event, sres):
        if event != EVENT_SOLUTION:
            return
        try:
            if not sres.is_solution():
                return
            obj = sres.get_objective_value()
        except Exception:
            return

        if obj is None:
            return
        # Only keep strict improvements (matches the C++ progress collector).
        if self.best_obj is not None and obj >= self.best_obj:
            return

        if self._start_time is None:
            self._start_time = time.perf_counter()

        # Prefer the engine solve time (same basis as the C++ progress records),
        # falling back to elapsed wall time if unavailable.
        try:
            discovery_time = float(sres.get_solve_time())
        except (TypeError, ValueError, AttributeError):
            discovery_time = time.perf_counter() - self._start_time

        self.best_obj = obj
        self.records.append(
            {
                "time": discovery_time,
                "best": obj,
                "kpis": dict(sres.get_kpis() or {}),
            }
        )


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

    # with open(f"{instance_folder}/cp_intermediate_records.pkl", "rb") as f:
    # cp_records = pickle.load(f)
    return instance, heur_sol, instance_folder, instance_config


def prepare_model_docplex(config, instance, heur_sol, objective_func):

    mdl, handles = docplex_model.build_model(
        instance,
        add_symmetry_breaking=config["symmetry_breaking"],
        horizon=config["horizon"],
        objective_func=objective_func,
    )

    starting_point = docplex_model.inject_warmstart(
        heur_sol, heur_sol.pick_events, mdl, handles
    )
    return mdl, handles, starting_point


class MockCpoStartingPoint:
    def __init__(self):
        self.state_dict = {}

    def add_interval_var_solution(
        self, var, presence=False, start=None, end=None, size=None
    ):
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


def prepare_model_cpp(config, instance, heur_sol, objective_func):
    mdl, handles = cpp_cp_model.build_model(
        instance,
        add_symmetry_breaking=config["symmetry_breaking"],
        horizon=config["horizon"],
        objective_func=objective_func,
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
    return (
        mdl,
        handles,
        CppSolveResult(
            {
                "status": "Feasible",
                "objective": heur_sol.makespan,
                "var_solutions": mock_sp.state_dict,
                "kpis": {
                    "makespan": heur_sol.makespan,
                    "bin_fetches": sum(
                        len(events) for events in heur_sol.bin_events.values()
                    ),
                    "total_flow_time": sum(
                        end for s, ln, start, end in heur_sol.order_assignments.values()
                    ),
                },
            }
        ),
    )


class Solver:
    def __init__(self, experiment_config, instance_config=None, instance_path=None):
        (
            self.instance,
            self.heur_sol,
            # self.cp_records,
            instance_path,
            instance_config,
        ) = load_instance(instance_config, instance_path)
        self.instance_config = instance_config
        self.vlg = VisualLogger(
            instance_path, instance_config, experiment_config, self.instance
        )
        self.severity = 1
        self.stagnation_count = 0
        self.experiment_config = experiment_config
        self.cycling_schedule = []
        for phase in self.experiment_config["cp_objective_phase_schedule"]:
            if phase == "m":
                self.cycling_schedule.append("makespan")
            elif phase == "f":
                self.cycling_schedule.append("total_flow_time")
            elif phase == "b":
                self.cycling_schedule.append("bin_fetches")
            elif phase == "i":
                self.cycling_schedule.append("idle_time")
            elif phase == "p":
                self.cycling_schedule.append("max_picks")
            elif phase == "w":
                self.cycling_schedule.append("total_wages")
            else:
                raise ValueError(phase)

        self.cp_obj = experiment_config["cp_objective"]
        self.cycle_ix = self.cycling_schedule.index(self.cp_obj)
        for run in range(self.experiment_config["runs"]):
            self.run_solver()
        self.vlg.save_experiment()

    def update_iteration_objective(self, mdl, target="makespan"):
        """Swaps the active objective function for the current LNS iteration."""
        if target == self.cp_obj:
            return  # already have the correct objective

        if self.experiment_config["backend"] == "docplex":
            # 1. Safely remove the current objective from the model
            current_obj = mdl.get_objective()
            if current_obj is not None:
                mdl.remove(current_obj)

            # 2. Inject the new target (assuming makespan_expr and fetches_expr are saved as class attributes)
            logger.info(
                f"Changing objective to '{target}', cycle index {self.cycle_ix}"
            )
            if target == "makespan":
                mdl.minimize(self.handles["makespan"])
                self.cp_obj = "makespan"
            elif target == "bin_fetches":
                mdl.minimize(self.handles["bin_fetches"])
                self.cp_obj = "bin_fetches"
            elif target == "total_flow_time":
                mdl.minimize(self.handles["total_flow_time"])
                self.cp_obj = "total_flow_time"
            elif target == "idle_time":
                mdl.minimize(self.handles["idle_time"])
                self.cp_obj = "idle_time"
            elif target == "max_picks":
                mdl.minimize(self.handles["max_picks"])
                self.cp_obj = "max_picks"
            else:
                raise ValueError(f"Unknown objective target: {target}")
        elif self.experiment_config["backend"] == "cpp":
            logger.info(
                f"Changing objective to '{target}', cycle index {self.cycle_ix}"
            )
            mdl.set_objective(target)
            self.cp_obj = target

    def rrt_acceptance(self, sol_object, solve_status=None):
        """
        Perform Record-to-Record Travel (RRT) acceptance on incumbent solution:
            - Accept solution if its makespan is at most (1 + rrt_threshold) * best_makespan

        Handles best/current solution updates, stagnation count and logging to VLG
        """

        # solver didn't find any solution
        if sol_object.get_solve_status() == "Infeasible":
            if self.experiment_config["improvement_constr"]:
                self.stagnation_count += 1
            self.vlg.add_stat_to_current("statuses", Status.Infeasible)
            self.vlg.add_stat_to_current("best", self.best_result)
            self.vlg.add_stat_to_current("current", self.current_result)
            self._log_kpis()
            return Status.Infeasible

        if sol_object is None or sol_object.get_solve_status() == "Unknown":
            self.vlg.add_stat_to_current("statuses", Status.Unknown)
            self.vlg.add_stat_to_current("best", self.best_result)
            self.vlg.add_stat_to_current("current", self.current_result)
            self._log_kpis()
            return Status.Unknown

        solve_status = sol_object.get_solve_status()
        current_makespan = sol_object.get_kpis()["makespan"]
        incumbent = current_makespan

        if incumbent < self.current_result:
            # result is better than current: auto-accept
            accept = True

            if incumbent < self.best_result:
                status = (
                    Status.Optimal_New_Best
                    if solve_status == "Optimal"
                    else Status.Feasible_New_Best
                )
                self.best_solution = sol_object
                self.best_result = incumbent
            else:
                status = (
                    Status.Optimal_Improve
                    if solve_status == "Optimal"
                    else Status.Feasible_Improve
                )
        else:
            # Check RRT threshold: accept if within threshold of best_result
            rrt_limit = self.best_result * (1 + self.experiment_config["rrt_threshold"])
            accept = incumbent <= rrt_limit

            if incumbent > self.current_result + 1e-6:
                if solve_status != "Feasible" and self.cp_obj == "makespan":
                    raise AssertionError
                status = Status.Feasible_Degradation
            elif incumbent >= self.current_result:
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
            self.current_solution = sol_object

        # log current results and status
        self.vlg.add_stat_to_current("statuses", status)
        self.vlg.add_stat_to_current("best", self.best_result)
        self.vlg.add_stat_to_current("current", self.current_result)
        self._log_kpis()

        return status

    def _log_kpis(self):
        cur_kpis = (
            self.current_solution.get_kpis()
            if getattr(self, "current_solution", None)
            else {}
        )
        best_kpis = (
            self.best_solution.get_kpis()
            if getattr(self, "best_solution", None)
            else {}
        )

        curr_fetches = cur_kpis.get("bin_fetches", -1)
        curr_flow = cur_kpis.get("total_flow_time", -1)
        curr_idle = cur_kpis.get("idle_time", -1)
        curr_picks = cur_kpis.get("max_picks", -1)

        if curr_fetches != -1 and (
            self.best_bin_fetches == -1 or curr_fetches < self.best_bin_fetches
        ):
            self.best_bin_fetches = curr_fetches

        if curr_flow != -1 and (
            getattr(self, "best_total_flow_time", -1) == -1
            or curr_flow < self.best_total_flow_time
        ):
            self.best_total_flow_time = curr_flow

        if curr_idle != -1 and (
            getattr(self, "best_idle_time", -1) == -1
            or curr_idle < getattr(self, "best_idle_time", -1)
        ):
            self.best_idle_time = curr_idle

        if curr_picks != -1 and (
            getattr(self, "best_max_picks", -1) == -1
            or curr_picks < getattr(self, "best_max_picks", -1)
        ):
            self.best_max_picks = curr_picks

        self.vlg.add_stat_to_current(
            "current_makespan", cur_kpis.get("makespan", self.current_result)
        )
        self.vlg.add_stat_to_current("current_bin_fetches", curr_fetches)
        self.vlg.add_stat_to_current("current_total_flow_time", curr_flow)
        self.vlg.add_stat_to_current("current_idle_time", curr_idle)
        self.vlg.add_stat_to_current("current_max_picks", curr_picks)

        self.vlg.add_stat_to_current(
            "best_makespan", best_kpis.get("makespan", self.best_result)
        )
        self.vlg.add_stat_to_current("best_bin_fetches", self.best_bin_fetches)
        self.vlg.add_stat_to_current(
            "best_total_flow_time", getattr(self, "best_total_flow_time", -1)
        )
        self.vlg.add_stat_to_current(
            "best_idle_time", getattr(self, "best_idle_time", -1)
        )
        self.vlg.add_stat_to_current(
            "best_max_picks", getattr(self, "best_max_picks", -1)
        )
        self.vlg.add_stat_to_current("cp_obj", self.cp_obj)

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
        if self.experiment_config["backend"] == "cpp":
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
        if self.experiment_config["backend"] == "docplex":
            self.mdl, self.handles, sp = prepare_model_docplex(
                self.instance_config,
                self.instance,
                self.heur_sol,
                self.experiment_config["cp_objective"],
            )
        elif self.experiment_config["backend"] == "cpp":
            self.mdl, self.handles, sp = prepare_model_cpp(
                self.instance_config,
                self.instance,
                self.heur_sol,
                self.experiment_config["cp_objective"],
            )

        if (
            self.experiment_config["delta_freezing"]
            and self.experiment_config["backend"] != "cpp"
        ):
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

        strategy_manager = StrategyManager(
            self.handles,
            sp,
            self.vlg,
            self.weighted_jaccard_matrix,
            self.experiment_config["strat_preset"],
            self.experiment_config["strat_choice"],
            default_percent=self.experiment_config["default_percent"],
            default_n_lanes=self.experiment_config["default_n_lanes"],
            default_timeslice_width=self.experiment_config["default_timeslice_width"],
        )

        # prepare variable index sorted look-up for unfrozen set logging
        var_to_idx, num_variables = sort_variables(
            self.mdl, sp, self.experiment_config["backend"], self.handles
        )

        # Listener/callback used to capture every solution discovered per subproblem.
        if self.experiment_config["backend"] == "docplex":
            self.progress_callback = IterationProgressCallback()
            self.mdl.add_solver_callback(self.progress_callback)
        else:
            self.progress_callback = None

        self.severity = 1
        self.stagnation_count = 0

        self.best_result = self.heur_sol.makespan
        self.best_bin_fetches = sum(
            len(events) for events in self.heur_sol.bin_events.values()
        )
        self.best_total_flow_time = sum(
            end for s, ln, start, end in self.heur_sol.order_assignments.values()
        )

        self.best_solution = sp
        self.current_solution = self.best_solution
        self.current_result = self.best_result

        self.obj_constraint = None
        self.cpp_frozen_vars = set()

        old_freeze_constraints = []

        # start iterations
        run_start_time = time.time()
        self.vlg.log_run_start(num_variables, var_to_idx, sp)

        n_iters = (
            self.experiment_config["iters"]
            if self.experiment_config["iters"] is not None
            else 999999
        )
        for i in range(n_iters):
            if self.experiment_config.get("time_limit") is not None:
                elapsed = time.time() - run_start_time
                if (
                    self.experiment_config["time_limit"] - elapsed
                    < self.experiment_config["iter_time_limit"]
                ):
                    logger.info(
                        f"Time limit reached ({elapsed:.1f}s / {self.experiment_config['time_limit']}s). Stopping iterations."
                    )
                    break

            self.vlg.log_iteration_start(i)
            if (
                self.stagnation_count >= self.experiment_config["stagnation_th"]
                and self.severity < self.experiment_config["max_severity"]
            ):
                self.severity += self.experiment_config["severity_step"]
                self.vlg.add_stat_to_current("severity_increases", 1)
                self.stagnation_count = 0

            self.vlg.time_from_here()
            strat_idx, strat_name, strat_lambda = strategy_manager.choose_strat()
            strategy_results = strat_lambda(self.severity)
            self.vlg.log_strategy(strat_name, strat_idx)

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

                self.mdl.apply_delta_freezing(
                    frozen_names, frozen_p, frozen_s, frozen_e, unfrozen_names
                )
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

            cycle_per = self.experiment_config["cp_objective_phase_duration"]
            if cycle_per > 0 and (i + 1) % cycle_per == 0:
                self.cycle_ix = (self.cycle_ix + 1) % len(self.cycling_schedule)
                next_obj = self.cycling_schedule[self.cycle_ix]
                self.update_iteration_objective(self.mdl, next_obj)

            impr_bound = -1
            if self.experiment_config["improvement_constr"]:
                if self.cp_obj == "makespan":
                    impr_bound = int(self.current_result - 1)
                elif (
                    self.cp_obj == "bin_fetches"
                    and getattr(self, "best_bin_fetches", -1) != -1
                ):
                    impr_bound = int(self.best_bin_fetches - 1)
                elif (
                    self.cp_obj == "total_flow_time"
                    and getattr(self, "best_total_flow_time", -1) != -1
                ):
                    impr_bound = int(self.best_total_flow_time - 1)
                elif (
                    self.cp_obj == "idle_time"
                    and getattr(self, "best_idle_time", -1) != -1
                ):
                    impr_bound = int(self.best_idle_time - 1)
                elif (
                    self.cp_obj == "max_picks"
                    and getattr(self, "best_max_picks", -1) != -1
                ):
                    impr_bound = int(self.best_max_picks - 1)

            if self.experiment_config["backend"] == "docplex":
                # set iteration starting point
                self.mdl.set_starting_point(starting_point)
                # add and remove freeze constraints
                self.mdl.remove(to_remove)
                self.mdl.add(to_add)
                old_freeze_constraints = freeze_constraints
                self.vlg.log_freeze_constr(to_optimize, var_to_idx, strat_idx)

                # Force improvement
                if getattr(self, "obj_constraint", None) is not None:
                    self.mdl.remove(self.obj_constraint)
                    self.obj_constraint = None

                if impr_bound != -1:
                    self.obj_constraint = self.mdl.add(
                        self.handles[self.cp_obj] <= impr_bound
                    )

                # run solve on partially frozen model
                self.vlg.time_from_here()
                solve_kwargs = {
                    "TimeLimit": self.experiment_config["iter_time_limit"],
                    "LogVerbosity": "Quiet",
                    "Workers": self.experiment_config["workers"],
                    "Presolve": self.experiment_config["presolve"],
                    "SearchType": self.experiment_config["search_type"],
                }
                if self.experiment_config.get("only_first_sol", False):
                    solve_kwargs["SolutionLimit"] = 1

                # Reset before solving so we only keep solutions from this subproblem.
                self.progress_callback.reset()
                sol = self.mdl.solve(**solve_kwargs)
                progress = self.progress_callback.records
            elif self.experiment_config["backend"] == "cpp":
                self.vlg.log_freeze_constr(to_optimize, var_to_idx, strat_idx)

                self.vlg.time_from_here()
                result_dict = self.mdl.solve(
                    self.experiment_config["iter_time_limit"],
                    impr_bound,
                    self.experiment_config["workers"],
                    self.experiment_config["presolve"],
                    self.experiment_config["search_type"],
                    self.experiment_config.get("only_first_sol", False),
                    collect_progress=True,
                    collect_deltas=False,
                )
                sol = CppSolveResult(result_dict)
                progress = result_dict.get("progress") or []
            else:
                raise ValueError(self.experiment_config["backend"])

            # log solution and run eps-greedy
            self.vlg.log_solve_time(sol)
            self.vlg.log_discovered_solutions(progress)
            prev_solution = self.current_solution
            status = self.rrt_acceptance(sol, sol.get_solve_status())

            # log to strategy manager
            strategy_manager.strat_statistics[strat_name]["num_uses"] += 1
            strategy_manager.strat_statistics[strat_name]["status"][status] += 1
            strategy_manager.current_solution = self.current_solution

            # log iteration end in VisualLogger
            self.vlg.log_solution_delta(
                prev_solution, self.current_solution, to_optimize
            )
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
        "instances",
        nargs="+",
        type=str,
        help="Name of the precalculated instance folder under precalculated_instances/",
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="docplex",
        help="Optimizer for the reparing CP model. Options: docplex, cpp",
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
        default=None,
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
        "--severity-backup",
        type=str,
        default="none",
        choices=["none", "new_best", "status"],
        help="Presolve setting for CP Optimizer",
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
        default="On",
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
        "--strat-preset",
        type=str,
        default="all",
        choices=["all", "shaw_random", "random_balance", "random_orders"],
        help="Which strategies are used during optimization (presets defined in neighborhood_selection.py)",
    )

    parser.add_argument(
        "--strat-choice",
        type=str,
        default="uniform",
        choices=["uniform", "adaptive"],
        help="How strategies are chosen in each iteration.",
    )

    parser.add_argument(
        "--cp-objective",
        type=str,
        default="makespan",
        choices=[
            "makespan",
            "bin_fetches",
            "total_flow_time",
            "idle_time",
            "max_picks",
        ],
        help="Objective function for the underlying CP model.",
    )

    parser.add_argument(
        "--cp-objective-phase-duration",
        type=int,
        default="0",
        help="Every how many iterations to switch the CP objective (0 means no cycling)",
    )

    parser.add_argument(
        "--cp-objective-phase-schedule",
        type=str,
        default="m",
        help="Manual design of the objective schedule. Use a string like 'mmfmb' where 'm' is a makespan phase, 'f' is a flow time phase, 'b' is a number of bin fetches phase, 'i' is an idle time phase, and 'p' is a max picks phase.",
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
        "--render-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="render_video",
        help="Render a Manim video at the end of each run",
    )

    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        dest="time_limit",
        help="Hard time limit in seconds to override the iteration cap (default: None)",
    )

    parser.add_argument(
        "--rrt-threshold",
        type=float,
        default=0.05,
        dest="rrt_threshold",
        help="Record-to-Record Travel threshold percentage (default: 0.05 means accept if within 5% of best)",
    )

    parser.add_argument(
        "--default-timeslice-width",
        type=int,
        default=100,
        dest="default_timeslice_width",
        help="Default width of window for timeslice strategies",
    )

    parser.add_argument(
        "--default-percent",
        type=float,
        default=0.1,
        dest="default_percent",
        help="Default percentage for unfreezing of orders/sku",
    )

    parser.add_argument(
        "--default-n-lanes",
        type=int,
        default=2,
        dest="default_n_lanes",
        help="Default number of unfrozen lanes",
    )

    parser.add_argument(
        "--only-first-sol",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="only_first_sol",
        help="Exit iteration after finding the first solution (SolutionLimit=1)",
    )

    args = parser.parse_args()

    # Check termination conditions
    il_set, tl_set = args.iters is not None, args.time_limit is not None
    if (il_set and tl_set) or (not il_set and not tl_set):
        raise ValueError("Exactly one of '--iters' and '--time-limit' has to be set!")

    for instance in tqdm.tqdm(args.instances):
        if "precalculated_instances/" in instance:
            instance = instance.split("/")[1]

        experiment_config = {
            "runs": args.runs,
            "iters": args.iters,
            "stagnation_th": args.stagnation_th,
            "severity_step": args.severity_step,
            "severity_backup": args.severity_backup,
            "max_severity": args.max_severity,
            "iter_time_limit": args.iter_time_limit,
            "delta_freezing": args.delta_freezing,
            "starting_point_for_frozen": args.starting_point_for_frozen,
            "rrt_threshold": args.rrt_threshold,
            "time_limit": args.time_limit,
            "backend": args.backend,
            "improvement_constr": args.improvement_constr,
            "presolve": args.presolve,
            "workers": args.workers,
            "search_type": args.search_type,
            "strat_choice": args.strat_choice,
            "strat_preset": args.strat_preset,
            "default_timeslice_width": args.default_timeslice_width,
            "default_n_lanes": args.default_n_lanes,
            "default_percent": args.default_percent,
            "cp_objective": args.cp_objective,
            "cp_objective_phase_duration": args.cp_objective_phase_duration,
            "cp_objective_phase_schedule": args.cp_objective_phase_schedule,
            "render_video": args.render_video,
            "only_first_sol": args.only_first_sol,
        }
        solver = Solver(
            experiment_config,
            instance_path=f"precalculated_instances/{instance}",
        )
