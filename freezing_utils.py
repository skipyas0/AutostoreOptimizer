from loguru import logger


def get_assigned_station(handles, solution, order):
    """Finds which station an order is currently assigned to by checking I_os presence."""
    for s in handles["S"]:
        var = handles["I_os"].get((order, s))
        if var is not None:
            if hasattr(solution, "BooleanValue"):
                if solution.BooleanValue(var.pres):
                    return s
            else:
                var_sol = solution.get_var_solution(var)
                if var_sol is not None and var_sol.is_present():
                    return s
    return None


def is_bin_present_in_solution(handles, solution, station, sku, visit):
    """Checks if a specific bin cycle is active in the current solution."""
    var = handles["B"].get((station, sku, visit))
    if var is not None:
        if hasattr(solution, "BooleanValue"):
            return solution.BooleanValue(var.pres)
        else:
            var_sol = solution.get_var_solution(var)
            return var_sol is not None and var_sol.is_present()
    return False


def is_pick_present_in_solution(handles, solution, order, station, sku, visit):
    """Checks if a specific pick interval is active in the current solution."""
    var = handles["P"].get((order, station, sku, visit))
    if var is not None:
        if hasattr(solution, "BooleanValue"):
            return solution.BooleanValue(var.pres)
        else:
            var_sol = solution.get_var_solution(var)
            return var_sol is not None and var_sol.is_present()
    return False


def get_associated_indices(handles, solution, selection_result):
    """
    Crawls the solution graph to depth 1.
    Unfreezes the requested orders and their required bins, leaving all other
    intersecting orders frozen to act as temporal anchors.
    """
    active_o = set(selection_result.seed_orders)
    active_s = set(selection_result.seed_stations)
    active_k = set(selection_result.seed_skus)
    active_e = set()

    # Map SKUs to orders which need them and to stations these orders are mapped to:
    # Don't add to active_o now to avoid snowballing
    temp_o = set()
    for k in list(active_k):
        temp_o.update(handles["orders_demanding_sku"][k])
    for o in list(temp_o):
        s_assigned = get_assigned_station(handles, solution, o)
        if s_assigned is not None:
            active_s.add(s_assigned)

    # Map orders to their required SKUs and assigned stations
    for o in list(active_o):
        active_k.update(handles["orders_req"][o])
        s_assigned = get_assigned_station(handles, solution, o)
        if s_assigned is not None:
            active_s.add(s_assigned)

    # Add orders from first step now
    active_o.update(temp_o)

    # Map SKUs/Stations to specific Bin visits (e)
    for s in list(active_s):
        for k in list(active_k):
            added_absent = False
            for e in range(handles["U"].get(k, 0)):
                # If the bin is already in the schedule, unfreeze it
                if is_bin_present_in_solution(handles, solution, s, k, e):
                    active_e.add((s, k, e))

                # If it is absent, unfreeze it ONLY if we haven't added a spare yet
                elif not added_absent:
                    active_e.add((s, k, e))
                    added_absent = True
    return active_o, active_s, active_k, active_e


def unfreeze_order_attributes(handles, active_o, active_s, active_k):
    unfrozen = set()

    for o in active_o:
        for s in active_s:
            # Station and lane windows
            if (o, s) in handles["I_os"]:
                unfrozen.add(handles["I_os"][(o, s)])
            for ln in handles["L"]:
                if (o, s, ln) in handles["I_os_lane"]:
                    unfrozen.add(handles["I_os_lane"][(o, s, ln)])

            # Order consumptions
            for k in handles["orders_req"][o]:
                if (o, k, s) in handles["C"]:
                    unfrozen.add(handles["C"][(o, k, s)])
    return unfrozen


def unfreeze_bin_attributes(handles, active_o, active_e):
    unfrozen = set()

    for s, k, e in active_e:
        # Bin physical cycles
        unfrozen.add(handles["F"][(s, k, e)])
        unfrozen.add(handles["B"][(s, k, e)])
        unfrozen.add(handles["R"][(s, k, e)])
        unfrozen.add(handles["Block"][(s, k, e)])

        # Associated picks
        for o in active_o:
            if (o, s, k, e) in handles["P"]:
                unfrozen.add(handles["P"][(o, s, k, e)])

    return unfrozen


class FreezeManager:
    def __init__(self, mdl, backend="docplex", horizon=None):
        self.mdl = mdl
        self.backend = backend
        self.horizon = horizon
        # Maps a docplex variable object to its active freeze constraint objects
        self.active_constraints = {}
        self.active_frozen_vars = set()


    def apply_delta_freezing(self, solution, to_optimize, all_intervals_flat):
        if self.backend == "ortools":
            # For OR-Tools, we rebuild the model every iteration to avoid SWIG protobuf bugs.
            # Thus, we simply add constraints for all frozen variables.
            target_frozen_vars = set(all_intervals_flat.values()) - set(to_optimize)
            for var in target_frozen_vars:
                pres_val = solution.Value(var.pres)
                self.mdl.Add(var.pres == pres_val)

                if pres_val:
                    start_val = solution.Value(var.start)
                    self.mdl.Add(var.start == start_val)

                    end_val = solution.Value(var.end)
                    self.mdl.Add(var.end == end_val)
            
            return target_frozen_vars, []

        # 1. Identify the target state
        all_vars = {v.get_var() for v in solution.get_all_var_solutions()}
        target_frozen_vars = all_vars - to_optimize
        current_frozen_vars = set(self.active_constraints.keys())

        # 2. Calculate the Deltas using Set Math
        vars_to_unfreeze = current_frozen_vars - target_frozen_vars
        vars_to_newly_freeze = target_frozen_vars - current_frozen_vars

        # 3. Unfreeze (Remove old constraints)
        constraints_to_remove = []
        for var in vars_to_unfreeze:
            constraints_to_remove.extend(self.active_constraints[var])
            del self.active_constraints[var]

        # if constraints_to_remove:
        # self.mdl.remove(constraints_to_remove)

        # 4. Freeze (Generate and add new constraints)
        constraints_to_add = []
        new_mapping = {}

        for var in vars_to_newly_freeze:
            var_sol = solution.get_var_solution(var)
            c_list = []

            if "Interval" in var.type.name:
                if var_sol.is_present():
                    c_list.append(self.mdl.presence_of(var) == 1)
                    c_list.append(self.mdl.start_of(var) == var_sol.get_start())
                    c_list.append(self.mdl.end_of(var) == var_sol.get_end())
                else:
                    c_list.append(self.mdl.presence_of(var) == 0)
            elif any(t in var.type.name for t in ("Int", "Bool", "Float")):
                c_list.append(var == var_sol.get_value())

            new_mapping[var] = c_list
            constraints_to_add.extend(c_list)

        if constraints_to_add:
            # self.mdl.add(constraints_to_add)
            self.active_constraints.update(new_mapping)

        return constraints_to_add, constraints_to_remove


def generate_freeze_constraints(model, solution, handles, to_optimize):
    logger.debug(" -- resulted in {} variables for optimization", len(to_optimize))
    freeze_constraints = []
    for var_sol in solution.get_all_var_solutions():
        var = var_sol.get_var()

        # Check if the variable object itself is in the set
        if var in to_optimize:
            continue

        if "Interval" in var.type.name:
            # check if the interval is present in the current solution
            is_present = var_sol.is_present()

            if is_present:
                freeze_constraints.append(model.presence_of(var) == 1)
                freeze_constraints.append(model.start_of(var) == var_sol.get_start())
                freeze_constraints.append(model.end_of(var) == var_sol.get_end())
            else:
                freeze_constraints.append(model.presence_of(var) == 0)

        elif any(t in var.type.name for t in ("Int", "Bool", "Float")):
            val = var_sol.get_value()  # record assigned numeric value
            freeze_constraints.append(var == val)

    return freeze_constraints


def get_optimized_set(handles, solution, strategy_results):
    act_o, act_s, act_k, act_e = get_associated_indices(
        handles, solution, strategy_results
    )
    to_optimize = set()
    to_optimize.update(unfreeze_order_attributes(handles, act_o, act_s, act_k))
    to_optimize.update(unfreeze_bin_attributes(handles, act_o, act_e))
    return to_optimize


def create_partial_starting_point(mdl, solution, to_optimize):
    """
    Creates a warm start (starting point) containing ONLY the variables
    that are currently unfrozen (in to_optimize). Frozen variables are
    left blank to be deduced by the engine's presolver.
    """
    sp = mdl.create_empty_solution()

    for var in to_optimize:
        var_sol = solution.get_var_solution(var)

        # Skip if the variable somehow doesn't exist in the current solution
        if var_sol is None:
            continue

        if "Interval" in var.type.name:
            if var_sol.is_present():
                sp.add_interval_var_solution(
                    var,
                    presence=True,
                    start=var_sol.get_start(),
                    end=var_sol.get_end(),
                    size=var_sol.get_size(),
                )
            else:
                sp.add_interval_var_solution(var, presence=False)

        elif any(t in var.type.name for t in ("Int", "Bool", "Float")):
            # Docplex uses add_integer_var_solution for Ints and Bools
            sp.add_integer_var_solution(var, var_sol.get_value())

    return sp


def apply_partial_starting_point(mdl, solver, to_optimize):
    """
    Creates a warm start containing ONLY the variables in the active neighborhood.
    Frozen variables do not need hints as their domains are restricted.
    """
    mdl.ClearHints()
    
    for var in to_optimize:
        pres_val = solver.Value(var.pres)
        mdl.AddHint(var.pres, pres_val)
        
        if pres_val:
            mdl.AddHint(var.start, solver.Value(var.start))
            mdl.AddHint(var.end, solver.Value(var.end))
