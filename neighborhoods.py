import random


def select_station_neighborhood(handles, solution, num_stations=2):
    unfrozen_vars = set()

    S = handles["S"]
    O = handles["O"]
    L = handles["L"]
    orders_req = handles["orders_req"]
    U = handles["U"]

    # 1. Pick a random subset of stations to completely clear out
    free_stations = set(random.sample(S, min(num_stations, len(S))))

    # 2. Find ALL orders that are currently assigned to these stations
    free_orders = set()
    for o in O:
        for s in free_stations:
            var = handles["I_os"][(o, s)]
            var_sol = solution.get_var_solution(var)

            # If the order is present at the free station, add it to our list
            if var_sol is not None and var_sol.is_present():
                free_orders.add(o)
                break

    # 3. Unfreeze EVERYTHING for the selected orders (across all stations)
    # This allows them to change stations, change lanes, and change times
    for o in free_orders:
        for s in S:
            unfrozen_vars.add(handles["I_os"][(o, s)])
            for ln in L:
                unfrozen_vars.add(handles["I_os_lane"][(o, s, ln)])
            for k in orders_req[o]:
                unfrozen_vars.add(handles["C"][(o, k, s)])
                for e in range(U[k]):
                    if (o, s, k, e) in handles["P"]:
                        unfrozen_vars.add(handles["P"][(o, s, k, e)])

    # 4. Unfreeze ALL Bin variables at the selected stations
    # Because all orders at these stations are free, the bins are completely
    # detached from the frozen schedule and can be freely rearranged.
    for s in free_stations:
        for k in handles["K"]:
            if U.get(k, 0) > 0:
                for e in range(U[k]):
                    unfrozen_vars.add(handles["F"][(s, k, e)])
                    unfrozen_vars.add(handles["R"][(s, k, e)])
                    unfrozen_vars.add(handles["B"][(s, k, e)])
                    unfrozen_vars.add(handles["Block"][(s, k, e)])

    return unfrozen_vars


def select_order_neighborhood(handles, percent_to_free=0.15):
    unfrozen_vars = set()

    O = handles["O"]
    S = handles["S"]
    L = handles["L"]
    orders_req = handles["orders_req"]

    # 1. Select a random subset of orders
    num_orders = max(1, int(len(O) * percent_to_free))
    free_orders = set(random.sample(O, num_orders))

    # 2. Find which SKUs those orders need
    free_skus = set()
    for o in free_orders:
        free_skus.update(orders_req[o])

    # 3. Unfreeze all Order-level variables for the selected orders
    for o in free_orders:
        for s in S:
            unfrozen_vars.add(handles["I_os"][(o, s)])
            for ln in L:
                unfrozen_vars.add(handles["I_os_lane"][(o, s, ln)])
            for k in orders_req[o]:
                unfrozen_vars.add(handles["C"][(o, k, s)])

                # Unfreeze the Picks associated with this order
                for e in range(handles["U"][k]):
                    unfrozen_vars.add(handles["P"][(o, s, k, e)])

    # 4. Unfreeze ALL Bin cycle variables (F, R, B, Block) for the affected SKUs
    # This allows the solver to re-route bins across different stations
    # to serve the newly freed orders optimally.
    for k in free_skus:
        for s in S:
            for e in range(handles["U"][k]):
                unfrozen_vars.add(handles["F"][(s, k, e)])
                unfrozen_vars.add(handles["R"][(s, k, e)])
                unfrozen_vars.add(handles["B"][(s, k, e)])
                unfrozen_vars.add(handles["Block"][(s, k, e)])

    return unfrozen_vars


def select_timeslice_neighborhood(
    handles, solution, current_makespan, window_size=1000, random_pct=0.1
):
    unfrozen_vars = set()

    # Pick a time window
    start_time = random.randint(0, max(0, current_makespan - window_size))
    end_time = start_time + window_size

    # Iterate through the solution and unfreeze any variable that touches this window
    for var_sol in solution.get_all_var_solutions():
        var = var_sol.get_var()

        if "Interval" in var.type.name:
            if var_sol.is_present():
                if var_sol.get_start() <= end_time and var_sol.get_end() >= start_time:
                    unfrozen_vars.add(var)
            else:
                # Unfreeze a portion of absent variables so the solver
                # can try to insert them into the slack
                if random.random() < random_pct:
                    unfrozen_vars.add(var)

    return unfrozen_vars


def select_random_neighborhood(handles, solution, percent_to_free=0.01):
    """
    Randomly unfreezes a percentage of all variables in the solution.
    Expected to gridlock frequently due to broken temporal and logical links.
    """
    unfrozen_vars = set()

    # 1. Extract all variable objects from the current solution
    all_vars = [var_sol.get_var() for var_sol in solution.get_all_var_solutions()]

    # 2. Determine the exact number of variables to unfreeze
    num_vars_to_free = max(1, int(len(all_vars) * percent_to_free))

    # 3. Randomly sample the variable objects
    selected_vars = random.sample(all_vars, min(num_vars_to_free, len(all_vars)))

    # 4. Add them to the unfrozen set
    unfrozen_vars.update(selected_vars)

    return unfrozen_vars
