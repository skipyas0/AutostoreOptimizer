import random
from dataclasses import dataclass, field

from freezing_utils import get_assigned_station

random.seed(42)


@dataclass
class SelectionResult:
    seed_orders: set = field(default_factory=set)
    seed_skus: set = field(default_factory=set)
    seed_stations: set = field(default_factory=set)


def strategy_random_orders(handles, solution, k=None, p=None) -> SelectionResult:
    orders = handles["O"]
    num_orders = len(orders)

    if k is not None:
        n_select = min(k, num_orders)
    elif p is not None:
        p = min(1.0, p)
        n_select = max(1, int(num_orders * p))
    else:
        n_select = max(1, int(num_orders * 0.1))

    seed_o = set(random.sample(orders, n_select))
    return SelectionResult(seed_orders=seed_o)


def strategy_random_skus(handles, solution, k=None, p=None) -> SelectionResult:
    skus = handles["active_K"]
    num_skus = len(skus)
    if k is not None:
        n_select = min(k, num_skus)
    elif p is not None:
        p = min(1.0, p)
        n_select = max(1, int(num_skus * p))
    else:
        n_select = max(1, int(num_skus * 0.1))

    seed_k = set(random.sample(skus, n_select))
    return SelectionResult(seed_skus=seed_k)


def strategy_single_timeslice(handles, solution, length) -> SelectionResult:
    makespan = 0
    order_intervals = {}

    for o in handles["O"]:
        s_assigned = get_assigned_station(handles, solution, o)
        if s_assigned is not None:
            var = handles["I_os"].get((o, s_assigned))
            if var is not None:
                if hasattr(solution, "BooleanValue"):
                    if solution.BooleanValue(var.pres):
                        start, end = solution.Value(var.start), solution.Value(var.end)
                        order_intervals[o] = (start, end)
                        makespan = max(makespan, end)
                else:
                    var_sol = solution.get_var_solution(var)
                    if var_sol and var_sol.is_present():
                        start, end = var_sol.get_start(), var_sol.get_end()
                        order_intervals[o] = (start, end)
                        makespan = max(makespan, end)

    if makespan <= length:
        t_start, t_end = 0, makespan
    else:
        t_start = random.randint(0, max(0, makespan - length))
        t_end = t_start + length

    seed_o = {o for o, (s, e) in order_intervals.items() if s < t_end and e > t_start}
    return SelectionResult(seed_orders=seed_o)


def strategy_multi_timeslice(handles, solution, n, total_length) -> SelectionResult:
    return combine_strategies(
        *[
            strategy_single_timeslice(handles, solution, total_length // n)
            for _ in range(n)
        ]
    )


def strategy_random_lanes(handles, solution, k) -> SelectionResult:
    all_lanes = [(s, ln) for s in handles["S"] for ln in handles["L"]]
    chosen_lanes = set(random.sample(all_lanes, min(k, len(all_lanes))))

    seed_o = set()
    for o in handles["O"]:
        for s, ln in chosen_lanes:
            var = handles["I_os_lane"].get((o, s, ln))
            if var is not None:
                if hasattr(solution, "BooleanValue"):
                    if solution.BooleanValue(var.pres):
                        seed_o.add(o)
                        break
                else:
                    var_sol = solution.get_var_solution(var)
                    if var_sol and var_sol.is_present():
                        seed_o.add(o)
                        break

    return SelectionResult(seed_orders=seed_o)


def strategy_similar_orders(
    handles, solution, p_most_similar, jaccard
) -> SelectionResult:

    seed_order = random.choice(handles["O"])
    selected_orders = {seed_order}
    sorted_orders = sorted(
        handles["O"],
        key=lambda o: (
            float("inf")
            if seed_order == o
            else jaccard[min(seed_order, o), max(seed_order, o)]
        ),
    )

    p = min(1.0, p_most_similar)
    n_select = max(1, int(len(handles["O"]) * p))
    selected_orders.update(
        {handles["O"][order_ix] for order_ix in sorted_orders[-n_select:]}
    )
    return SelectionResult(seed_orders=selected_orders)


def combine_strategies(*strategy_outputs: SelectionResult) -> SelectionResult:
    """Combines any number of selection results into a single selection."""
    combined_res = SelectionResult()
    for sel_results in strategy_outputs:
        combined_res.seed_orders.update(sel_results.seed_orders)
        combined_res.seed_skus.update(sel_results.seed_skus)
        combined_res.seed_stations.update(sel_results.seed_stations)
    return combined_res
