import random
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from cp_model_utils import (
    get_fleet_utilization_timeseries,
    get_pickface_utilization_per_station,
)
from freezing_utils import get_assigned_station
from matheuristic_plots import Status

random.seed(42)


@dataclass
class SelectionResult:
    seed_orders: set = field(default_factory=set)
    seed_skus: set = field(default_factory=set)
    seed_stations: set = field(default_factory=set)


# Strategy ideas
#
# Dynamic strategies (change with the solution)
# Longest scheduled orders/bin events
# SKUs with the most fetches
# Time slice where movecap is least saturated + slice where it is most
# Orders from most used stations + orders from least used station
# Orders from most used lanes + orders from least used lanes of the same station
# Dynamic similarity scores (mix jaccard and being in the same station etc.)
# Least/most utilized timeslice from each station
#
# Static strategies (defined by the instance)
# Similar orders (already done Shaw) or SKUs


def strategy_random_orders(handles, k=None, p=None) -> SelectionResult:
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


def strategy_random_skus(handles, k=None, p=None) -> SelectionResult:
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


def get_order_intervals(handles, solution):
    order_intervals = {}
    makespan = 0
    if solution is None:
        return order_intervals, makespan
    for o in handles["O"]:
        s_assigned = get_assigned_station(handles, solution, o)
        if s_assigned is not None:
            var = handles["I_os"].get((o, s_assigned))
            if var is not None:
                if (
                    hasattr(solution, "Value")
                    and hasattr(var, "start")
                    and hasattr(var, "end")
                ):
                    start, end = solution.Value(var.start), solution.Value(var.end)
                    order_intervals[o] = (start, end, s_assigned)
                    makespan = max(makespan, end)
                else:
                    var_sol = solution.get_var_solution(var)
                    if var_sol and var_sol.is_present():
                        start, end = var_sol.get_start(), var_sol.get_end()
                        order_intervals[o] = (start, end, s_assigned)
                        makespan = max(makespan, end)
    return order_intervals, makespan


def orders_in_timeslice(
    order_intervals, t_start, t_end, stations_mask=None
) -> SelectionResult:
    seed_o = {
        o
        for o, (s, e, station) in order_intervals.items()
        if s < t_end
        and e > t_start
        and (stations_mask is None or station in stations_mask)
    }
    return SelectionResult(seed_orders=seed_o)


def strategy_multi_random_timeslice(
    handles, solution, n, total_length
) -> SelectionResult:
    order_intervals, makespan = get_order_intervals(handles, solution)

    length = total_length // n
    timeslices = []
    for i in range(n):
        if makespan <= length:
            t_start, t_end = 0, makespan
        else:
            t_start = random.randint(0, max(0, makespan - length))
            t_end = t_start + length
        timeslices.append((t_start, t_end))

    return combine_strategies(
        *[
            orders_in_timeslice(order_intervals, t_start, t_end)
            for (t_start, t_end) in timeslices
        ]
    )


def strategy_movecap_balancing_slices(handles, solution, total_length):
    slice_length = total_length // 2
    order_intervals, makespan = get_order_intervals(handles, solution)

    if makespan <= slice_length:
        t_start, t_end = 0, makespan
        return orders_in_timeslice(order_intervals, t_start, t_end)

    fleet_util = get_fleet_utilization_timeseries(solution, handles, makespan)
    rolling_sums = np.convolve(
        fleet_util, np.ones(slice_length, dtype=int), mode="valid"
    )

    noisy_sums = rolling_sums + np.random.normal(
        0, np.std(rolling_sums) / 10, len(rolling_sums)
    )

    t_start_min = np.argmin(noisy_sums)
    t_start_max = np.argmax(noisy_sums)
    return combine_strategies(
        orders_in_timeslice(order_intervals, t_start_min, t_start_min + slice_length),
        orders_in_timeslice(order_intervals, t_start_max, t_start_max + slice_length),
    )


def strategy_pickface_balancing_station_slices(handles, solution, total_length):
    slice_length = total_length // 2
    order_intervals, makespan = get_order_intervals(handles, solution)
    per_station_util = get_pickface_utilization_per_station(solution, handles, makespan)

    order_strats = []
    for s in handles["S"]:
        rolling_sums = np.convolve(
            per_station_util[s], np.ones(slice_length, dtype=int), mode="valid"
        )

        noisy_sums = rolling_sums + np.random.normal(
            0, np.std(rolling_sums) / 10, len(rolling_sums)
        )

        t_start_min = np.argmin(noisy_sums)
        t_start_max = np.argmax(noisy_sums)
        (
            order_strats.append(
                orders_in_timeslice(
                    order_intervals,
                    t_start_min,
                    t_start_min + slice_length,
                    stations_mask=[s],
                )
            ),
        )
        (
            order_strats.append(
                orders_in_timeslice(
                    order_intervals,
                    t_start_max,
                    t_start_max + slice_length,
                    stations_mask=[s],
                )
            ),
        )
    return combine_strategies(*order_strats)


def compute_lane_congestion(handles, solution, use_duration):

    all_lanes = [(s, ln) for s in handles["S"] for ln in handles["L"]]
    congestion = defaultdict(int)
    for o in handles["O"]:
        for s, ln in all_lanes:
            var = handles["I_os_lane"].get((o, s, ln))
            if var is not None:
                var_sol = solution.get_var_solution(var)
                if var_sol and var_sol.is_present():
                    if use_duration:
                        congestion[(s, ln)] += var_sol.get_end() - var_sol.get_start()
                    else:
                        congestion[(s, ln)] += 1
                    break
    return congestion


def orders_at_lanes(handles, solution, chosen_lanes):
    seed_o = set()
    for o in handles["O"]:
        for s, ln in chosen_lanes:
            var = handles["I_os_lane"].get((o, s, ln))
            if var is not None:
                var_sol = solution.get_var_solution(var)
                if var_sol and var_sol.is_present():
                    seed_o.add(o)
                    break
    return seed_o


def strategy_random_lanes(handles, solution, k) -> SelectionResult:
    all_lanes = [(s, ln) for s in handles["S"] for ln in handles["L"]]
    chosen_lanes = set(random.sample(all_lanes, min(k, len(all_lanes) - 1)))
    seed_o = orders_at_lanes(handles, solution, chosen_lanes)
    return SelectionResult(seed_orders=seed_o)


def strategy_balancing_lanes(handles, solution, k, use_duration=True):
    """
    Compute congestion of lanes by number of orders and select (1) most congested and (k-1) least congested.
    """
    k = max(k, 2)
    congestion = compute_lane_congestion(handles, solution, use_duration)
    sorted_lanes = sorted(
        congestion.keys(), key=lambda k: (congestion[k], random.random())
    )
    top = sorted_lanes[-1]
    least = sorted_lanes[: k - 1]
    seed_o = orders_at_lanes(handles, solution, least + [top])
    return SelectionResult(seed_orders=seed_o)


def strategy_similar_orders(handles, p_most_similar, jaccard) -> SelectionResult:

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
    selected_orders.update(sorted_orders[-n_select:])
    return SelectionResult(seed_orders=selected_orders)


def is_interval_present(solution, var) -> bool:
    """Checks if an interval variable is active in the given solution."""
    if var is None or solution is None:
        return False
    if hasattr(solution, "BooleanValue"):
        pres_var = getattr(var, "pres", var)
        try:
            return bool(solution.BooleanValue(pres_var))
        except Exception:
            pass
    if hasattr(solution, "get_var_solution"):
        var_sol = solution.get_var_solution(var)
        return var_sol is not None and var_sol.is_present()
    if hasattr(var, "is_present"):
        return bool(var.is_present())
    return False


def strategy_frequently_refetched_skus(
    handles, solution, k=None, p=None, weighted: bool = False
) -> SelectionResult:
    """
    Selects top p% (or k) SKUs with the highest number of refetches.
    If weighted is True, counts total cycle time cost instead: (r + p + r_return) * num_refetch.
    """
    if solution is None:
        return SelectionResult()

    active_skus = handles.get("active_K", [])
    if not active_skus:
        return SelectionResult()

    active_k_set = set(active_skus)
    fetch_counts = {sku: 0 for sku in active_skus}

    F_handles = handles.get("F", {})
    for (s, sku, e), f_var in F_handles.items():
        if sku in active_k_set and is_interval_present(solution, f_var):
            fetch_counts[sku] += 1

    if weighted:
        rt = handles.get("rt", handles.get("r", {}))
        p_times = handles.get("p", {})
        rt_return = handles.get("rt_return", handles.get("r_return", rt))

        scores = {}
        for sku in active_skus:
            r_k = (
                rt.get(sku, 0)
                if isinstance(rt, dict)
                else (rt if isinstance(rt, (int, float)) else 0)
            )
            p_k = (
                p_times.get(sku, 0)
                if isinstance(p_times, dict)
                else (p_times if isinstance(p_times, (int, float)) else 0)
            )
            r_ret_k = (
                rt_return.get(sku, r_k)
                if isinstance(rt_return, dict)
                else (rt_return if isinstance(rt_return, (int, float)) else r_k)
            )
            cycle_time = r_k + p_k + r_ret_k
            scores[sku] = cycle_time * fetch_counts[sku]
    else:
        scores = fetch_counts

    num_skus = len(active_skus)
    if k is not None:
        n_select = min(k, num_skus)
    elif p is not None:
        p = min(1.0, p)
        n_select = max(1, int(num_skus * p))
    else:
        n_select = max(1, int(num_skus * 0.1))

    n_select = min(n_select, num_skus)

    # Sort descending by score, tie-break randomly
    sorted_skus = sorted(
        active_skus,
        key=lambda sku: (scores[sku], random.random()),
        reverse=True,
    )

    candidate_skus = [sku for sku in sorted_skus if scores[sku] > 0]
    seed_k = set(candidate_skus[:n_select])
    return SelectionResult(seed_skus=seed_k)


def strategy_cycle_time_weighted_refetched_skus(
    handles, solution, k=None, p=None
) -> SelectionResult:
    """
    Selects top p% (or k) SKUs with the highest cycle time-weighted refetch cost:
    (r + p + r_return) * num_refetch.
    """
    return strategy_frequently_refetched_skus(
        handles, solution, k=k, p=p, weighted=True
    )


# Convenience alias
strategy_refetched_skus = strategy_frequently_refetched_skus


def strategy_longest_orders(handles, solution, k=None, p=None) -> SelectionResult:
    """
    Finds the top p% (or k) orders which are taking the longest to fulfill
    by sorting their I_os interval lengths (flow time = end - start).
    """
    if solution is None:
        return SelectionResult()

    order_intervals, _ = get_order_intervals(handles, solution)
    if not order_intervals:
        return SelectionResult()

    num_orders = len(handles.get("O", []))
    if k is not None:
        n_select = min(k, len(order_intervals))
    elif p is not None:
        p = min(1.0, p)
        n_select = max(1, int(num_orders * p))
    else:
        n_select = max(1, int(num_orders * 0.1))

    n_select = min(n_select, len(order_intervals))

    # Sort descending by flow time (end - start), breaking ties randomly
    sorted_orders = sorted(
        order_intervals.keys(),
        key=lambda o: (order_intervals[o][1] - order_intervals[o][0], random.random()),
        reverse=True,
    )
    seed_o = set(sorted_orders[:n_select])
    return SelectionResult(seed_orders=seed_o)


def combine_strategies(*strategy_outputs: SelectionResult) -> SelectionResult:
    """Combines any number of selection results into a single selection."""
    combined_res = SelectionResult()
    for sel_results in strategy_outputs:
        combined_res.seed_orders.update(sel_results.seed_orders)
        combined_res.seed_skus.update(sel_results.seed_skus)
        combined_res.seed_stations.update(sel_results.seed_stations)
    return combined_res


class StrategyManager:
    def __init__(
        self,
        handles,
        solution,
        weighted_jaccard_matrix,
        strategy_preset: str = "all",
        strat_choice: str = "uniform",
        default_percent: float = 0.1,
        default_n_lanes: int = 2,
        default_timeslice_width: int = 100,
    ):
        self.strategy_preset = strategy_preset
        self.strat_choice = strat_choice

        self.handles = handles
        self.current_solution = solution
        self.weighted_jaccard_matrix = weighted_jaccard_matrix

        self.strategies = {
            "random_orders": lambda sev: strategy_random_orders(
                self.handles, p=default_percent * sev
            ),
            "similar_orders": lambda sev: strategy_similar_orders(
                self.handles,
                default_percent * sev,
                self.weighted_jaccard_matrix,
            ),
            "random_skus": lambda sev: strategy_random_skus(
                self.handles, p=default_percent * sev
            ),
            "frequently_refetched_skus": lambda sev: strategy_frequently_refetched_skus(
                self.handles,
                self.current_solution,
                p=default_percent * sev,
                weighted=False,
            ),
            "refetched_skus": lambda sev: strategy_frequently_refetched_skus(
                self.handles,
                self.current_solution,
                p=default_percent * sev,
                weighted=False,
            ),
            "cycle_time_weighted_refetched_skus": lambda sev: (
                strategy_frequently_refetched_skus(
                    self.handles,
                    self.current_solution,
                    p=default_percent * sev,
                    weighted=True,
                )
            ),
            "longest_orders": lambda sev: strategy_longest_orders(
                self.handles, self.current_solution, p=default_percent * sev
            ),
            # "random_orders_and_skus": lambda sev: combine_strategies(
            #     strategy_random_orders(self.handles, p=0.5 * default_percent * sev),
            #     strategy_random_skus(self.handles, p=0.5 * default_percent * sev),
            # ),
            "random_lanes": lambda sev: strategy_random_lanes(
                self.handles, self.current_solution, default_n_lanes + sev
            ),
            "balancing_lanes": lambda sev: strategy_balancing_lanes(
                self.handles, self.current_solution, default_n_lanes + sev
            ),
            "single_timeslice": lambda sev: strategy_multi_random_timeslice(
                self.handles, self.current_solution, 1, default_timeslice_width * sev
            ),
            "double_timeslice": lambda sev: strategy_multi_random_timeslice(
                self.handles, self.current_solution, 2, default_timeslice_width * sev
            ),
            # "triple_timeslice": lambda sev: strategy_multi_random_timeslice(
            #     self.handles, self.current_solution, 3, default_timeslice_width * sev
            # ),
            "movecap_balancing_timeslice": lambda sev: (
                strategy_movecap_balancing_slices(
                    self.handles, self.current_solution, default_timeslice_width * sev
                )
            ),
            "pickface_balancing_station_timeslice": lambda sev: (
                strategy_pickface_balancing_station_slices(
                    self.handles, self.current_solution, default_timeslice_width * sev
                )
            ),
        }

        self.presets = {
            "all": None,
            "shaw_random": ["similar_orders", "random_orders"],
            "random_balance": [
                "random_orders_and_skus",
                "movecap_balancing_timeslice",
                "pickface_balancing_station_timeslice",
                "balancing_lanes",
                "similar_orders",
            ],
            "random_orders": ["random_orders"],
        }

        active_preset = self.presets[self.strategy_preset]
        if active_preset == None:
            self.active_strats = dict(self.strategies)
        else:
            self.active_strats = {
                name: strat
                for name, strat in self.strategies.items()
                if name in active_preset
            }
        self.active_strat_names = list(self.active_strats.keys())

        self.strat_statistics = {
            name: {"num_uses": 0, "status": {s.value: 0 for s in Status}}
            for name, strat in self.active_strats.items()
        }

        self.A, self.B, self.C, self.D = (
            100,
            50,
            1,
            0,
        )  # const term, new best, feasible (something was found), unknown/optimal no improve)

        self.scores = {name: self.A for name, strat in self.active_strats.items()}

    def update_score(self, strat: str):
        A, B, C, D = (
            5,
            10,
            1,
            0,
        )  # const term, new best, some searching happened but no improvement, gridlocked or too large

        score = A

        score += (
            len(self.strat_statistics[strat]["status"][Status.Optimal_New_Best]) * B
        )
        score += (
            len(self.strat_statistics[strat]["status"][Status.Feasible_New_Best]) * B
        )

        score += len(self.strat_statistics[strat]["status"][Status.Optimal_Improve]) * C
        score += (
            len(self.strat_statistics[strat]["status"][Status.Feasible_Improve]) * C
        )
        score += (
            len(self.strat_statistics[strat]["status"][Status.Feasible_No_Improve]) * C
        )
        score += len(self.strat_statistics[strat]["status"][Status.Unknown]) * C

        score += (
            len(self.strat_statistics[strat]["status"][Status.Optimal_No_Improve]) * D
        )
        score += (
            len(self.strat_statistics[strat]["status"][Status.Feasible_Degradation]) * D
        )
        score += len(self.strat_statistics[strat]["status"][Status.Infeasible]) * D

        self.scores[strat] = score / (1 + self.strat_statistics[strat]["num_uses"])

    def choose_strat(self) -> Callable:
        if self.strat_choice == "uniform":
            strat_idx = random.randint(0, len(self.active_strat_names) - 1)
        elif self.strat_choice == "adaptive":
            strat_idx = random.choices(
                range(len(self.active_strat_names)), weights=self.scores.values(), k=1
            )[0]

        strat_name = self.active_strat_names[strat_idx]
        strat = self.active_strats[strat_name]

        return strat_idx, strat_name, strat
