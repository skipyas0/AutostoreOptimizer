#!/usr/bin/env python3
"""
AutoStore Adaptive Multi-Attribute SGC (AMA-SGC) Heuristic.

Runs the SGC construction multiple times with different combinations of
order-priority rule and bin-fetch ordering rule, then returns the best solution.
Mirrors variable names from the CP model (cp_model.py).
"""
from autostore_heuristic import (
    BinEvent, OrderPlan, HeuristicState, Solution,
    init_state, commit_plan, validate_solution, compute_U,
    find_shared_bin, earliest_feasible_fetch,
    build_viz_handles,
)
import sys
import os
from collections import defaultdict
from typing import Optional
from instance import Instance

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# Step 1: Precompute order-level and SKU-level statistics
# ============================================================

def precompute_attributes(
        instance,
        *args,
        **kwargs
) -> tuple[dict[int, dict], dict[int, dict]]:
    """Compute all order-level and SKU-level statistics used by sorting rules.

    Returns (order_attrs, sku_attrs) where:
      order_attrs[o] is a dict of attribute_key -> value for order o
      sku_attrs[k] is a dict of attribute_key -> value for SKU k
    """
    if not isinstance(instance, Instance):
        O = instance
        orders_req = args[0]
        rt = args[1]
        rt_ret = args[2]
        p = args[3]
        N = args[4]
        K = args[5]
        instance = Instance([], [], K, orders_req, rt, p, N, rt_ret=rt_ret)
    else:
        rt_ret = kwargs.get('rt_ret', args[0] if len(args) > 0 else None)

    S, L, K, orders_req, rt, p, N = instance
    O = instance.O
    if rt_ret is None:
        rt_ret = instance.rt_ret
    # demand_count[k] = number of orders that need SKU k
    demand_count: dict[int, int] = defaultdict(int)
    for o in O:
        for k in orders_req[o]:
            demand_count[k] += 1

    # --- SKU-level attributes ---
    sku_attrs: dict[int, dict] = {}
    for k in K:
        sku_attrs[k] = {
            "rt": rt[k],
            "p": p[k],
            "cycle": rt[k] + p[k] + rt_ret[k],
            "demand_ratio": demand_count[k] / max(N[k], 1),
            "copies": N[k],
            "demand": demand_count[k],
        }

    # --- Order-level attributes ---
    order_attrs: dict[int, dict] = {}
    for o in O:
        req = orders_req[o]
        sum_rt = sum(rt[k] for k in req)
        order_size = len(req)
        sum_cycle = sum(rt[k] + p[k] + rt_ret[k] for k in req)
        max_rt = max((rt[k] for k in req), default=0)
        # sku_rarity: orders with rare (low-demand) SKUs have high sum of 1/demand
        sku_rarity = sum(1.0 / max(demand_count[k], 1) for k in req)
        # sku_contention: sum of demand[k]/N[k] — high means copy pressure
        sku_contention = sum(demand_count[k] / max(N[k], 1) for k in req)
        # sharing_degree: number of other orders that share at least one SKU
        sharing_degree = len({
            o2 for o2 in O if o2 != o
            and any(k in orders_req.get(o2, []) for k in req)
        })
        min_copies = min((N[k] for k in req), default=1)

        order_attrs[o] = {
            "sum_rt": sum_rt,
            "order_size": order_size,
            "sum_cycle": sum_cycle,
            "max_rt": max_rt,
            "sku_rarity": sku_rarity,
            "sku_contention": sku_contention,
            "sharing_degree": sharing_degree,
            "min_copies": min_copies,
        }

    return order_attrs, sku_attrs


# ============================================================
# Step 2: Parameterised sorting functions
# ============================================================

def sort_orders(
        O: list[int],
        order_attrs: dict[int, dict],
        attr_key: str,
        descending: bool,
) -> list[int]:
    """Sort order ids by the given attribute key and direction."""
    return sorted(O, key=lambda o: order_attrs[o][attr_key], reverse=descending)


def sort_skus_for_order(
        order_skus: list[int],
        sku_attrs: dict[int, dict],
        attr_key: str,
        descending: bool,
) -> list[int]:
    """Sort SKU ids within an order by the given attribute key and direction."""
    return sorted(order_skus, key=lambda k: sku_attrs[k][attr_key], reverse=descending)


# ============================================================
# Step 3: plan_order_at_station accepting bin-sort parameters
# ============================================================

def plan_order_at_station_parameterised(
        o: int, s: int,
        state: HeuristicState,
        orders_req: dict[int, list[int]],
        rt: dict[int, int], rt_ret: dict[int, int], p: dict[int, int],
        N: dict[int, int],
        horizon: int,
        ALPHA: float, BETA: float,
        sku_attrs: dict[int, dict],
        bin_attr_key: str,
        bin_descending: bool,
) -> Optional[OrderPlan]:
    """Tentatively schedule order o at station s with parameterised bin-fetch ordering.

    Identical to plan_order_at_station in autostore_heuristic.py except the
    within-order SKU sequence is driven by (bin_attr_key, bin_descending) instead
    of the hardcoded rt-descending sort.
    """
    L_at_s = [ln for (ss, ln) in state.lane_free if ss == s]
    if not L_at_s:
        return None

    best_lane = min(L_at_s, key=lambda ln: state.lane_free[(s, ln)])
    t_lane = state.lane_free[(s, best_lane)]
    t_order_start = t_lane

    # Parameterised bin-fetch ordering
    skus_sorted = sort_skus_for_order(orders_req[o], sku_attrs, bin_attr_key, bin_descending)

    t_cursor = t_order_start
    current_station_free = state.pickface_free[s]

    new_bin_events: list[BinEvent] = []
    shared_picks: list[tuple[int, int, int]] = []
    pending_moves: list[tuple[int, int]] = []
    pending_copies: dict[int, list[tuple[int, int]]] = defaultdict(list)
    pick_times_dict: dict[int, tuple[int, int]] = {}

    for k in skus_sorted:
        # --- Try shared bin first ---
        shared = find_shared_bin(state.station_bin_events[s], k, t_cursor, p)

        if shared is None:
            for ev in new_bin_events:
                if ev.sku == k:
                    start = max(t_cursor, ev.presence_start)
                    if start + p[k] <= ev.presence_end:
                        shared = ev
                        break

        if shared is not None:
            pick_start = max(t_cursor, shared.presence_start)
            pick_end = pick_start + p[k]
            t_cursor = pick_end
            pick_times_dict[k] = (pick_start, pick_end)
            shared_picks.append((k, shared, pick_end))
            continue

        # --- New bin fetch ---
        try:
            earliest_presence = max(t_cursor, current_station_free)
            desired_fetch = earliest_presence - rt[k]
            t_fetch, copy_id = earliest_feasible_fetch(
                k, s, desired_fetch, state, rt, rt_ret,
                pending_moves, pending_copies, horizon,
            )
        except ValueError:
            return None

        fetch_end = t_fetch + rt[k]
        presence_start = fetch_end
        pick_start = presence_start
        pick_end = pick_start + p[k]
        pick_times_dict[k] = (pick_start, pick_end)
        return_start = pick_end
        return_end = return_start + rt_ret[k]
        presence_end = return_start

        if return_end > horizon:
            return None

        if state.move_cap is not None:
            return_delayed = state.move_da.delay_for_movecap_with_pending(
                return_start, rt_ret[k], state.move_cap, pending_moves
            )
            if return_delayed != return_start:
                return_start = return_delayed
                return_end = return_start + rt_ret[k]
                presence_end = return_start
                if return_end > horizon:
                    return None

        t_cursor = pick_end
        current_station_free = max(current_station_free, presence_end)

        ev = BinEvent(
            sku=k, copy_id=copy_id,
            fetch_start=t_fetch, fetch_end=fetch_end,
            presence_start=presence_start, presence_end=presence_end,
            return_start=return_start, return_end=return_end,
            orders_served=[o],
        )
        new_bin_events.append(ev)
        pending_moves.append((t_fetch, fetch_end))
        pending_moves.append((return_start, return_end))
        pending_copies[k].append((copy_id, return_end))

    order_start = min((ps for ps, pe in pick_times_dict.values()), default=t_order_start)
    order_end = max((pe for ps, pe in pick_times_dict.values()), default=t_order_start)
    score = ALPHA * order_end + BETA * (order_end - order_start)

    return OrderPlan(
        order=o, station=s, lane=best_lane,
        start=order_start, end=order_end,
        bin_events=new_bin_events,
        shared_picks=shared_picks,
        score=score,
        pick_times=pick_times_dict,
    )


# ============================================================
# Step 4: run_sgc_parameterised
# ============================================================

def run_sgc_parameterised(
        instance,
        *args,
        **kwargs
) -> Solution:
    """Run SGC with parameterised order-sort and bin-fetch-sort rules."""
    if not isinstance(instance, Instance):
        def get_arg(name, idx, default):
            if name in kwargs:
                return kwargs[name]
            real_idx = idx + 8
            if len(args) > real_idx:
                return args[real_idx]
            return default

        S = instance
        L = args[0]
        K = args[1]
        O = args[2]
        orders_req = args[3]
        rt = args[4]
        rt_ret = args[5]
        p = args[6]
        N = args[7]
        horizon = get_arg('horizon', 0, 10000)
        move_cap = get_arg('move_cap', 1, None)
        if move_cap is None:
            move_cap = kwargs.get('movecap', None)
        ALPHA = get_arg('ALPHA', 2, 1.0)
        BETA = get_arg('BETA', 3, 0.0)
        order_attrs = get_arg('order_attrs', 4, None)
        sku_attrs = get_arg('sku_attrs', 5, None)
        order_attr_key = get_arg('order_attr_key', 6, None)
        order_descending = get_arg('order_descending', 7, None)
        bin_attr_key = get_arg('bin_attr_key', 8, None)
        bin_descending = get_arg('bin_descending', 9, None)
        instance = Instance(S, L, K, orders_req, rt, p, N, rt_ret=rt_ret)
    else:
        def get_arg_new(name, idx, default):
            if name in kwargs:
                return kwargs[name]
            if len(args) > idx:
                return args[idx]
            return default

        horizon = get_arg_new('horizon', 0, 10000)
        move_cap = get_arg_new('move_cap', 1, None)
        if move_cap is None:
            move_cap = kwargs.get('movecap', None)
        ALPHA = get_arg_new('ALPHA', 2, 1.0)
        BETA = get_arg_new('BETA', 3, 0.0)
        order_attrs = get_arg_new('order_attrs', 4, None)
        sku_attrs = get_arg_new('sku_attrs', 5, None)
        order_attr_key = get_arg_new('order_attr_key', 6, None)
        order_descending = get_arg_new('order_descending', 7, None)
        bin_attr_key = get_arg_new('bin_attr_key', 8, None)
        bin_descending = get_arg_new('bin_descending', 9, None)
        rt_ret = kwargs.get('rt_ret', args[10] if len(args) > 10 else None)

    S, L, K, orders_req, rt, p, N = instance
    O = instance.O
    if rt_ret is None:
        rt_ret = instance.rt_ret

    state = init_state(S, L, K, N, horizon, move_cap)
    sorted_orders = sort_orders(O, order_attrs, order_attr_key, order_descending)

    order_assignments: dict[int, tuple[int, int, int, int]] = {}
    pick_events_map: dict[tuple[int, int, int], tuple[int, int]] = {}
    failed_orders: list[int] = []

    for o in sorted_orders:
        best_plan: Optional[OrderPlan] = None

        for s in S:
            plan = plan_order_at_station_parameterised(
                o, s, state, orders_req, rt, rt_ret, p, N,
                horizon, ALPHA, BETA,
                sku_attrs, bin_attr_key, bin_descending,
            )
            if plan is not None and (best_plan is None or plan.score < best_plan.score):
                best_plan = plan

        if best_plan is not None:
            commit_plan(best_plan, state)
            order_assignments[o] = (
                best_plan.station, best_plan.lane,
                best_plan.start, best_plan.end,
            )
            for k, times in best_plan.pick_times.items():
                pick_events_map[(o, best_plan.station, k)] = times
        else:
            failed_orders.append(o)

    makespan = max((end for _, _, _, end in order_assignments.values()), default=0)
    total_moves = sum(len(evts) * 2 for evts in state.station_bin_events.values())
    feasible = len(failed_orders) == 0

    return Solution(
        order_assignments=order_assignments,
        bin_events=dict(state.station_bin_events),
        makespan=makespan,
        total_moves=total_moves,
        feasible=feasible,
        pick_events=pick_events_map,
    )


# ============================================================
# Step 5: Adaptive selection loop
# ============================================================

ORDER_ATTRS = [
    "sum_rt", "order_size", "sum_cycle", "max_rt",
    "sku_rarity", "sku_contention", "sharing_degree", "min_copies",
]
BIN_ATTRS = ["rt", "p", "cycle", "demand_ratio", "copies", "demand"]
DIRECTIONS = [True, False]  # True = descending, False = ascending


def _compute_objective(sol: Solution, ALPHA: float, BETA: float, S: list[int]) -> float:
    """Compute α·C_max + β·Σ_s(last_pick_s − first_pick_s)."""
    if not sol.feasible:
        return float("inf")
    c_max = float(sol.makespan)
    if BETA == 0.0:
        return ALPHA * c_max

    span_sum = 0.0
    for s in S:
        picks = [
            (start, end)
            for (ss, ln, start, end) in sol.order_assignments.values()
            if ss == s
        ]
        if picks:
            first = min(ps for ps, pe in picks)
            last = max(pe for ps, pe in picks)
            span_sum += last - first
    return ALPHA * c_max + BETA * span_sum


def run_ama_sgc(
        instance,
        *args,
        **kwargs
) -> tuple[Solution, tuple[str, bool, str, bool], list[dict]]:
    """Run Adaptive Multi-Attribute SGC.

    Tries multiple (order_sort, bin_sort) combinations and returns the solution
    with the lowest objective value.

    Args:
        mode: 'full_grid' (192 runs) or 'two_phase' (28 runs).
        verbose: if True, print winning combination.

    Returns:
        (best_solution, best_config, all_runs) where best_config is
        (order_attr_key, order_descending, bin_attr_key, bin_descending).
        all_runs includes dicts with performance metrics for each configuration.
    """
    if not isinstance(instance, Instance):
        S = instance
        L = args[0]
        K = args[1]
        O = args[2]
        orders_req = args[3]
        rt = args[4]
        rt_ret = args[5]
        p = args[6]
        N = args[7]
        horizon = kwargs.get('horizon', args[8] if len(args) > 8 else 10000)
        move_cap = kwargs.get('move_cap', args[9] if len(args) > 9 else None)
        ALPHA = kwargs.get('ALPHA', args[10] if len(args) > 10 else 1.0)
        BETA = kwargs.get('BETA', args[11] if len(args) > 11 else 0.0)
        mode = kwargs.get('mode', args[12] if len(args) > 12 else "full_grid")
        verbose = kwargs.get('verbose', args[13] if len(args) > 13 else False)
        instance = Instance(S, L, K, orders_req, rt, p, N, rt_ret=rt_ret)
    else:
        horizon = kwargs.get('horizon', args[0] if len(args) > 0 else 10000)
        move_cap = kwargs.get('move_cap', args[1] if len(args) > 1 else None)
        ALPHA = kwargs.get('ALPHA', args[2] if len(args) > 2 else 1.0)
        BETA = kwargs.get('BETA', args[3] if len(args) > 3 else 0.0)
        mode = kwargs.get('mode', args[4] if len(args) > 4 else "full_grid")
        verbose = kwargs.get('verbose', args[5] if len(args) > 5 else False)
        rt_ret = kwargs.get('rt_ret', args[6] if len(args) > 6 else None)

    S, L, K, orders_req, rt, p, N = instance
    O = instance.O
    if rt_ret is None:
        rt_ret = instance.rt_ret

    order_attrs, sku_attrs = precompute_attributes(instance, rt_ret=rt_ret)

    best_sol: Optional[Solution] = None
    best_obj = float("inf")
    best_config: tuple[str, bool, str, bool] = ("sharing_degree", False, "demand", True)
    all_runs: list[dict] = []

    def record_run(phase: str, oa: str, od: bool, ba: str, bd: bool, sol_run: Solution, obj_val: float):
        all_runs.append({
            "phase": phase,
            "order_attr": oa, "order_desc": od,
            "bin_attr": ba, "bin_desc": bd,
            "feasible": sol_run.feasible,
            "makespan": float(sol_run.makespan) if sol_run.feasible else None,
            "total_moves": sol_run.total_moves if sol_run.feasible else None,
            "objective": float(obj_val) if sol_run.feasible else None,
        })

    if mode == "full_grid":
        for oa in ORDER_ATTRS:
            for od in DIRECTIONS:
                for ba in BIN_ATTRS:
                    for bd in DIRECTIONS:
                        sol = run_sgc_parameterised(
                            instance,
                            horizon,
                            move_cap,
                            ALPHA,
                            BETA,
                            order_attrs,
                            sku_attrs,
                            oa,
                            od,
                            ba,
                            bd,
                            rt_ret=rt_ret,
                        )
                        obj = _compute_objective(sol, ALPHA, BETA, S)
                        record_run("full_grid", oa, od, ba, bd, sol, obj)
                        if best_sol is None or obj < best_obj:
                            best_obj = obj
                            best_sol = sol
                            best_config = (oa, od, ba, bd)

    elif mode == "two_phase":
        # Phase 1: find best order sort (bin sort fixed to demand desc)
        best_oa, best_od = "sharing_degree", False
        phase1_obj = float("inf")
        for oa in ORDER_ATTRS:
            for od in DIRECTIONS:
                sol = run_sgc_parameterised(
                    instance,
                    horizon,
                    move_cap,
                    ALPHA,
                    BETA,
                    order_attrs,
                    sku_attrs,
                    oa,
                    od,
                    "demand",
                    True,
                    rt_ret=rt_ret,
                )
                obj = _compute_objective(sol, ALPHA, BETA, S)
                record_run("two_phase_1", oa, od, "demand", True, sol, obj)
                if best_sol is None or obj < phase1_obj:
                    phase1_obj = obj
                    best_sol = sol
                    best_oa, best_od = oa, od

        # Phase 2: find best bin sort (order sort fixed to best from phase 1)
        best_ba, best_bd = "demand", True
        for ba in BIN_ATTRS:
            for bd in DIRECTIONS:
                sol = run_sgc_parameterised(
                    instance,
                    horizon,
                    move_cap,
                    ALPHA,
                    BETA,
                    order_attrs,
                    sku_attrs,
                    best_oa,
                    best_od,
                    ba,
                    bd,
                    rt_ret=rt_ret,
                )
                obj = _compute_objective(sol, ALPHA, BETA, S)
                record_run("two_phase_2", best_oa, best_od, ba, bd, sol, obj)
                if obj < best_obj:
                    best_obj = obj
                    best_sol = sol
                    best_config = (best_oa, best_od, ba, bd)

    else:
        raise ValueError(f"Unknown mode: {mode!r}. Choose 'full_grid' or 'two_phase'.")

    if verbose:
        oa, od, ba, bd = best_config
        def dir_str(d): return "desc" if d else "asc"
        print(
            f"[AMA-SGC] Best config: order={oa}({dir_str(od)}), "
            f"bin={ba}({dir_str(bd)}), obj={best_obj:.1f}"
        )

    assert best_sol is not None
    return best_sol, best_config, all_runs


# ============================================================
# Step 6: solve_heuristic_instance wrapper
# ============================================================

def solve_heuristic_instance(config: dict, return_raw: bool = False):
    """Run one AMA-SGC instance described by *config*.

    Accepts the same config keys as the base SGC's ``solve_heuristic_instance``
    plus optional 'mode' ('full_grid' or 'two_phase').
    Returns a dict with the same top-level keys so benchmark scripts can
    treat both implementations interchangeably.

    Returns::

        {
            'status':          'Feasible' | 'Infeasible',
            'solve_time':      float,
            'objective_value': float | None,
            'num_vars':        0,
            'progress':        [],
            'total_moves':     int,
            'winning_config':  (order_attr, order_desc, bin_attr, bin_desc),
        }
    """
    import time
    from datagen import generate_data

    num_stations = config.get("stations", 1)
    lanes_per_station = config.get("lanes", 2)
    num_orders = config.get("orders", 7)
    num_skus = config.get("skus", 5)
    seed = config.get("seed", 42)
    pick_touch_time = config.get("pick", 4)
    horizon = config.get("horizon", 10000)
    move_cap = config.get("movecap", None)
    alpha = config.get("alpha", 1.0)
    beta = config.get("beta", 0.0)
    mode = config.get("mode", "full_grid")

    instance = generate_data(
        num_stations=num_stations,
        lanes_per_station=lanes_per_station,
        num_orders=num_orders,
        num_skus=num_skus,
        seed=seed,
        pick_touch_time=pick_touch_time,
    )
    S, L, K, orders_req, rt, p, N = instance
    O = instance.O

    t0 = time.perf_counter()
    sol, best_config, all_runs = run_ama_sgc(
        instance,
        horizon=horizon, move_cap=move_cap, ALPHA=alpha, BETA=beta,
        mode=mode,
    )
    elapsed = time.perf_counter() - t0
    status = "Feasible" if sol.feasible else "Infeasible"
    violations = validate_solution(
        sol, instance, horizon=horizon, move_cap=move_cap
    )
    if violations:
        print(f"VALIDATION FAILED ({len(violations)} violations)")
        status = "Invalid"
        for v in violations[:10]:
            print(f"  Violation: {v}")
    else:
        print("Validation PASSED")

    res = {
        "status": status,
        "solve_time": elapsed,
        "objective_value": float(sol.makespan) if sol.feasible else None,
        "num_vars": 0,
        "progress": [],
        "total_moves": sol.total_moves,
        "winning_config": best_config,
        "all_runs": all_runs,
    }
    return (res, sol) if return_raw else res


# ============================================================
# Step 7: CLI entry point
# ============================================================

def main() -> None:
    """CLI entry point for running the AMA-SGC heuristic standalone."""
    import argparse
    import time
    from datagen import generate_data

    ap = argparse.ArgumentParser(
        description="AMA-SGC Heuristic for AutoStore Task B (v5)"
    )
    ap.add_argument("--stations", type=int, default=4)
    ap.add_argument("--lanes", type=int, default=4)
    ap.add_argument("--orders", type=int, default=160)
    ap.add_argument("--skus", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pick", type=int, default=4)
    ap.add_argument("--movecap", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument(
        "--mode", choices=["full_grid", "two_phase"], default="full_grid",
        help="full_grid: 192 runs; two_phase: 28 runs (faster, may miss interactions)",
    )
    ap.add_argument("--verbose", action="store_true", help="Print winning attribute combination")
    ap.add_argument("--no_vis", action="store_true", help="Skip HTML schedule visualisation")
    args = ap.parse_args()

    print("Generating data...")
    instance = generate_data(
        num_stations=args.stations,
        lanes_per_station=args.lanes,
        num_orders=args.orders,
        num_skus=args.skus,
        seed=args.seed,
        pick_touch_time=args.pick,
    )
    S, L, K, orders_req, rt, p, N = instance
    O = instance.O

    print(f"Stations={len(S)}, Lanes={len(L)}, SKUs={len(K)}, Orders={len(O)}, RobotLimit={args.movecap}\n")

    print(f"\nRunning AMA-SGC heuristic (alpha={args.alpha}, beta={args.beta}, mode={args.mode})...")
    t0 = time.perf_counter()
    sol, best_config, all_runs = run_ama_sgc(
        instance,
        horizon=args.horizon, move_cap=args.movecap,
        ALPHA=args.alpha, BETA=args.beta,
        mode=args.mode, verbose=True,
    )
    elapsed = time.perf_counter() - t0

    oa, od, ba, bd = best_config
    def dir_str(d): return "desc" if d else "asc"
    print(f"\n=== AMA-SGC Result ===")
    print(f"Feasible:    {sol.feasible}")
    print(f"Makespan:    {sol.makespan}")
    print(f"Total bin events (moves/2): {sol.total_moves // 2}")
    print(f"Time:        {elapsed:.4f}s")
    print(f"Winning:     order={oa}({dir_str(od)}), bin={ba}({dir_str(bd)})")

    violations = validate_solution(
        sol, instance, horizon=args.horizon, move_cap=args.movecap
    )
    if violations:
        print(f"VALIDATION FAILED ({len(violations)} violations)")
        for v in violations[:10]:
            print(f"  Violation: {v}")
    else:
        print("Validation PASSED")

    if not args.no_vis:
        try:
            from schedule_visualizer import plot_schedule
            mock_sol, handles = build_viz_handles(
                sol, instance
            )
            plot_schedule(mock_sol, handles)
        except Exception as exc:
            print(f"[VIS] Skipped: {exc}")


if __name__ == "__main__":
    main()
