from dataclasses import dataclass
from typing import Optional, TypedDict
import copy

from autostore_heuristic import (
    BinEvent, DifferenceArray, BinCopyPool, Solution, build_viz_handles, validate_solution
)


class OrderPlan(TypedDict):
    station: int
    lane: int
    admission_time: int
    completion_time: int
    pick_times_dict: dict[int, tuple[int, int]]
    fetch_times_dict: dict[int, tuple[int, int]]


@dataclass
class SKUDemand:
    sku: int
    orders: list[int]
    total_weight: float
    n_orders: int


@dataclass
class LaneSlot:
    lane: int
    station: int
    current_order: Optional[int]
    free_at: int


@dataclass
class GBSState:
    lane_slots: list[LaneSlot]
    active_orders: set[int]
    pending_picks: dict[int, set[int]]
    pickface_free: int
    bin_events: list[BinEvent]
    order_pick_times: dict[tuple[int, int], tuple[int, int]]
    order_windows: dict[int, tuple[int, int]]
    move_da: DifferenceArray
    bin_pools: dict[int, BinCopyPool]
    move_cap: Optional[int]
    order_admission_times: dict[int, int]
    order_lanes: dict[int, int]

    def clone(self) -> 'GBSState':
        return copy.deepcopy(self)


def precompute_sku_demand(
    O: list[int],
    orders_req: dict[int, list[int]],
    rt: dict[int, int],
    rt_ret: dict[int, int]
) -> dict[int, SKUDemand]:
    demand: dict[int, SKUDemand] = {}
    for o in O:
        for k in orders_req[o]:
            if k not in demand:
                demand[k] = SKUDemand(sku=k, orders=[], total_weight=float(rt[k] + rt_ret[k]), n_orders=0)
            demand[k].orders.append(o)
            demand[k].n_orders += 1
    return demand


def seed_initial_lanes(
    S: list[int], L: list[int], O: list[int],
    orders_req: dict[int, list[int]],
    rt: dict[int, int], rt_ret: dict[int, int], p: dict[int, int]
) -> tuple[dict[int, int], list[int]]:

    def sum_rt(o: int) -> int:
        return sum(rt[k] for k in orders_req[o])

    sorted_orders = sorted(O, key=sum_rt)

    if not sorted_orders:
        return {}, []

    assignment: dict[int, int] = {}
    active_set: set[int] = set()
    unscheduled = list(sorted_orders)

    anchor = unscheduled.pop(0)
    assignment[anchor] = L[0]
    active_set.add(anchor)

    for lane_idx in range(1, min(len(L), len(O))):
        best_order = None
        best_score = -1.0
        best_tiebreak = float('inf')

        active_skus = set()
        for ao in active_set:
            active_skus.update(orders_req[ao])

        for cand in unscheduled:
            overlap = set(orders_req[cand]).intersection(active_skus)
            score = sum(rt[k] + rt_ret[k] for k in overlap)
            tiebreak = sum_rt(cand)

            if score > best_score or (score == best_score and tiebreak < best_tiebreak):
                best_score = score
                best_tiebreak = tiebreak
                best_order = cand

        if best_order is not None:
            assignment[best_order] = L[lane_idx]
            active_set.add(best_order)
            unscheduled.remove(best_order)

    return assignment, unscheduled


def admit_next_order(
    freed_lane: LaneSlot,
    unscheduled: list[int],
    active_orders: set[int],
    pending_picks: dict[int, set[int]],
    orders_req: dict[int, list[int]],
    rt: dict[int, int], rt_ret: dict[int, int],
) -> Optional[int]:
    if not unscheduled:
        return None

    active_remaining_skus = set()
    for o in active_orders:
        active_remaining_skus.update(pending_picks[o])

    best_order = None
    best_score = -1.0
    best_tiebreak = float('inf')

    for cand in unscheduled:
        overlap = set(orders_req[cand]).intersection(active_remaining_skus)
        score = sum(rt[k] + rt_ret[k] for k in overlap)
        tiebreak = sum(rt[k] for k in orders_req[cand])

        if score > best_score or (score == best_score and tiebreak < best_tiebreak):
            best_score = score
            best_tiebreak = tiebreak
            best_order = cand

    return best_order


@dataclass
class BinTask:
    sku: int
    concurrent_orders: list[int]
    n_sharing: int
    sharing_savings: float
    earliest_fetch_start: int
    copy_id: int
    score: float


def score_bin_task(
    task: BinTask,
    gbs_state: GBSState,
    orders_req: dict[int, list[int]],
    rt: dict[int, int], rt_ret: dict[int, int], p: dict[int, int],
    scoring_rule: str,
) -> float:
    k = task.sku
    if scoring_rule == "max_sharing":
        return float(task.n_sharing * (rt[k] + rt_ret[k]))

    elif scoring_rule == "critical_path":
        max_rem = 0
        for o in task.concurrent_orders:
            rem = sum(rt[k2] + p[k2] + rt_ret[k2] for k2 in gbs_state.pending_picks[o])
            if rem > max_rem:
                max_rem = rem
        return float(max_rem)

    elif scoring_rule == "readiness_weighted":
        fetch_arrival = task.earliest_fetch_start + rt[k]
        idle_gap = max(0, fetch_arrival - gbs_state.pickface_free)
        base_score = task.n_sharing * (rt[k] + rt_ret[k])
        penalty = 1.0  # Assuming PENALTY=1.0 for simplicity
        return float(base_score - penalty * idle_gap)

    return 0.0


def build_bin_tasks(
    gbs_state: GBSState,
    orders_req: dict[int, list[int]],
    rt: dict[int, int], rt_ret: dict[int, int], p: dict[int, int],
    N: dict[int, int],
    horizon: int,
    scoring_rule: str
) -> list[BinTask]:

    needed_skus = set()
    for o in gbs_state.active_orders:
        needed_skus.update(gbs_state.pending_picks[o])

    tasks = []
    for k in needed_skus:
        concurrent = [o for o in gbs_state.active_orders if k in gbs_state.pending_picks[o]]
        n_sharing = len(concurrent)

        # Determine earliest fetch based on pool (Simplified check for now)
        earliest_copy_free, copy_id = gbs_state.bin_pools[k].get_earliest_free()
        earliest_fetch = max(gbs_state.pickface_free - rt[k], earliest_copy_free)

        sharing_savings = float((n_sharing - 1) * (rt[k] + p[k] + rt_ret[k]))

        task = BinTask(
            sku=k, concurrent_orders=concurrent, n_sharing=n_sharing,
            sharing_savings=sharing_savings, earliest_fetch_start=earliest_fetch,
            copy_id=copy_id, score=0.0
        )
        task.score = score_bin_task(task, gbs_state, orders_req, rt, rt_ret, p, scoring_rule)
        tasks.append(task)

    tasks.sort(key=lambda t: t.score, reverse=True)
    return tasks


def schedule_bin_event(
    task: BinTask,
    gbs_state: GBSState,
    rt: dict[int, int], rt_ret: dict[int, int], p: dict[int, int],
    horizon: int,
) -> Optional[BinEvent]:
    t_fetch = task.earliest_fetch_start
    if gbs_state.move_cap is not None:
        t_fetch = gbs_state.move_da.delay_for_movecap(t_fetch, rt[task.sku], gbs_state.move_cap)

    t_fetch_end = t_fetch + rt[task.sku]
    t_presence_start = t_fetch_end
    t_presence_end = t_presence_start + p[task.sku]
    t_return_start = t_presence_end

    if gbs_state.move_cap is not None:
        t_return_start = gbs_state.move_da.delay_for_movecap(t_return_start, rt_ret[task.sku], gbs_state.move_cap)
        t_presence_end = t_return_start

    t_return_end = t_return_start + rt_ret[task.sku]

    if t_return_end > horizon:
        return None

    if gbs_state.move_cap is not None:
        gbs_state.move_da.add_move(t_fetch, t_fetch_end)
        gbs_state.move_da.add_move(t_return_start, t_return_end)

    ev = BinEvent(
        sku=task.sku, copy_id=task.copy_id,
        fetch_start=t_fetch, fetch_end=t_fetch_end,
        presence_start=t_presence_start, presence_end=t_presence_end,
        return_start=t_return_start, return_end=t_return_end,
        orders_served=list(task.concurrent_orders)
    )

    gbs_state.bin_pools[task.sku].commit(task.copy_id, t_return_end)
    gbs_state.pickface_free = t_presence_end
    gbs_state.bin_events.append(ev)

    for o in task.concurrent_orders:
        gbs_state.order_pick_times[(o, task.sku)] = (t_presence_start, t_presence_end)
        gbs_state.pending_picks[o].remove(task.sku)

        # update windows
        old_win = gbs_state.order_windows.get(o, None)
        if old_win is None:
            gbs_state.order_windows[o] = (t_presence_start, t_presence_end)
        else:
            gbs_state.order_windows[o] = (min(old_win[0], t_presence_start), max(old_win[1], t_presence_end))

    return ev


def check_completions(
    gbs_state: GBSState,
    unscheduled: list[int],
    orders_req: dict[int, list[int]],
    rt: dict[int, int], rt_ret: dict[int, int],
) -> list[int]:
    newly_admitted = []
    completed = [o for o in gbs_state.active_orders if not gbs_state.pending_picks[o]]

    for o in completed:
        gbs_state.active_orders.remove(o)
        freed_lane = None
        for ls in gbs_state.lane_slots:
            if ls.current_order == o:
                ls.current_order = None
                ls.free_at = gbs_state.order_windows[o][1]
                freed_lane = ls
                break

        if freed_lane and unscheduled:
            new_o = admit_next_order(freed_lane, unscheduled, gbs_state.active_orders,
                                     gbs_state.pending_picks, orders_req, rt, rt_ret)
            if new_o is not None:
                unscheduled.remove(new_o)
                gbs_state.active_orders.add(new_o)
                gbs_state.pending_picks[new_o] = set(orders_req[new_o])
                freed_lane.current_order = new_o
                gbs_state.order_admission_times[new_o] = freed_lane.free_at
                gbs_state.order_lanes[new_o] = freed_lane.lane
                newly_admitted.append(new_o)

    return newly_admitted


def run_gbs(
    S: list[int], L: list[int], K: list[int], O: list[int],
    orders_req: dict[int, list[int]],
    rt: dict[int, int], rt_ret: dict[int, int], p: dict[int, int],
    N: dict[int, int],
    horizon: int,
    move_cap: Optional[int] = None,
    ALPHA: float = 1.0,
    BETA: float = 0.0,
    scoring_rule: str = "max_sharing",
) -> Solution:
    # 1. Simple greedy partition of orders to stations to preserve sharing while balancing load
    O_sorted = sorted(O, key=lambda o: sum(rt[k] for k in orders_req[o]), reverse=True)
    station_orders = {s: [] for s in S}
    station_skus = {s: set() for s in S}
    station_rt_load = {s: 0.0 for s in S}

    for o in O_sorted:
        o_skus = set(orders_req[o])
        o_rt = sum(rt[k] for k in o_skus)

        best_s = S[0]
        best_score = -float('inf')
        for s in S:
            overlap_rt = sum(rt[k] for k in station_skus[s].intersection(o_skus))
            # Tiebreak by negative load to distribute orders evenly.
            # We strongly penalize overloaded stations to prevent piling all orders into S0.
            score = overlap_rt - station_rt_load[s]
            if score > best_score:
                best_score = score
                best_s = s

        station_orders[best_s].append(o)
        station_skus[best_s].update(o_skus)
        station_rt_load[best_s] += o_rt

    # 2. Shared globals across stations
    global_move_da = DifferenceArray(horizon)
    global_bin_pools = {k: BinCopyPool(k, N[k]) for k in K}

    all_order_plans: dict[int, OrderPlan] = {}
    all_bin_events: dict[int, list[BinEvent]] = {}

    # 3. Schedule each station independently
    for station in S:
        O_s = station_orders[station]
        if not O_s:
            all_bin_events[station] = []
            continue

        assignment, unscheduled = seed_initial_lanes([station], L, O_s, orders_req, rt, rt_ret, p)

        lane_slots = [LaneSlot(l, station, None, 0) for l in L]
        for o, l in assignment.items():
            for ls in lane_slots:
                if ls.lane == l:
                    ls.current_order = o
                    break

        gbs_state = GBSState(
            lane_slots=lane_slots,
            active_orders=set(assignment.keys()),
            pending_picks={o: set(orders_req[o]) for o in assignment.keys()},
            pickface_free=0,
            bin_events=[],
            order_pick_times={},
            order_windows={},
            move_da=global_move_da,
            bin_pools=global_bin_pools,
            move_cap=move_cap,
            order_admission_times={o: 0 for o in assignment.keys()},
            order_lanes={o: l for o, l in assignment.items()}
        )

        while gbs_state.active_orders or unscheduled:
            tasks = build_bin_tasks(gbs_state, orders_req, rt, rt_ret, p, N, horizon, scoring_rule)
            if not tasks:
                break

            ev = None
            for best_task in tasks:
                ev = schedule_bin_event(best_task, gbs_state, rt, rt_ret, p, horizon)
                if ev is not None:
                    break

            if ev is None:
                break

            check_completions(gbs_state, unscheduled, orders_req, rt, rt_ret)

        # Collect station's order plans
        for o in O_s:
            all_order_plans[o] = {
                "station": station,
                "lane": gbs_state.order_lanes.get(o, 0),
                # "admission_time": gbs_state.order_admission_times.get(o, 0),
                "admission_time": gbs_state.order_windows.get(o, (0, 0))[0],
                "completion_time": gbs_state.order_windows.get(o, (0, 0))[1],
                "pick_times_dict": {k: gbs_state.order_pick_times.get((o, k), (0, 0)) for k in orders_req[o]},
                "fetch_times_dict": {}
            }

        all_bin_events[station] = gbs_state.bin_events

    feasible = True
    for o in O:
        if o not in all_order_plans or len(all_order_plans[o]["pick_times_dict"]) < len(orders_req[o]):
            feasible = False

    makespan = max(op["completion_time"] for op in all_order_plans.values()) if all_order_plans else 0
    total_moves = sum(len(bevs) for bevs in all_bin_events.values()) * 2

    return Solution(
        feasible=feasible,
        makespan=makespan,
        total_moves=total_moves,
        order_assignments={o: (op["station"], op["lane"], op["admission_time"], op["completion_time"])
                           for o, op in all_order_plans.items()},
        pick_events={(o, op["station"], k): op["pick_times_dict"].get(k, (0, 0))
                     for o, op in all_order_plans.items() for k in orders_req[o]},
        bin_events=all_bin_events
    )


def run_gbs_adaptive(
    S: list[int], L: list[int], K: list[int], O: list[int],
    orders_req: dict[int, list[int]],
    rt: dict[int, int], rt_ret: dict[int, int], p: dict[int, int],
    N: dict[int, int],
    horizon: int,
    move_cap: Optional[int] = None,
    ALPHA: float = 1.0,
    BETA: float = 0.0,
) -> Solution:
    best_sol = None
    best_obj = float('inf')
    best_moves = float('inf')

    for rule in ["max_sharing", "critical_path", "readiness_weighted"]:
        orders_req_c = copy.deepcopy(orders_req)
        rt_c = copy.deepcopy(rt)
        rt_ret_c = copy.deepcopy(rt_ret)
        p_c = copy.deepcopy(p)
        N_c = copy.deepcopy(N)
        O_c = list(O)
        S_c = list(S)
        L_c = list(L)
        K_c = list(K)

        sol = run_gbs(S_c, L_c, K_c, O_c, orders_req_c, rt_c, rt_ret_c, p_c,
                      N_c, horizon, move_cap, ALPHA, BETA, scoring_rule=rule)
        if not sol.feasible:
            continue

        flow_time = sum(sol.order_assignments[o][3] - sol.order_assignments[o][2] for o in O_c)
        obj = ALPHA * sol.makespan + BETA * flow_time
        moves = sol.total_moves

        if obj < best_obj or (obj == best_obj and moves < best_moves):
            best_sol = sol
            best_obj = obj
            best_moves = moves

    return best_sol


def solve_heuristic_instance(config: dict, return_raw: bool = False):
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
    gbs_rule = config.get("gbs_rule", "readiness_weighted")

    S, L, K, orders_req, rt, p, N = generate_data(
        num_stations=num_stations,
        lanes_per_station=lanes_per_station,
        num_orders=num_orders,
        num_skus=num_skus,
        seed=seed,
        pick_touch_time=pick_touch_time,
    )
    rt_ret = dict(rt)
    O = sorted(orders_req.keys())

    t0 = time.perf_counter()
    if gbs_rule == "adaptive":
        sol = run_gbs_adaptive(
            S, L, K, O,
            orders_req, rt, rt_ret, p,
            N, horizon, move_cap,
            alpha, beta
        )
    else:
        sol = run_gbs(
            S, L, K, O,
            orders_req, rt, rt_ret, p,
            N, horizon, move_cap,
            alpha, beta,
            scoring_rule=gbs_rule
        )
    elapsed = time.perf_counter() - t0
    status = "Feasible" if sol.feasible else "Infeasible"
    violations = validate_solution(
        sol, S, L, K, O, orders_req, rt, rt_ret, p, N,
        horizon=horizon, move_cap=move_cap,
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
    }
    return (res, sol) if return_raw else res


def main() -> None:
    """CLI entry point for running the GBS heuristic standalone."""
    import argparse
    import time
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from datagen import generate_data

    ap = argparse.ArgumentParser(
        description="Global Bin Scheduling (GBS) Heuristic for AutoStore Task B"
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
        "--rule", choices=["max_sharing", "critical_path", "readiness_weighted", "adaptive"],
        default="readiness_weighted",
        help="Bin task scoring rule",
    )
    ap.add_argument("--no_vis", action="store_true", help="Skip HTML schedule visualisation")
    args = ap.parse_args()

    print("Generating data...")
    S, L, K, orders_req, rt, p, N = generate_data(
        num_stations=args.stations,
        lanes_per_station=args.lanes,
        num_orders=args.orders,
        num_skus=args.skus,
        seed=args.seed,
        pick_touch_time=args.pick,
    )
    rt_ret = dict(rt)
    O = sorted(orders_req.keys())

    print(f"Stations={len(S)}, Lanes={len(L)}, SKUs={len(K)}, Orders={len(O)}, RobotLimit={args.movecap}\n")

    print(f"\nRunning GBS heuristic (rule={args.rule}, alpha={args.alpha}, beta={args.beta})...")
    t0 = time.perf_counter()
    if args.rule == 'adaptive':
        sol = run_gbs_adaptive(
            S, L, K, O, orders_req, rt, rt_ret, p, N,
            horizon=args.horizon, move_cap=args.movecap,
            ALPHA=args.alpha, BETA=args.beta
        )
    else:
        sol = run_gbs(
            S, L, K, O, orders_req, rt, rt_ret, p, N,
            horizon=args.horizon, move_cap=args.movecap,
            ALPHA=args.alpha, BETA=args.beta,
            scoring_rule=args.rule
        )
    elapsed = time.perf_counter() - t0

    print(f"\n=== GBS Result ===")
    print(f"Feasible:    {sol.feasible}")
    print(f"Makespan:    {sol.makespan}")
    print(f"Total bin events (moves/2): {sol.total_moves // 2}")
    print(f"Time:        {elapsed:.4f}s")

    violations = validate_solution(
        sol, S, L, K, O, orders_req, rt, rt_ret, p, N,
        horizon=args.horizon, move_cap=args.movecap,
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
                sol, S, L, K, O, orders_req, rt, rt_ret, p,
            )
            plot_schedule(mock_sol, handles)
        except Exception as exc:
            print(f"[VIS] Skipped: {exc}")


if __name__ == "__main__":
    main()
