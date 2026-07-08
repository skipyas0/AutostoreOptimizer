from collections import defaultdict
from autostore_heuristic import find_shared_bin, earliest_feasible_fetch, BinEvent, OrderPlan, HeuristicState, Solution
import argparse
import time
from autostore_heuristic import commit_plan, validate_solution, init_state
from typing import Literal
from dataclasses import dataclass
from typing import Optional, Dict, Set, List


def build_sku_order_index(orders_req: Dict[int, List[int]]) -> Dict[int, Set[int]]:
    idx = {}
    for o, skus in orders_req.items():
        for k in skus:
            if k not in idx:
                idx[k] = set()
            idx[k].add(o)
    return idx


def compute_regret_k(station_scores: Dict[int, float], k: int) -> float:
    feasible = sorted(s for s in station_scores.values() if s < float('inf'))
    if len(feasible) <= 1:
        return float('inf')
    best = feasible[0]
    return sum(feasible[i] - best for i in range(1, min(k, len(feasible))))


@dataclass
class InsertionScores:
    order: int
    station_scores: Dict[int, float]
    station_plans: Dict[int, Optional[OrderPlan]]
    best_score: float
    second_score: float
    best_station: int
    best_plan: Optional[OrderPlan]
    regret: float


def find_earliest_gap(intervals: List[tuple[int, int]], ready_time: int, duration: int) -> int:
    current_time = ready_time
    for start, end in intervals:
        if current_time + duration <= start:
            return current_time
        current_time = max(current_time, end)
    return current_time


def gap_filling_plan_order_at_station(
        o: int, s: int, state: HeuristicState, orders_req: Dict[int, List[int]],
        rt: Dict[int, int], rt_ret: Dict[int, int], p: Dict[int, int],
        N: Dict[int, int], horizon: int, ALPHA: float, BETA: float, demand_count: Dict[int, int],
) -> Optional[OrderPlan]:
    L_at_s = [ln for (ss, ln) in state.lane_free if ss == s]
    if not L_at_s:
        return None
    best_lane = min(L_at_s, key=lambda ln: state.lane_free[(s, ln)])
    t_order_start = state.lane_free[(s, best_lane)]

    skus_sorted = sorted(orders_req[o], key=lambda k: demand_count.get(k, 0), reverse=True)
    t_cursor = t_order_start

    new_bin_events: List[BinEvent] = []
    shared_picks: List[tuple[int, BinEvent, int]] = []
    pending_moves: List[tuple[int, int]] = []
    pending_copies: Dict[int, List[tuple[int, int]]] = defaultdict(list)
    pick_times_dict: Dict[int, tuple[int, int]] = {}

    local_intervals = list(state.pickface_intervals[s])

    for k in skus_sorted:
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

        desired_presence = max(t_cursor, rt[k])

        while True:
            gap_start = find_earliest_gap(local_intervals, desired_presence, p[k])
            desired_fetch = gap_start - rt[k]

            try:
                t_fetch, copy_id = earliest_feasible_fetch(
                    k, s, desired_fetch, state, rt, rt_ret,
                    pending_moves, pending_copies, horizon,
                )
            except ValueError:
                return None

            if t_fetch > desired_fetch:
                desired_presence = t_fetch + rt[k]
                continue

            fetch_end = t_fetch + rt[k]
            presence_start = fetch_end
            pick_start = presence_start
            pick_end = pick_start + p[k]
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
                    if gap_start != find_earliest_gap(local_intervals, presence_start, presence_end - presence_start):
                        desired_presence = presence_start + 1
                        continue

            break

        t_cursor = pick_end
        pick_times_dict[k] = (pick_start, pick_end)
        local_intervals.append((presence_start, presence_end))
        local_intervals.sort(key=lambda x: x[0])

        ev = BinEvent(
            sku=k, copy_id=copy_id, fetch_start=t_fetch, fetch_end=fetch_end,
            presence_start=presence_start, presence_end=presence_end,
            return_start=return_start, return_end=return_end, orders_served=[o],
        )
        new_bin_events.append(ev)
        pending_moves.append((t_fetch, fetch_end))
        pending_moves.append((return_start, return_end))
        pending_copies[k].append((copy_id, return_end))

    order_start = min((ps for ps, pe in pick_times_dict.values()), default=t_order_start)
    order_end = max((pe for ps, pe in pick_times_dict.values()), default=t_order_start)
    score = ALPHA * order_end + BETA * (order_end - order_start)

    return OrderPlan(
        order=o, station=s, lane=best_lane, start=order_start, end=order_end,
        bin_events=new_bin_events, shared_picks=shared_picks, score=score,
        pick_times=pick_times_dict,
    )


def score_order_insertions(
    o: int, S: List[int], state: HeuristicState,
    orders_req: Dict[int, List[int]], rt: Dict[int, int],
    rt_ret: Dict[int, int], p: Dict[int, int], N: Dict[int, int],
    horizon: int, ALPHA: float, BETA: float, demand_count: Dict[int, int],
    regret_k: int
) -> InsertionScores:
    station_scores = {}
    station_plans = {}

    for s in S:
        plan = gap_filling_plan_order_at_station(
            o, s, state, orders_req, rt, rt_ret, p, N, horizon, ALPHA, BETA, demand_count
        )
        if plan is not None:
            station_scores[s] = plan.score
            station_plans[s] = plan
        else:
            station_scores[s] = float('inf')
            station_plans[s] = None

    feasible = sorted((score, s) for s, score in station_scores.items() if score < float('inf'))

    if not feasible:
        return InsertionScores(o, station_scores, station_plans, float('inf'), float('inf'), -1, None, -1.0)

    best_score, best_station = feasible[0]
    second_score = feasible[1][0] if len(feasible) > 1 else float('inf')
    best_plan = station_plans[best_station]
    regret = compute_regret_k(station_scores, regret_k)

    return InsertionScores(o, station_scores, station_plans, best_score, second_score, best_station, best_plan, regret)


def compute_dirty_set(
    committed_order: int, committed_station: int,
    orders_req: Dict[int, List[int]], sku_order_index: Dict[int, Set[int]],
    cached_scores: Dict[int, InsertionScores], unscheduled: Set[int],
    state: Optional[HeuristicState] = None, committed_plan: Optional[OrderPlan] = None
) -> Set[int]:
    dirty = set()
    for k in orders_req[committed_order]:
        if k in sku_order_index:
            dirty.update(sku_order_index[k])

    for o in unscheduled:
        if o in cached_scores and cached_scores[o].best_station == committed_station:
            dirty.add(o)

    if state is not None and committed_plan is not None and state.move_cap is not None:
        # Find all intervals consumed by the committed plan
        committed_intervals = []
        for ev in committed_plan.bin_events:
            committed_intervals.append((ev.fetch_start, ev.fetch_end))
            committed_intervals.append((ev.return_start, ev.return_end))

        for o in unscheduled:
            if o in dirty:
                continue
            plan = cached_scores[o].best_plan
            if plan is None:
                continue
            overlap = False
            for ev in plan.bin_events:
                for c_start, c_end in committed_intervals:
                    # Check overlap of fetch
                    if max(ev.fetch_start, c_start) < min(ev.fetch_end, c_end):
                        overlap = True
                        break
                    # Check overlap of return
                    if max(ev.return_start, c_start) < min(ev.return_end, c_end):
                        overlap = True
                        break
                if overlap:
                    break
            if overlap:
                dirty.add(o)

    return dirty & unscheduled


def run_rdi_sgc(
    S: List[int], L: List[int], K: List[int], O: List[int],
    orders_req: Dict[int, List[int]], rt: Dict[int, int],
    rt_ret: Dict[int, int], p: Dict[int, int], N: Dict[int, int],
    horizon: int, move_cap: Optional[int] = None,
    ALPHA: float = 1.0, BETA: float = 0.0,
    regret_k: int = 2, use_lazy: bool = True,
    tiebreaker: Literal["sharing_degree", "best_score", "sum_rt_asc"] = "sharing_degree"
) -> Solution:
    state = init_state(S, L, K, N, horizon, move_cap)

    demand_count = defaultdict(int)
    for reqs in orders_req.values():
        for k in reqs:
            demand_count[k] += 1

    sku_index = build_sku_order_index(orders_req)

    req_sets = {o: set(req) for o, req in orders_req.items()}

    def get_sharing_degree(o: int) -> int:
        req = req_sets[o]
        return sum(1 for o2 in O if o2 != o and not req.isdisjoint(req_sets[o2]))

    def get_sum_rt(o: int) -> int:
        return sum(rt[k] for k in req_sets[o])

    sharing_degrees = {o: get_sharing_degree(o) for o in O}
    sum_rts = {o: get_sum_rt(o) for o in O}

    unscheduled = set(O)
    cached_scores: Dict[int, InsertionScores] = {}

    for o in O:
        cached_scores[o] = score_order_insertions(
            o, S, state, orders_req, rt, rt_ret, p, N, horizon, ALPHA, BETA, demand_count, regret_k
        )

    order_assignments = {}
    pick_events_map = {}

    while unscheduled:
        best_o = None
        best_regret = -float('inf')

        for o in unscheduled:
            c = cached_scores[o]
            r = c.regret
            if r > best_regret:
                best_regret = r
                best_o = o
            elif r == best_regret and best_o is not None:
                if tiebreaker == "sharing_degree":
                    # Priority to LOW sharing degree
                    if sharing_degrees[o] < sharing_degrees[best_o]:
                        best_o = o
                elif tiebreaker == "sum_rt_asc":
                    # Priority to LOW sum_rt
                    if sum_rts[o] < sum_rts[best_o]:
                        best_o = o
                else:
                    # Priority to HIGH best_score
                    if c.best_score > cached_scores[best_o].best_score:
                        best_o = o

        if best_o is None or cached_scores[best_o].best_plan is None:
            break  # Failed or infeasible

        plan = cached_scores[best_o].best_plan
        commit_plan(plan, state)
        order_assignments[best_o] = (plan.station, plan.lane, plan.start, plan.end)
        for k, (ps, pe) in plan.pick_times.items():
            pick_events_map[(best_o, plan.station, k)] = (ps, pe)
        unscheduled.remove(best_o)

        if not unscheduled:
            break

        if use_lazy:
            dirty = compute_dirty_set(best_o, plan.station, orders_req, sku_index,
                                      cached_scores, unscheduled, state, plan)
        else:
            dirty = unscheduled

        for o in dirty:
            cached_scores[o] = score_order_insertions(
                o, S, state, orders_req, rt, rt_ret, p, N, horizon, ALPHA, BETA, demand_count, regret_k
            )

    # Post-process results into a Solution object
    final_makespan = max((e for (_, _, _, e) in order_assignments.values()), default=0)
    total_moves = sum(len(state.station_bin_events[s]) * 2 for s in S)
    feasible = (len(unscheduled) == 0) and (final_makespan <= horizon)

    return Solution(
        feasible=feasible,
        makespan=final_makespan,
        total_moves=total_moves,
        order_assignments=order_assignments,
        bin_events=state.station_bin_events,
        pick_events=pick_events_map
    )


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
    sol = run_rdi_sgc(
        S=S, L=L, K=K, O=O,
        orders_req=orders_req, rt=rt,
        rt_ret=rt_ret, p=p, N=N,
        horizon=horizon, move_cap=move_cap,
        ALPHA=alpha, BETA=beta,
        regret_k=config.get("regret_k", 2),
        use_lazy=config.get("use_lazy", True),
        tiebreaker=config.get("tiebreaker", "sum_rt_asc")
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


def main():
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from datagen import generate_data

    ap = argparse.ArgumentParser(description="Regret-Based Dynamic Insertion SGC")
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
    ap.add_argument("--regret_k", type=int, default=2)
    ap.add_argument("--no_lazy", action="store_true")
    ap.add_argument("--tiebreaker", type=str, choices=["sharing_degree",
                    "best_score", "sum_rt_asc"], default="sum_rt_asc")
    ap.add_argument("--no_vis", action="store_true")
    args = ap.parse_args()

    print("Generating data...")
    S, L, K, orders_req, rt, p, N = generate_data(
        num_stations=args.stations, lanes_per_station=args.lanes,
        num_orders=args.orders, num_skus=args.skus, seed=args.seed,
        pick_touch_time=args.pick
    )
    rt_ret = dict(rt)
    O = sorted(orders_req.keys())

    print(f"Stations={len(S)}, Lanes={len(L)}, SKUs={len(K)}, Orders={len(O)}, RobotLimit={args.movecap}\n")
    print(f"Running RDI-SGC (regret_k={args.regret_k}, tiebreaker={args.tiebreaker}, lazy={not args.no_lazy})")

    t0 = time.perf_counter()
    sol = run_rdi_sgc(
        S, L, K, O, orders_req, rt, rt_ret, p, N,
        horizon=args.horizon, move_cap=args.movecap,
        ALPHA=args.alpha, BETA=args.beta, regret_k=args.regret_k,
        use_lazy=not args.no_lazy, tiebreaker=args.tiebreaker
    )
    elapsed = time.perf_counter() - t0

    print(f"\n=== RDI-SGC Result ===")
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
            from autostore_heuristic import build_viz_handles
            mock_sol, handles = build_viz_handles(
                sol, S, L, K, O, orders_req, rt, rt_ret, p,
            )
            plot_schedule(mock_sol, handles)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"[VIS] Skipped: {exc}")


if __name__ == "__main__":
    main()
