import argparse
import time
from collections import defaultdict
from dataclasses import dataclass

from autostore_heuristic import (
    BinEvent,
    HeuristicState,
    OrderPlan,
    Solution,
    commit_plan,
    earliest_feasible_fetch,
    find_pick_slot,
    find_shared_bin,
    init_state,
    validate_solution,
)


def build_sku_order_index(orders_req: dict[int, list[int]]) -> dict[int, set[int]]:
    idx = {}
    for o, skus in orders_req.items():
        for k in skus:
            if k not in idx:
                idx[k] = set()
            idx[k].add(o)
    return idx


def compute_regret_k(station_scores: dict[int, float], k: int) -> float:
    feasible = sorted(s for s in station_scores.values() if s < float("inf"))
    if len(feasible) <= 1:
        return float("inf")
    best = feasible[0]
    return sum(feasible[i] - best for i in range(1, min(k, len(feasible))))


@dataclass
class InsertionScores:
    order: int
    station_scores: dict[int, float]
    station_plans: dict[int, OrderPlan | None]
    best_score: float
    second_score: float
    best_station: int
    best_plan: OrderPlan | None
    regret: float


def find_earliest_gap(
    intervals: list[tuple[int, int]], ready_time: int, duration: int
) -> int:
    current_time = ready_time
    for start, end in intervals:
        if current_time + duration <= start:
            return current_time
        current_time = max(current_time, end)
    return current_time


def gap_filling_plan_order_at_station(
    o: int,
    s: int,
    state: HeuristicState,
    orders_req: dict[int, list[int]],
    rt: dict[int, int],
    rt_ret: dict[int, int],
    p: dict[int, int],
    N: dict[int, int],
    horizon: int,
    ALPHA: float,
    BETA: float,
    demand_count: dict[int, int],
    pick_cap: int,
) -> OrderPlan | None:
    L_at_s = [ln for (ss, ln) in state.lane_free if ss == s]
    if not L_at_s:
        return None
    best_lane = min(L_at_s, key=lambda ln: state.lane_free[(s, ln)])
    t_order_start = state.lane_free[(s, best_lane)]

    remaining_skus = sorted(
        orders_req[o], key=lambda k: demand_count.get(k, 0), reverse=True
    )
    t_cursor = t_order_start

    new_bin_events: list[BinEvent] = []
    # (sku, bin_event, pick_start, pick_end, new_return_start, new_return_end)
    shared_picks: list[tuple[int, BinEvent, int, int, int, int]] = []
    pending_moves: list[tuple[int, int]] = []
    removed_moves: list[tuple[int, int]] = []
    pending_copies: dict[int, list[tuple[int, int]]] = defaultdict(list)
    pick_times_dict: dict[int, tuple[int, int]] = {}

    local_intervals = list(state.pickface_intervals[s])
    temp_pick_intervals: dict[int, list[tuple[int, int]]] = {}
    temp_bounds: dict[int, tuple[int, int]] = {}

    candidate_bins = state.station_bin_events[s]

    while remaining_skus:
        # 1. Dynamic Shared-First Check: see if any remaining SKU can share an existing bin
        best_shared_k = None
        best_shared_res = (None, float("inf"), None, None)

        for k_cand in remaining_skus:
            sh_ev, p_start, n_rs, n_re = find_shared_bin(
                candidate_bins + new_bin_events,
                k_cand,
                t_cursor,
                p,
                pick_cap,
                state=state,
                station=s,
                rt_ret=rt_ret,
                horizon=horizon,
                local_pickface_intervals=local_intervals,
                pending_moves=pending_moves,
                pending_copies=pending_copies,
                temp_pick_intervals=temp_pick_intervals,
                temp_bounds=temp_bounds,
                rt=rt,
                removed_moves=removed_moves,
            )
            if sh_ev is not None and p_start < best_shared_res[1]:
                best_shared_k = k_cand
                best_shared_res = (sh_ev, p_start, n_rs, n_re)

        if best_shared_k is not None:
            k = best_shared_k
            remaining_skus.remove(k)
            shared, pick_start, new_ret_start, new_ret_end = best_shared_res
            pick_end = pick_start + p[k]
            ev_id = id(shared)
            if ev_id not in temp_pick_intervals:
                temp_pick_intervals[ev_id] = list(shared.pick_intervals)
            temp_pick_intervals[ev_id].append((pick_start, pick_end))

            old_pres_end, old_ret_end = (
                temp_bounds[ev_id]
                if ev_id in temp_bounds
                else (shared.presence_end, shared.return_end)
            )
            temp_bounds[ev_id] = (new_ret_start, new_ret_end)

            if new_ret_start > old_pres_end:
                for idx_iv, (iv_s, iv_e) in enumerate(local_intervals):
                    if iv_s == shared.presence_start and iv_e == old_pres_end:
                        local_intervals[idx_iv] = (shared.presence_start, new_ret_start)
                        break
                local_intervals.sort(key=lambda x: x[0])

                if (old_pres_end, old_ret_end) in pending_moves:
                    pending_moves.remove((old_pres_end, old_ret_end))
                elif shared in state.station_bin_events[s]:
                    orig_ret = (shared.return_start, shared.return_end)
                    if orig_ret not in removed_moves:
                        removed_moves.append(orig_ret)
                pending_moves.append((new_ret_start, new_ret_end))

            t_cursor = pick_end
            pick_times_dict[k] = (pick_start, pick_end)
            shared_picks.append(
                (k, shared, pick_start, pick_end, new_ret_start, new_ret_end)
            )
            continue

        # 2. No shared bin available: pick highest-demand remaining SKU and fetch a new bin
        k = remaining_skus.pop(0)
        desired_presence = max(t_cursor, rt[k])

        while True:
            gap_start = find_earliest_gap(local_intervals, desired_presence, p[k])
            desired_fetch = gap_start - rt[k]

            try:
                t_fetch, copy_id = earliest_feasible_fetch(
                    k,
                    s,
                    desired_fetch,
                    state,
                    rt,
                    rt_ret,
                    pending_moves,
                    pending_copies,
                    horizon,
                    removed_moves=removed_moves,
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
                    return_start,
                    rt_ret[k],
                    state.move_cap,
                    pending_moves,
                    removed_moves=removed_moves,
                )
                if return_delayed != return_start:
                    return_start = return_delayed
                    return_end = return_start + rt_ret[k]
                    presence_end = return_start
                    if return_end > horizon:
                        return None
                    if gap_start != find_earliest_gap(
                        local_intervals, presence_start, presence_end - presence_start
                    ):
                        desired_presence = presence_start + 1
                        continue

            break

        t_cursor = pick_end
        pick_times_dict[k] = (pick_start, pick_end)
        local_intervals.append((presence_start, presence_end))
        local_intervals.sort(key=lambda x: x[0])

        ev = BinEvent(
            sku=k,
            copy_id=copy_id,
            fetch_start=t_fetch,
            fetch_end=fetch_end,
            presence_start=presence_start,
            presence_end=presence_end,
            return_start=return_start,
            return_end=return_end,
            orders_served=[o],
            pick_intervals=[(pick_start, pick_end)],
        )
        new_bin_events.append(ev)
        pending_moves.append((t_fetch, fetch_end))
        pending_moves.append((return_start, return_end))
        pending_copies[k].append((copy_id, return_end))

    order_start = min(
        (ps for ps, pe in pick_times_dict.values()), default=t_order_start
    )
    order_end = max((pe for ps, pe in pick_times_dict.values()), default=t_order_start)

    # Reward shared picks by subtracting 50% of the saved retrieval + pick time
    move_savings = sum(rt[k_sh] + p[k_sh] for (k_sh, *_rest) in shared_picks)
    score = ALPHA * order_end + BETA * (order_end - order_start) - 0.5 * move_savings

    return OrderPlan(
        order=o,
        station=s,
        lane=best_lane,
        start=order_start,
        end=order_end,
        bin_events=new_bin_events,
        shared_picks=shared_picks,
        score=score,
        pick_times=pick_times_dict,
    )


def score_order_insertions(
    o: int,
    S: list[int],
    state: HeuristicState,
    orders_req: dict[int, list[int]],
    rt: dict[int, int],
    rt_ret: dict[int, int],
    p: dict[int, int],
    N: dict[int, int],
    horizon: int,
    ALPHA: float,
    BETA: float,
    demand_count: dict[int, int],
    regret_k: int,
    pick_cap: int,
) -> InsertionScores:
    station_scores = {}
    station_plans = {}

    for s in S:
        plan = gap_filling_plan_order_at_station(
            o,
            s,
            state,
            orders_req,
            rt,
            rt_ret,
            p,
            N,
            horizon,
            ALPHA,
            BETA,
            demand_count,
            pick_cap,
        )
        if plan is not None:
            station_scores[s] = plan.score
            station_plans[s] = plan
        else:
            station_scores[s] = float("inf")
            station_plans[s] = None

    feasible = sorted(
        (score, s) for s, score in station_scores.items() if score < float("inf")
    )

    if not feasible:
        return InsertionScores(
            o, station_scores, station_plans, float("inf"), float("inf"), -1, None, -1.0
        )

    best_score, best_station = feasible[0]
    second_score = feasible[1][0] if len(feasible) > 1 else float("inf")
    best_plan = station_plans[best_station]
    regret = compute_regret_k(station_scores, regret_k)

    return InsertionScores(
        o,
        station_scores,
        station_plans,
        best_score,
        second_score,
        best_station,
        best_plan,
        regret,
    )


def compute_dirty_set(
    committed_order: int,
    committed_station: int,
    orders_req: dict[int, list[int]],
    sku_order_index: dict[int, set[int]],
    cached_scores: dict[int, InsertionScores],
    unscheduled: set[int],
    state: HeuristicState | None = None,
    committed_plan: OrderPlan | None = None,
) -> set[int]:
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


def update_cached_scores_incrementally(
    committed_order: int,
    committed_plan: OrderPlan,
    S: list[int],
    state: HeuristicState,
    orders_req: dict[int, list[int]],
    sku_order_index: dict[int, set[int]],
    cached_scores: dict[int, InsertionScores],
    unscheduled: set[int],
    rt: dict[int, int],
    rt_ret: dict[int, int],
    p: dict[int, int],
    N: dict[int, int],
    horizon: int,
    ALPHA: float,
    BETA: float,
    demand_count: dict[int, int],
    regret_k: int,
    pick_cap: int,
    use_lazy: bool,
) -> None:
    committed_station = committed_plan.station

    # Orders that share an SKU with committed_order may be affected on ALL stations (due to BinCopyPool)
    all_stations_dirty: set[int] = set()
    if use_lazy:
        for k in orders_req[committed_order]:
            if k in sku_order_index:
                all_stations_dirty.update(sku_order_index[k])
    else:
        all_stations_dirty = set(unscheduled)

    # Collect move intervals committed or extended by committed_plan (for MoveCap overlap checks)
    committed_moves: list[tuple[int, int]] = []
    if state.move_cap is not None:
        for ev in committed_plan.bin_events:
            committed_moves.append((ev.fetch_start, ev.fetch_end))
            committed_moves.append((ev.return_start, ev.return_end))
        for _k, sh_ev, _ps, _pe, new_rs, new_re in committed_plan.shared_picks:
            committed_moves.append((sh_ev.return_start, sh_ev.return_end))
            committed_moves.append((new_rs, new_re))

    for o in unscheduled:
        c = cached_scores[o]
        if not use_lazy or o in all_stations_dirty:
            stations_to_update = S
        else:
            # Always update committed_station for every unscheduled order!
            # This guarantees best_score, second_score, and regret are never stale,
            # and lets orders discover newly created shared bins at committed_station.
            stations_to_update = [committed_station]
            if state.move_cap is not None and committed_moves:
                for s in S:
                    if s == committed_station:
                        continue
                    st_plan = c.station_plans.get(s)
                    if st_plan is None:
                        continue
                    overlap = False
                    for ev in st_plan.bin_events:
                        for ms, me in committed_moves:
                            if max(ev.fetch_start, ms) < min(ev.fetch_end, me) or max(
                                ev.return_start, ms
                            ) < min(ev.return_end, me):
                                overlap = True
                                break
                        if overlap:
                            break
                    if overlap:
                        stations_to_update.append(s)

        for s in stations_to_update:
            plan = gap_filling_plan_order_at_station(
                o,
                s,
                state,
                orders_req,
                rt,
                rt_ret,
                p,
                N,
                horizon,
                ALPHA,
                BETA,
                demand_count,
                pick_cap,
            )
            if plan is not None:
                c.station_scores[s] = plan.score
                c.station_plans[s] = plan
            else:
                c.station_scores[s] = float("inf")
                c.station_plans[s] = None

        feasible = sorted(
            (score, s) for s, score in c.station_scores.items() if score < float("inf")
        )
        if not feasible:
            c.best_score = float("inf")
            c.second_score = float("inf")
            c.best_station = -1
            c.best_plan = None
            c.regret = -1.0
        else:
            c.best_score, c.best_station = feasible[0]
            c.second_score = feasible[1][0] if len(feasible) > 1 else float("inf")
            c.best_plan = c.station_plans[c.best_station]
            c.regret = compute_regret_k(c.station_scores, regret_k)


def run_rdi_sgc(instance, *args, **kwargs) -> Solution:

    horizon = kwargs.get("horizon", args[0] if len(args) > 0 else 10000)
    ALPHA = kwargs.get("ALPHA", args[2] if len(args) > 2 else 1.0)
    BETA = kwargs.get("BETA", args[3] if len(args) > 3 else 0.0)
    regret_k = kwargs.get("regret_k", args[4] if len(args) > 4 else 2)
    use_lazy = kwargs.get("use_lazy", args[5] if len(args) > 5 else True)
    tiebreaker = kwargs.get(
        "tiebreaker", args[6] if len(args) > 6 else "sharing_degree"
    )

    S, L, K, orders_req, rt, p, N = instance
    O = instance.O
    rt_ret = instance.rt_ret
    move_cap = instance.movecap
    pick_cap = instance.pickcap
    exo_lanes = kwargs.get("exo_lanes")
    exo_blocks = kwargs.get("exo_blocks")
    exo_moves = kwargs.get("exo_moves")
    batch_start_time = kwargs.get("batch_start_time", 0)

    state = init_state(
        S,
        L,
        K,
        N,
        horizon,
        move_cap,
        exo_lanes,
        exo_blocks,
        exo_moves,
        batch_start_time,
    )

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
    cached_scores: dict[int, InsertionScores] = {}

    for o in O:
        cached_scores[o] = score_order_insertions(
            o,
            S,
            state,
            orders_req,
            rt,
            rt_ret,
            p,
            N,
            horizon,
            ALPHA,
            BETA,
            demand_count,
            regret_k,
            pick_cap,
        )

    order_assignments = {}
    pick_events_map = {}

    while unscheduled:
        best_o = None
        best_key = None

        for o in unscheduled:
            c = cached_scores[o]
            if c.best_plan is None:
                continue

            num_shared = len(c.best_plan.shared_picks)
            # Combine regret with a best_score penalty (-0.35 * c.best_score) so RDI
            # prioritizes orders that both have high regret AND land early/share bins,
            # preventing runaway station imbalance (PI).
            priority_val = (
                c.regret - 0.35 * c.best_score
                if c.regret < float("inf")
                else float("inf")
            )

            if tiebreaker == "sharing_degree":
                tb_val = sharing_degrees[o]
            elif tiebreaker == "sum_rt_asc":
                tb_val = -sum_rts[o]
            else:
                tb_val = -c.best_score

            cand_key = (
                priority_val,
                num_shared,
                sharing_degrees[o],
                tb_val,
                -c.best_score,
            )
            if best_key is None or cand_key > best_key:
                best_key = cand_key
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

        update_cached_scores_incrementally(
            best_o,
            plan,
            S,
            state,
            orders_req,
            sku_index,
            cached_scores,
            unscheduled,
            rt,
            rt_ret,
            p,
            N,
            horizon,
            ALPHA,
            BETA,
            demand_count,
            regret_k,
            pick_cap,
            use_lazy,
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
        pick_events=pick_events_map,
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

    instance = generate_data(
        num_stations=num_stations,
        lanes_per_station=lanes_per_station,
        num_orders=num_orders,
        num_skus=num_skus,
        seed=seed,
        pick_touch_time=pick_touch_time,
        movecap=move_cap,
    )
    S, L, K, orders_req, rt, p, N = instance
    O = instance.O

    t0 = time.perf_counter()
    sol = run_rdi_sgc(
        instance,
        horizon=horizon,
        move_cap=move_cap,
        ALPHA=alpha,
        BETA=beta,
        regret_k=config.get("regret_k", 2),
        use_lazy=config.get("use_lazy", True),
        tiebreaker=config.get("tiebreaker", "sum_rt_asc"),
    )
    elapsed = time.perf_counter() - t0
    status = "Feasible" if sol.feasible else "Infeasible"
    violations = validate_solution(sol, instance, horizon=horizon, move_cap=move_cap)
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
    ap.add_argument(
        "--tiebreaker",
        type=str,
        choices=["sharing_degree", "best_score", "sum_rt_asc"],
        default="sum_rt_asc",
    )
    ap.add_argument("--no_vis", action="store_true")
    args = ap.parse_args()

    print("Generating data...")
    instance = generate_data(
        num_stations=args.stations,
        lanes_per_station=args.lanes,
        num_orders=args.orders,
        num_skus=args.skus,
        seed=args.seed,
        pick_touch_time=args.pick,
        movecap=args.movecap,
    )
    S, L, K, orders_req, rt, p, N = instance
    O = instance.O

    print(
        f"Stations={len(S)}, Lanes={len(L)}, SKUs={len(K)}, Orders={len(O)}, RobotLimit={args.movecap}\n"
    )
    print(
        f"Running RDI-SGC (regret_k={args.regret_k}, tiebreaker={args.tiebreaker}, lazy={not args.no_lazy})"
    )

    t0 = time.perf_counter()
    sol = run_rdi_sgc(
        instance,
        horizon=args.horizon,
        move_cap=args.movecap,
        ALPHA=args.alpha,
        BETA=args.beta,
        regret_k=args.regret_k,
        use_lazy=not args.no_lazy,
        tiebreaker=args.tiebreaker,
    )
    elapsed = time.perf_counter() - t0

    print("\n=== RDI-SGC Result ===")
    print(f"Feasible:    {sol.feasible}")
    print(f"Makespan:    {sol.makespan}")
    print(f"Total bin events (moves/2): {sol.total_moves // 2}")
    print(f"Time:        {elapsed:.4f}s")

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
            from autostore_heuristic import build_viz_handles
            from schedule_visualizer import plot_schedule

            mock_sol, handles = build_viz_handles(sol, instance)
            plot_schedule(mock_sol, handles)
        except Exception as exc:
            import traceback

            traceback.print_exc()
            print(f"[VIS] Skipped: {exc}")


if __name__ == "__main__":
    main()
