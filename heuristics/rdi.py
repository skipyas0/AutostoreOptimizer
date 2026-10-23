"""Regret-based Dynamic Insertion (RDI-SGC).

Unlike SGC (static order sequence), RDI keeps, for every unscheduled order, a
cached insertion plan at every station and repeatedly commits the order with
the highest priority

    key = (regret_k - 0.35 * best_score, #shared picks, sharing_degree, tiebreak, -best_score)

where regret_k = sum of the (k-1) score gaps to the best station. After each
commit only affected (order, station) pairs are re-planned (``use_lazy``):
orders sharing an SKU with the committed order are re-planned everywhere,
all others only at the committed station plus stations whose cached plan
overlaps the committed robot moves (MoveCap).

The per-station planner (``plan_order_gap_filling``) differs from SGC's:
  * it picks *dynamically* the next SKU that can be served from a shared bin
    earliest, before falling back to fetching the highest-demand SKU;
  * new bins are placed into the earliest pickface gap (gap filling) instead
    of after the last presence;
  * its score rewards shared picks: ``alpha*end + beta*(end-start) - 0.5*sum(rt+p of shared SKUs)``.
"""

from __future__ import annotations

from dataclasses import dataclass

from heuristics.heuristic_utils import (
    BinEvent,
    ExoState,
    HeuristicState,
    OrderPlan,
    PlanScratch,
    commit_plan,
    earliest_feasible_fetch,
    earliest_lane,
    find_shared_bin,
    init_state,
    record_plan,
)
from heuristics.solution import HeuristicSolution
from instance import Instance

TIEBREAKERS = ("sharing_degree", "sum_rt_asc", "best_score")


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

    def refresh(self, regret_k: int) -> None:
        """Recompute best/second/regret from station_scores."""
        feasible = sorted((score, s) for s, score in self.station_scores.items() if score < float("inf"))
        if not feasible:
            self.best_score = float("inf")
            self.second_score = float("inf")
            self.best_station = -1
            self.best_plan = None
            self.regret = -1.0
        else:
            self.best_score, self.best_station = feasible[0]
            self.second_score = feasible[1][0] if len(feasible) > 1 else float("inf")
            self.best_plan = self.station_plans[self.best_station]
            self.regret = compute_regret_k(self.station_scores, regret_k)


def compute_regret_k(station_scores: dict[int, float], k: int) -> float:
    feasible = sorted(s for s in station_scores.values() if s < float("inf"))
    if len(feasible) <= 1:
        return float("inf")
    best = feasible[0]
    return sum(feasible[i] - best for i in range(1, min(k, len(feasible))))


def find_earliest_gap(intervals: list[tuple[int, int]], ready_time: int, duration: int) -> int:
    """Earliest t >= ready_time with [t, t+duration) free of the (sorted) intervals."""
    current_time = ready_time
    for start, end in intervals:
        if current_time + duration <= start:
            return current_time
        current_time = max(current_time, end)
    return current_time


# ======================================================================
# Gap-filling planner
# ======================================================================


def plan_order_gap_filling(o: int, s: int, state: HeuristicState, alpha: float, beta: float) -> OrderPlan | None:
    inst = state.inst
    rt, rt_ret, p = inst.rt, inst.rt_ret, inst.p
    lane = earliest_lane(state, s)
    if lane is None:
        return None
    best_lane, t_order_start = lane

    demand_count = inst.demand_count
    remaining_skus = sorted(inst.orders_req[o], key=lambda k: demand_count.get(k, 0), reverse=True)
    t_cursor = t_order_start
    sc = PlanScratch.for_station(state, s)
    candidate_bins = state.station_bin_events[s]

    while remaining_skus:
        # 1. Shared-first: the remaining SKU that can be picked earliest from an existing bin
        best_shared_k = None
        best_shared_res = (None, float("inf"), None, None)
        for k_cand in remaining_skus:
            res = find_shared_bin(candidate_bins + sc.new_bin_events, k_cand, t_cursor, state, s, sc)
            if res[0] is not None and res[1] < best_shared_res[1]:
                best_shared_k, best_shared_res = k_cand, res

        if best_shared_k is not None:
            remaining_skus.remove(best_shared_k)
            shared, pick_start, new_ret_start, new_ret_end = best_shared_res
            t_cursor, _ = sc.apply_shared_pick(state, s, best_shared_k, shared, pick_start, new_ret_start, new_ret_end)
            continue

        # 2. Otherwise fetch the highest-demand remaining SKU into the earliest pickface gap
        k = remaining_skus.pop(0)
        # NOTE (time-based decomposition): rt[k] is an *absolute* lower bound here (a fetch cannot
        # start before t=0). With ExoState.batch_start_time > 0 the meaningful bound would be
        # batch_start_time + rt[k]. Harmless today because t_cursor >= batch_start_time and
        # copy availability starts at batch_start_time, but revisit when tuning time decomposition.
        desired_presence = max(t_cursor, rt[k])
        while True:
            gap_start = find_earliest_gap(sc.local_intervals, desired_presence, p[k])
            desired_fetch = gap_start - rt[k]
            try:
                t_fetch, copy_id = earliest_feasible_fetch(k, desired_fetch, state, sc)
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
            if return_end > state.horizon:
                return None

            return_delayed = sc.delay_move(state, return_start, rt_ret[k])
            if return_delayed != return_start:
                return_start = return_delayed
                return_end = return_start + rt_ret[k]
                presence_end = return_start
                if return_end > state.horizon:
                    return None
                if gap_start != find_earliest_gap(sc.local_intervals, presence_start, presence_end - presence_start):
                    desired_presence = presence_start + 1
                    continue
            break

        t_cursor = pick_end
        sc.add_new_bin(
            BinEvent(
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
        )

    start, end = sc.window(t_order_start)
    # Reward shared picks by 50% of the saved retrieval + pick time
    move_savings = sum(rt[k_sh] + p[k_sh] for (k_sh, *_rest) in sc.shared_picks)
    return sc.to_plan(o, s, best_lane, start, end, alpha * end + beta * (end - start) - 0.5 * move_savings)


# ======================================================================
# Insertion-score cache
# ======================================================================


def score_order_insertions(
    o: int, state: HeuristicState, alpha: float, beta: float, regret_k: int
) -> InsertionScores:
    station_scores, station_plans = {}, {}
    for s in state.inst.S:
        plan = plan_order_gap_filling(o, s, state, alpha, beta)
        station_scores[s] = plan.score if plan is not None else float("inf")
        station_plans[s] = plan
    c = InsertionScores(o, station_scores, station_plans, float("inf"), float("inf"), -1, None, -1.0)
    c.refresh(regret_k)
    return c


def _overlaps_moves(plan: OrderPlan, moves: list[tuple[int, int]]) -> bool:
    for ev in plan.bin_events:
        for ms, me in moves:
            if max(ev.fetch_start, ms) < min(ev.fetch_end, me) or max(ev.return_start, ms) < min(ev.return_end, me):
                return True
    return False


def update_cached_scores(
    committed: OrderPlan,
    state: HeuristicState,
    cached: dict[int, InsertionScores],
    unscheduled: dict[int, None],
    alpha: float,
    beta: float,
    regret_k: int,
    use_lazy: bool,
) -> None:
    inst = state.inst
    S = inst.S

    # Orders sharing an SKU with the committed order may change at ALL stations (BinCopyPool)
    if use_lazy:
        all_stations_dirty: set[int] = set()
        for k in inst.orders_req[committed.order]:
            if k in inst.sku_order_index:
                all_stations_dirty.update(inst.sku_order_index[k])
    else:
        all_stations_dirty = set(unscheduled)

    # Robot moves committed or moved by the plan (for MoveCap overlap checks)
    committed_moves: list[tuple[int, int]] = []
    for ev in committed.bin_events:
        committed_moves.append((ev.fetch_start, ev.fetch_end))
        committed_moves.append((ev.return_start, ev.return_end))
    for _k, sh_ev, _ps, _pe, new_rs, new_re in committed.shared_picks:
        committed_moves.append((sh_ev.return_start, sh_ev.return_end))
        committed_moves.append((new_rs, new_re))

    for o in unscheduled:
        c = cached[o]
        if not use_lazy or o in all_stations_dirty:
            stations = S
        else:
            # Always refresh the committed station (keeps best/second/regret exact and
            # lets the order discover new shared bins there) + stations whose cached
            # plan collides with the committed moves.
            stations = [committed.station]
            if committed_moves:
                for s in S:
                    if s == committed.station:
                        continue
                    st_plan = c.station_plans.get(s)
                    if st_plan is not None and _overlaps_moves(st_plan, committed_moves):
                        stations.append(s)

        for s in stations:
            plan = plan_order_gap_filling(o, s, state, alpha, beta)
            c.station_scores[s] = plan.score if plan is not None else float("inf")
            c.station_plans[s] = plan
        c.refresh(regret_k)


# ======================================================================
# Main loop
# ======================================================================


def run_rdi(
    inst: Instance,
    *,
    horizon: int = 10000,
    alpha: float = 1.0,
    beta: float = 0.0,
    regret_k: int = 2,
    use_lazy: bool = True,
    tiebreaker: str = "sharing_degree",
    exo: ExoState | None = None,
) -> HeuristicSolution:
    """Regret-based Dynamic Insertion SGC. ``tiebreaker`` in TIEBREAKERS."""
    O = inst.O
    state = init_state(inst, horizon, exo)
    sharing_degrees = inst.sharing_degree
    sum_rts = {o: sum(inst.rt[k] for k in inst.req_sets[o]) for o in O}

    # Insertion-ordered dict used as an ordered set: iteration (and hence exact-tie breaking,
    # first candidate wins) follows order id, independent of hash-table layout.
    unscheduled = dict.fromkeys(O)
    cached = {o: score_order_insertions(o, state, alpha, beta, regret_k) for o in O}

    order_assignments: dict[int, tuple[int, int, int, int]] = {}
    pick_events: dict[tuple[int, int, int], tuple[int, int]] = {}

    while unscheduled:
        best_o, best_key = None, None
        for o in unscheduled:  # in order-id order; the first of exactly tied candidates wins
            c = cached[o]
            if c.best_plan is None:
                continue
            # regret plus a best-score penalty: prefer high-regret orders that land early
            priority = c.regret - 0.35 * c.best_score if c.regret < float("inf") else float("inf")
            if tiebreaker == "sharing_degree":
                tb = sharing_degrees[o]
            elif tiebreaker == "sum_rt_asc":
                tb = -sum_rts[o]
            else:
                tb = -c.best_score
            key = (priority, len(c.best_plan.shared_picks), sharing_degrees[o], tb, -c.best_score)
            if best_key is None or key > best_key:
                best_key, best_o = key, o

        if best_o is None or cached[best_o].best_plan is None:
            break  # remaining orders infeasible

        plan = cached[best_o].best_plan
        commit_plan(plan, state)
        record_plan(plan, order_assignments, pick_events)
        del unscheduled[best_o]
        if not unscheduled:
            break
        update_cached_scores(plan, state, cached, unscheduled, alpha, beta, regret_k, use_lazy)

    makespan = max((e for (_, _, _, e) in order_assignments.values()), default=0)
    return HeuristicSolution(
        order_assignments=order_assignments,
        bin_events=state.station_bin_events,
        makespan=makespan,
        total_moves=sum(len(state.station_bin_events[s]) * 2 for s in inst.S),
        feasible=(len(unscheduled) == 0) and (makespan <= horizon),
        pick_events=pick_events,
        meta={"heuristic": "rdi", "tiebreaker": tiebreaker, "regret_k": regret_k, "use_lazy": use_lazy},
    )
