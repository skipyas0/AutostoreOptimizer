"""Global Bin Scheduling (GBS) — bin-centred construction.

1. Orders are partitioned over stations greedily (by sum rt desc), scoring
   each station by ``rt overlap with its SKUs - its rt load``.
2. Per station, lanes are seeded with the order of smallest sum rt plus the
   orders with the largest rt+rt_ret overlap with the active ones.
3. Repeatedly, every SKU still needed by an active order becomes a *bin task*;
   tasks are scored by ``rule`` and the best one is scheduled as ONE bin visit
   serving all active orders that need the SKU. Completed orders free their
   lane for the unscheduled order with the largest overlap.

Rules: "max_sharing" (n_sharing * (rt+rt_ret)), "critical_path" (largest
remaining work of a served order), "readiness_weighted" (max_sharing minus
pickface idle gap). "adaptive" runs the three and keeps the best feasible by
``alpha*C_max + beta*flow_time`` (ties: fewer moves).

Bin copy pools and the MoveCap profile are shared across stations.
Known issues (kept for exactness): pickcap/lanes/alpha/beta are ignored when
building the schedule; see docs/KNOWN_ISSUES.md.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

from heuristics.heuristic_utils import BinCopyPool, BinEvent, DifferenceArray
from heuristics.solution import HeuristicSolution
from instance import Instance

GBS_RULES = ("max_sharing", "critical_path", "readiness_weighted")


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

    def clone(self) -> "GBSState":
        return copy.deepcopy(self)


@dataclass
class BinTask:
    sku: int
    concurrent_orders: list[int]
    n_sharing: int
    sharing_savings: float
    earliest_fetch_start: int
    copy_id: int
    score: float


def _overlap_score(inst: Instance, cand: int, skus: set[int]) -> int:
    overlap = set(inst.orders_req[cand]).intersection(skus)
    return sum(inst.rt[k] + inst.rt_ret[k] for k in overlap)


def seed_initial_lanes(inst: Instance, L: list[int], O: list[int]) -> tuple[dict[int, int], list[int]]:
    """Initial order per lane. Returns ({order: lane}, remaining orders sorted by sum rt)."""
    rt, orders_req = inst.rt, inst.orders_req

    def sum_rt(o: int) -> int:
        return sum(rt[k] for k in orders_req[o])

    unscheduled = sorted(O, key=sum_rt)
    if not unscheduled:
        return {}, []

    anchor = unscheduled.pop(0)
    assignment = {anchor: L[0]}
    active_set = {anchor}

    for lane_idx in range(1, min(len(L), len(O))):
        active_skus: set[int] = set()
        for ao in active_set:
            active_skus.update(orders_req[ao])
        best_order, best_score, best_tb = None, -1.0, float("inf")
        for cand in unscheduled:
            score = _overlap_score(inst, cand, active_skus)
            tb = sum_rt(cand)
            if score > best_score or (score == best_score and tb < best_tb):
                best_score, best_tb, best_order = score, tb, cand
        if best_order is not None:
            assignment[best_order] = L[lane_idx]
            active_set.add(best_order)
            unscheduled.remove(best_order)
    return assignment, unscheduled


def admit_next_order(inst: Instance, unscheduled: list[int], gs: GBSState) -> Optional[int]:
    if not unscheduled:
        return None
    remaining_skus: set[int] = set()
    for o in gs.active_orders:
        remaining_skus.update(gs.pending_picks[o])
    best_order, best_score, best_tb = None, -1.0, float("inf")
    for cand in unscheduled:
        score = _overlap_score(inst, cand, remaining_skus)
        tb = sum(inst.rt[k] for k in inst.orders_req[cand])
        if score > best_score or (score == best_score and tb < best_tb):
            best_score, best_tb, best_order = score, tb, cand
    return best_order


def score_bin_task(task: BinTask, gs: GBSState, inst: Instance, rule: str) -> float:
    rt, rt_ret, p = inst.rt, inst.rt_ret, inst.p
    k = task.sku
    if rule == "max_sharing":
        return float(task.n_sharing * (rt[k] + rt_ret[k]))
    if rule == "critical_path":
        max_rem = 0
        for o in task.concurrent_orders:
            rem = sum(rt[k2] + p[k2] + rt_ret[k2] for k2 in gs.pending_picks[o])
            max_rem = max(max_rem, rem)
        return float(max_rem)
    if rule == "readiness_weighted":
        idle_gap = max(0, task.earliest_fetch_start + rt[k] - gs.pickface_free)
        return float(task.n_sharing * (rt[k] + rt_ret[k]) - 1.0 * idle_gap)
    return 0.0


def build_bin_tasks(gs: GBSState, inst: Instance, rule: str) -> list[BinTask]:
    rt, rt_ret, p = inst.rt, inst.rt_ret, inst.p
    needed: set[int] = set()
    for o in gs.active_orders:
        needed.update(gs.pending_picks[o])

    tasks = []
    for k in needed:
        concurrent = [o for o in gs.active_orders if k in gs.pending_picks[o]]
        n_sharing = len(concurrent)
        copy_free, copy_id = gs.bin_pools[k].get_earliest_free()
        task = BinTask(
            sku=k,
            concurrent_orders=concurrent,
            n_sharing=n_sharing,
            sharing_savings=float((n_sharing - 1) * (rt[k] + p[k] + rt_ret[k])),
            earliest_fetch_start=max(gs.pickface_free - rt[k], copy_free),
            copy_id=copy_id,
            score=0.0,
        )
        task.score = score_bin_task(task, gs, inst, rule)
        tasks.append(task)
    tasks.sort(key=lambda t: t.score, reverse=True)
    return tasks


def schedule_bin_event(task: BinTask, gs: GBSState, inst: Instance, horizon: int) -> Optional[BinEvent]:
    k = task.sku
    rt, rt_ret, p = inst.rt[k], inst.rt_ret[k], inst.p[k]

    t_fetch = task.earliest_fetch_start
    if gs.move_cap is not None:
        t_fetch = gs.move_da.delay_for_movecap(t_fetch, rt, gs.move_cap)
    t_fetch_end = t_fetch + rt
    t_presence_start = t_fetch_end
    t_presence_end = t_presence_start + p
    t_return_start = t_presence_end
    if gs.move_cap is not None:
        t_return_start = gs.move_da.delay_for_movecap(t_return_start, rt_ret, gs.move_cap)
        t_presence_end = t_return_start
    t_return_end = t_return_start + rt_ret
    if t_return_end > horizon:
        return None

    if gs.move_cap is not None:
        gs.move_da.add_move(t_fetch, t_fetch_end)
        gs.move_da.add_move(t_return_start, t_return_end)

    ev = BinEvent(
        sku=k,
        copy_id=task.copy_id,
        fetch_start=t_fetch,
        fetch_end=t_fetch_end,
        presence_start=t_presence_start,
        presence_end=t_presence_end,
        return_start=t_return_start,
        return_end=t_return_end,
        orders_served=list(task.concurrent_orders),
    )
    gs.bin_pools[k].commit(task.copy_id, t_return_end)
    gs.pickface_free = t_presence_end
    gs.bin_events.append(ev)

    for o in task.concurrent_orders:
        # NB: every served order "picks" during the whole presence window simultaneously
        gs.order_pick_times[(o, k)] = (t_presence_start, t_presence_end)
        gs.pending_picks[o].remove(k)
        old = gs.order_windows.get(o)
        gs.order_windows[o] = (
            (t_presence_start, t_presence_end)
            if old is None
            else (min(old[0], t_presence_start), max(old[1], t_presence_end))
        )
    return ev


def check_completions(gs: GBSState, unscheduled: list[int], inst: Instance) -> list[int]:
    newly_admitted = []
    completed = [o for o in gs.active_orders if not gs.pending_picks[o]]
    for o in completed:
        gs.active_orders.remove(o)
        freed = None
        for ls in gs.lane_slots:
            if ls.current_order == o:
                ls.current_order = None
                ls.free_at = gs.order_windows[o][1]
                freed = ls
                break
        if freed and unscheduled:
            new_o = admit_next_order(inst, unscheduled, gs)
            if new_o is not None:
                unscheduled.remove(new_o)
                gs.active_orders.add(new_o)
                gs.pending_picks[new_o] = set(inst.orders_req[new_o])
                freed.current_order = new_o
                gs.order_admission_times[new_o] = freed.free_at
                gs.order_lanes[new_o] = freed.lane
                newly_admitted.append(new_o)
    return newly_admitted


def _run_gbs_rule(inst: Instance, horizon: int, rule: str) -> HeuristicSolution:
    S, L, K, N, O = inst.S, inst.L, inst.K, inst.N, inst.O
    orders_req, rt = inst.orders_req, inst.rt

    # 1. greedy order -> station partition (overlap minus load)
    station_orders = {s: [] for s in S}
    station_skus = {s: set() for s in S}
    station_rt_load = {s: 0.0 for s in S}
    for o in sorted(O, key=lambda o: sum(rt[k] for k in orders_req[o]), reverse=True):
        o_skus = set(orders_req[o])
        o_rt = sum(rt[k] for k in o_skus)
        best_s, best_score = S[0], -float("inf")
        for s in S:
            score = sum(rt[k] for k in station_skus[s].intersection(o_skus)) - station_rt_load[s]
            if score > best_score:
                best_score, best_s = score, s
        station_orders[best_s].append(o)
        station_skus[best_s].update(o_skus)
        station_rt_load[best_s] += o_rt

    # 2. resources shared across stations
    move_da = DifferenceArray(horizon)
    bin_pools = {k: BinCopyPool(k, N[k]) for k in K}
    plans: dict[int, dict] = {}
    all_bin_events: dict[int, list[BinEvent]] = {}

    # 3. schedule stations one after another
    for station in S:
        O_s = station_orders[station]
        if not O_s:
            all_bin_events[station] = []
            continue
        assignment, unscheduled = seed_initial_lanes(inst, L, O_s)
        lane_slots = [LaneSlot(ln, station, None, 0) for ln in L]
        for o, ln in assignment.items():
            for ls in lane_slots:
                if ls.lane == ln:
                    ls.current_order = o
                    break
        gs = GBSState(
            lane_slots=lane_slots,
            active_orders=set(assignment.keys()),
            pending_picks={o: set(orders_req[o]) for o in assignment},
            pickface_free=0,
            bin_events=[],
            order_pick_times={},
            order_windows={},
            move_da=move_da,
            bin_pools=bin_pools,
            move_cap=inst.movecap,
            order_admission_times={o: 0 for o in assignment},
            order_lanes=dict(assignment),
        )
        while gs.active_orders or unscheduled:
            tasks = build_bin_tasks(gs, inst, rule)
            if not tasks:
                break
            ev = None
            for task in tasks:
                ev = schedule_bin_event(task, gs, inst, horizon)
                if ev is not None:
                    break
            if ev is None:
                break
            check_completions(gs, unscheduled, inst)

        for o in O_s:
            win = gs.order_windows.get(o, (0, 0))
            plans[o] = {
                "station": station,
                "lane": gs.order_lanes.get(o, 0),
                "start": win[0],
                "end": win[1],
                "picks": {k: gs.order_pick_times.get((o, k), (0, 0)) for k in orders_req[o]},
            }
        all_bin_events[station] = gs.bin_events

    feasible = all(o in plans and len(plans[o]["picks"]) >= len(orders_req[o]) for o in O)
    return HeuristicSolution(
        feasible=feasible,
        makespan=max(op["end"] for op in plans.values()) if plans else 0,
        total_moves=sum(len(b) for b in all_bin_events.values()) * 2,
        order_assignments={o: (op["station"], op["lane"], op["start"], op["end"]) for o, op in plans.items()},
        pick_events={(o, op["station"], k): op["picks"].get(k, (0, 0)) for o, op in plans.items() for k in orders_req[o]},
        bin_events=all_bin_events,
        meta={"heuristic": "gbs", "rule": rule},
    )


def run_gbs(
    inst: Instance,
    *,
    horizon: int = 10000,
    rule: str = "max_sharing",
    alpha: float = 1.0,
    beta: float = 0.0,
) -> HeuristicSolution | None:
    """GBS with one scoring rule, or rule="adaptive" (best of all rules; None if none feasible).

    alpha/beta only matter for "adaptive" model selection.
    """
    if rule != "adaptive":
        return _run_gbs_rule(inst, horizon, rule)

    best_sol, best_obj, best_moves = None, float("inf"), float("inf")
    for r in GBS_RULES:
        sol = _run_gbs_rule(inst, horizon, r)
        if not sol.feasible:
            continue
        flow_time = sum(sol.order_assignments[o][3] - sol.order_assignments[o][2] for o in inst.O)
        obj = alpha * sol.makespan + beta * flow_time
        if obj < best_obj or (obj == best_obj and sol.total_moves < best_moves):
            best_sol, best_obj, best_moves = sol, obj, sol.total_moves
    if best_sol is not None:
        best_sol.meta["rule"] = f"adaptive:{best_sol.meta['rule']}"
    return best_sol
