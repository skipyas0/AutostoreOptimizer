"""GBS: Global Bin Scheduling, a bin-centred constructive heuristic.

SGC and RDI are order-centred: they place one order at a time and let it borrow a bin that
happens to be at the pickface. GBS turns this around and schedules *bin visits*:

1. **Partition** orders over stations (``partition``):
   - ``"overlap"`` (original): largest ``sum_rt`` first, each order to the station maximising
     ``rt-overlap with the station's SKUs - station rt load`` (first station on ties).
   - ``"lpt"``: Longest Processing Time on picking workload. Largest ``Σ p`` first, each order to
     the station with the smallest ``Σ p`` so far; ties -> most SKUs shared with the station's
     orders, then the first station.
2. **Seed** the lanes of every station: the order with the smallest ``sum_rt`` takes the first lane,
   then each next lane takes the unscheduled order with the largest ``Σ (rt + rt_ret)`` overlap with
   the SKUs of the orders already seeded (ties: smaller ``sum_rt``).
3. **Step** at one station: one *bin task* per SKU still needed by an active order; the task serves
   **every** active order that needs that SKU. Schedule the best-scored task that fits in the
   horizon: fetch as soon as the pickface frees up (copy availability and MoveCap permitting); the
   served orders pick one after another, at most PickCap at a time; the bin then returns. An order
   whose picks are all done frees its lane; the unscheduled order of that station with the largest
   overlap with the remaining SKUs of the active orders takes the lane.
4. **Which station steps** (``interleave``):
   - ``True`` (default): always the station whose pickface frees up first (ties: first station),
     so bin visits are committed roughly in time order over all stations.
   - ``False`` (original): each station runs to completion before the next one starts. A bin copy
     is only free after its *last* return (``BinCopyPool``), so later stations cannot use copies in
     the idle gaps of earlier ones and their schedules stretch (about +50% makespan on the LNS
     benchmark instances).

Task scores (``rule``, higher is better):

- ``max_sharing``: ``n_served * (rt + rt_ret)``, the robot time of the visit times its use
- ``critical_path``: largest remaining ``Σ (rt + p + rt_ret)`` over the served orders
- ``readiness_weighted``: ``max_sharing - pickface idle gap`` (bin arrival after the pickface frees)
- ``adaptive``: run the three rules, keep the best feasible by ``objective(alpha, beta)``

Changes against the scrapped version (``heuristic_gbs.py`` / ``heuristics/gbs.py`` before ed5cadc),
needed for valid solutions (``interleave=False, partition="overlap"`` is the original construction):

- PickCap: the served orders pick one after another (``PickCap`` at a time) instead of all during
  one ``p[k]``; the presence is ``ceil(n / PickCap) * p[k]`` long (plus a MoveCap delay of the return).
- Feasibility: an order counts as scheduled only when all its SKUs were picked; incomplete orders
  are left out of ``order_assignments`` and make the solution infeasible.
- ``Instance`` API, ``HeuristicSolution`` result, ``ExoState`` (blocks, moves, batch start) via
  ``init_state``; tasks and served orders are iterated in sorted order (deterministic).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from heuristics.heuristic_utils import BinEvent, ExoState, HeuristicState, init_state
from heuristics.solution import HeuristicSolution
from instance import Instance

GBS_RULES = ("max_sharing", "critical_path", "readiness_weighted")
GBS_PARTITIONS = ("overlap", "lpt")


@dataclass
class _Lane:
    lane: int
    order: int | None = None
    free_at: int = 0


@dataclass
class _Station:
    """Construction state of one station (the global resources live in HeuristicState)."""

    s: int
    lanes: list[_Lane]
    pending: dict[int, set[int]] = field(default_factory=dict)  # active order -> SKUs still to pick
    unscheduled: list[int] = field(default_factory=list)
    pickface_free: int = 0


@dataclass
class _Task:
    sku: int
    orders: list[int]  # active orders served by this visit
    earliest_fetch: int  # before the MoveCap delay
    copy_id: int
    score: float = 0.0


# ======================================================================
# 1. Partition
# ======================================================================


def partition_orders(inst: Instance, method: str = "overlap") -> dict[int, list[int]]:
    """Station -> orders (in assignment order), see ``GBS_PARTITIONS``."""
    if method == "overlap":
        return _partition_overlap(inst)
    if method == "lpt":
        return _partition_lpt(inst)
    raise ValueError(f"Unknown GBS partition {method!r}; choose from {GBS_PARTITIONS}")


def _partition_lpt(inst: Instance) -> dict[int, list[int]]:
    """LPT on picking workload Σ p; ties -> most shared SKUs, then the first station."""
    p, req = inst.p, inst.orders_req
    work = {o: sum(p[k] for k in req[o]) for o in inst.O}
    station_orders: dict[int, list[int]] = {s: [] for s in inst.S}
    station_skus: dict[int, set[int]] = {s: set() for s in inst.S}
    load = {s: 0 for s in inst.S}
    for o in sorted(inst.O, key=lambda o: work[o], reverse=True):  # stable: order id breaks ties
        skus = set(req[o])
        best_s = min(inst.S, key=lambda s: (load[s], -len(station_skus[s] & skus), inst.S.index(s)))
        station_orders[best_s].append(o)
        station_skus[best_s] |= skus
        load[best_s] += work[o]
    return station_orders


def _partition_overlap(inst: Instance) -> dict[int, list[int]]:
    """rt-overlap gain minus rt load, largest sum_rt first (the original GBS partition)."""
    rt, req = inst.rt, inst.orders_req
    sum_rt = {o: sum(rt[k] for k in req[o]) for o in inst.O}
    station_orders: dict[int, list[int]] = {s: [] for s in inst.S}
    station_skus: dict[int, set[int]] = {s: set() for s in inst.S}
    load = {s: 0.0 for s in inst.S}
    for o in sorted(inst.O, key=lambda o: sum_rt[o], reverse=True):
        skus = set(req[o])
        best_s = max(inst.S, key=lambda s: (sum(rt[k] for k in station_skus[s] & skus) - load[s], -inst.S.index(s)))
        station_orders[best_s].append(o)
        station_skus[best_s] |= skus
        load[best_s] += sum(rt[k] for k in skus)
    return station_orders


# ======================================================================
# 2. Per-station construction
# ======================================================================


def _overlap_weight(o: int, skus: set[int], inst: Instance) -> int:
    return sum(inst.rt[k] + inst.rt_ret[k] for k in set(inst.orders_req[o]) & skus)


def _pick_by_overlap(cands: list[int], skus: set[int], inst: Instance) -> int:
    """Largest (rt + rt_ret) overlap with `skus`; ties -> smaller sum_rt, then earlier in `cands`."""
    rt = inst.rt
    return max(cands, key=lambda o: (_overlap_weight(o, skus, inst), -sum(rt[k] for k in inst.orders_req[o])))


def _seed_lanes(st: _Station, inst: Instance, t0: int) -> None:
    rt = inst.rt
    st.unscheduled.sort(key=lambda o: sum(rt[k] for k in inst.orders_req[o]))
    for ln in st.lanes:
        if not st.unscheduled:
            break
        if not st.pending:
            o = st.unscheduled[0]
        else:
            o = _pick_by_overlap(st.unscheduled, set().union(*(inst.orders_req[a] for a in st.pending)), inst)
        st.unscheduled.remove(o)
        st.pending[o] = set(inst.orders_req[o])
        ln.order, ln.free_at = o, t0


def _score(task: _Task, st: _Station, inst: Instance, rule: str) -> float:
    k, rt, rt_ret, p = task.sku, inst.rt, inst.rt_ret, inst.p
    if rule == "max_sharing":
        return float(len(task.orders) * (rt[k] + rt_ret[k]))
    if rule == "critical_path":
        return float(max(sum(rt[k2] + p[k2] + rt_ret[k2] for k2 in st.pending[o]) for o in task.orders))
    if rule == "readiness_weighted":
        idle_gap = max(0, task.earliest_fetch + rt[k] - st.pickface_free)
        return float(len(task.orders) * (rt[k] + rt_ret[k]) - idle_gap)
    raise ValueError(f"Unknown GBS rule {rule!r}; choose from {GBS_RULES} or 'adaptive'")


def _build_tasks(st: _Station, state: HeuristicState, rule: str) -> list[_Task]:
    inst = state.inst
    lane_of = {ln.order: i for i, ln in enumerate(st.lanes) if ln.order is not None}
    tasks = []
    for k in sorted(set().union(*st.pending.values())):
        served = sorted((o for o in st.pending if k in st.pending[o]), key=lambda o: lane_of[o])
        copy_free, copy_id = state.bin_pools[k].get_earliest_free()
        task = _Task(k, served, max(st.pickface_free - inst.rt[k], copy_free), copy_id)
        task.score = _score(task, st, inst, rule)
        tasks.append(task)
    tasks.sort(key=lambda t: t.score, reverse=True)  # stable: SKU id breaks ties
    return tasks


def _schedule_task(task: _Task, st: _Station, state: HeuristicState) -> BinEvent | None:
    """Commit one visit of task.sku serving task.orders; None (nothing changed) if past the horizon."""
    inst, k = state.inst, task.sku
    rt, rt_ret, p, cap = inst.rt[k], inst.rt_ret[k], inst.p[k], state.pick_cap

    t_fetch = state.move_da.delay_for_movecap_with_pending(task.earliest_fetch, rt, state.move_cap, [])
    pres_start = t_fetch + rt
    picks = [(pres_start + (i // cap) * p, pres_start + (i // cap + 1) * p) for i in range(len(task.orders))]
    t_ret = state.move_da.delay_for_movecap_with_pending(picks[-1][1], rt_ret, state.move_cap, [(t_fetch, pres_start)])
    if t_ret + rt_ret > state.horizon:
        return None

    ev = BinEvent(
        sku=k, copy_id=task.copy_id,
        fetch_start=t_fetch, fetch_end=pres_start,
        presence_start=pres_start, presence_end=t_ret,
        return_start=t_ret, return_end=t_ret + rt_ret,
        orders_served=list(task.orders), pick_intervals=list(picks),
    )
    state.move_da.add_move(ev.fetch_start, ev.fetch_end)
    state.move_da.add_move(ev.return_start, ev.return_end)
    state.bin_pools[k].commit(ev.copy_id, ev.return_end)
    state.station_bin_events[st.s].append(ev)
    st.pickface_free = ev.presence_end
    return ev


def _complete_and_admit(st: _Station, inst: Instance, windows: dict[int, tuple[int, int]]) -> None:
    for ln in st.lanes:
        o = ln.order
        if o is None or st.pending[o]:
            continue
        del st.pending[o]
        ln.order, ln.free_at = None, windows[o][1]
        if st.unscheduled:
            remaining = set().union(*st.pending.values()) if st.pending else set()
            new_o = _pick_by_overlap(st.unscheduled, remaining, inst)
            st.unscheduled.remove(new_o)
            st.pending[new_o] = set(inst.orders_req[new_o])
            ln.order = new_o


def _step(
    st: _Station, state: HeuristicState, rule: str,
    pick_events: dict, windows: dict[int, tuple[int, int]], lane_of: dict[int, int],
) -> bool:
    """Schedule one bin visit at station st. False if no task fits the horizon (station is stuck)."""
    for ln in st.lanes:
        if ln.order is not None:
            lane_of[ln.order] = ln.lane
    ev = task = None
    for task in _build_tasks(st, state, rule):
        ev = _schedule_task(task, st, state)
        if ev is not None:
            break
    if ev is None:
        return False
    for o, (a, b) in zip(task.orders, ev.pick_intervals):
        pick_events[(o, st.s, task.sku)] = (a, b)
        st.pending[o].discard(task.sku)
        w = windows.get(o)
        windows[o] = (a, b) if w is None else (min(w[0], a), max(w[1], b))
    _complete_and_admit(st, state.inst, windows)
    return True


# ======================================================================
# Entry point
# ======================================================================


def _run_gbs_rule(
    inst: Instance, horizon: int, rule: str, exo: ExoState | None, interleave: bool, partition: str
) -> HeuristicSolution:
    state = init_state(inst, horizon, exo)
    t0 = (exo.batch_start_time if exo else 0)
    assignment = partition_orders(inst, partition)
    pick_events: dict[tuple[int, int, int], tuple[int, int]] = {}
    windows: dict[int, tuple[int, int]] = {}
    lane_of: dict[int, int] = {}
    stations = [
        _Station(s, [_Lane(ln, None, t0) for ln in inst.L], unscheduled=list(assignment[s]), pickface_free=t0)
        for s in inst.S
    ]

    if interleave:
        for st in stations:
            _seed_lanes(st, inst, t0)
        active = [st for st in stations if st.pending]  # in station order: min() breaks ties by station
        while active:
            st = min(active, key=lambda x: x.pickface_free)
            if not _step(st, state, rule, pick_events, windows, lane_of) or not st.pending:
                active.remove(st)  # done, or horizon reached: its remaining orders stay incomplete
    else:
        for st in stations:
            _seed_lanes(st, inst, t0)
            while st.pending and _step(st, state, rule, pick_events, windows, lane_of):
                pass

    station_of = {o: s for s, os_ in assignment.items() for o in os_}
    req = inst.orders_req
    complete = [o for o in inst.O if o in windows and all((o, station_of[o], k) in pick_events for k in req[o])]
    order_assignments = {o: (station_of[o], lane_of[o], *windows[o]) for o in complete}
    failed = [o for o in inst.O if o not in order_assignments]
    if failed:
        print(f"[GBS] WARNING: {len(failed)} orders could not be scheduled: {failed}")

    return HeuristicSolution(
        order_assignments=order_assignments,
        bin_events=dict(state.station_bin_events),
        makespan=max((end for *_, end in order_assignments.values()), default=0),
        total_moves=2 * sum(len(evs) for evs in state.station_bin_events.values()),
        feasible=not failed,
        pick_events={key: t for key, t in pick_events.items() if key[0] in order_assignments},
        meta={"heuristic": "gbs", "rule": rule, "interleave": interleave, "partition": partition,
              "assignment": assignment},
    )


def run_gbs(
    inst: Instance,
    *,
    horizon: int = 10000,
    alpha: float = 1.0,
    beta: float = 0.0,
    rule: str = "max_sharing",
    interleave: bool = True,
    partition: str = "overlap",
    exo: ExoState | None = None,
) -> HeuristicSolution:
    """Global Bin Scheduling (see the module docstring).

    ``interleave``: advance the station whose pickface frees first (default) instead of building
    the stations one after another (the original GBS). ``partition``: ``"overlap"`` (original)
    or ``"lpt"`` (load-balanced on picking workload).

    ``alpha`` / ``beta`` only matter for ``rule="adaptive"``, which keeps the best feasible
    rule by ``alpha * C_max + beta * Σ_s span_s`` (``meta["runs"]``: one record per rule).
    """
    if partition not in GBS_PARTITIONS:
        raise ValueError(f"Unknown GBS partition {partition!r}; choose from {GBS_PARTITIONS}")
    if rule != "adaptive":
        if rule not in GBS_RULES:
            raise ValueError(f"Unknown GBS rule {rule!r}; choose from {GBS_RULES} or 'adaptive'")
        return _run_gbs_rule(inst, horizon, rule, exo, interleave, partition)

    best, best_obj, runs = None, float("inf"), []
    for r in GBS_RULES:
        sol = _run_gbs_rule(inst, horizon, r, exo, interleave, partition)
        obj = sol.objective(alpha, beta, inst.S) if sol.feasible else float("inf")
        runs.append({"rule": r, "feasible": sol.feasible, "makespan": sol.makespan, "total_moves": sol.total_moves})
        if best is None or obj < best_obj or (obj == best_obj and sol.total_moves < best.total_moves):
            best, best_obj = sol, obj
    best.meta.update({"heuristic": "gbs", "best_rule": best.meta["rule"], "rule": "adaptive", "runs": runs})
    return best
