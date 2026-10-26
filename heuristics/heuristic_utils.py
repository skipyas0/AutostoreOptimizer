"""Shared machinery for the constructive heuristics (SGC family: SGC, AMA, RDI, CFSS).

Data model (mirrors the CP model in cp_model.py):

* A *bin event* is one visit of a physical bin copy of SKU ``k`` to a station:
  fetch move ``[fetch_start, fetch_end)`` -> presence at the pickface
  ``[presence_start, presence_end)`` -> return move ``[return_start, return_end)``.
  Several orders at the same station may pick from one presence window
  ("shared bin"), at most ``pick_cap`` simultaneously.
* ``HeuristicState`` holds the committed partial schedule plus the instance and
  run parameters (horizon, pick capacity), so helpers only need ``state``.
* ``PlanScratch`` holds the *tentative* bookkeeping while one order is planned
  at one station (pending moves, extended bins, ...). Nothing in ``state`` is
  mutated until ``commit_plan``.

Everything here is behaviour-preserving relative to the pre-refactor
``autostore_heuristic.py`` (see tests/regression/equivalence.py).
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

from instance import Instance

# ============================================================
# Data classes
# ============================================================


@dataclass
class BinEvent:
    sku: int
    copy_id: int
    fetch_start: int
    fetch_end: int
    presence_start: int
    presence_end: int
    return_start: int
    return_end: int
    orders_served: list[int]
    pick_intervals: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class OrderPlan:
    """Result of tentatively scheduling one order at one station."""

    order: int
    station: int
    lane: int
    start: int
    end: int
    bin_events: list[BinEvent]  # new bin events created
    # (sku, bin_event, pick_start, pick_end, new_return_start, new_return_end)
    shared_picks: list[tuple[int, BinEvent, int, int, int, int]]
    score: float = float("inf")
    pick_times: dict[int, tuple[int, int]] = field(default_factory=dict)  # sku -> (start, end)


@dataclass
class ExoState:
    """Resources already consumed outside the (sub)problem being solved.

    Used by the decomposition schemes: lanes busy in ``lanes[(s, ln)]``,
    bin copies of SKU ``k`` blocked during ``blocks[k]``, robot moves in
    ``moves`` count towards MoveCap, and nothing may start before
    ``batch_start_time``.
    """

    lanes: dict[tuple[int, int], list[tuple[int, int]]] | None = None
    blocks: dict[int, list[tuple[int, int]]] | None = None
    moves: list[tuple[int, int]] | None = None
    batch_start_time: int = 0


# ============================================================
# Resource trackers
# ============================================================


class DifferenceArray:
    """Tracks concurrent robot moves via a difference array + prefix sum."""

    def __init__(self, horizon: int):
        self.H = horizon
        self.diff = [0] * (horizon + 2)  # diff[t]: delta at time t
        self.prefix = [0] * (horizon + 2)  # prefix[t]: total moves active at t
        self._dirty = False

    def add_move(self, start: int, end: int) -> None:
        """Register one move active during [start, end)."""
        if start >= end:
            return
        s = max(0, start)
        e = min(end, self.H + 1)
        self.diff[s] += 1
        self.diff[e] -= 1
        self._dirty = True

    def remove_move(self, start: int, end: int) -> None:
        """Undo one move previously added for [start, end)."""
        if start >= end:
            return
        s = max(0, start)
        e = min(end, self.H + 1)
        self.diff[s] -= 1
        self.diff[e] += 1
        self._dirty = True

    def build_prefix(self) -> None:
        """Rebuild cumulative profile from difference array. O(H)."""
        self.prefix[0] = self.diff[0]
        for t in range(1, self.H + 2):
            self.prefix[t] = self.prefix[t - 1] + self.diff[t]
        self._dirty = False

    def delay_for_movecap_with_pending(
        self,
        t_start: int,
        duration: int,
        cap: int,
        pending_moves: list[tuple[int, int]],
        removed_moves: list[tuple[int, int]] | None = None,
    ) -> int:
        """Slide t_start forward until [t_start, t_start+duration) fits under cap,
        counting committed moves plus uncommitted pending (+1) / removed (-1) moves.
        May return a time past the horizon — the caller must check."""
        if self._dirty:
            self.build_prefix()

        t = t_start
        limit = self.H - duration + 1
        while t <= limit:
            ok = True
            for i in range(t, t + duration):
                total = self.prefix[i]
                if removed_moves:
                    for rs, re in removed_moves:
                        if rs <= i < re:
                            total -= 1
                for ms, me in pending_moves:
                    if ms <= i < me:
                        total += 1
                if total >= cap:
                    ok = False
                    t = i + 1
                    break
            if ok:
                return t
        return t


class BinCopyPool:
    """Min-heap tracking availability times of the N[k] physical copies of one SKU."""

    def __init__(self, sku: int, n_copies: int):
        self.sku = sku
        self.n_copies = n_copies
        self._heap: list[tuple[int, int]] = [(0, i) for i in range(n_copies)]  # (available_at, copy_id)
        heapq.heapify(self._heap)

    def get_earliest_free(self) -> tuple[int, int]:
        """Peek at the earliest-available copy. Returns (available_at, copy_id)."""
        return self._heap[0]

    def commit(self, copy_id: int, return_end: int) -> None:
        """Set copy_id's availability to max(current, return_end)."""
        for i, (t, cid) in enumerate(self._heap):
            if cid == copy_id:
                self._heap[i] = (max(t, return_end), copy_id)
                heapq.heapify(self._heap)
                return
        raise ValueError(f"copy_id {copy_id} not found in pool for SKU {self.sku}")

    def __deepcopy__(self, memo):
        new = BinCopyPool.__new__(BinCopyPool)
        new.sku = self.sku
        new.n_copies = self.n_copies
        new._heap = list(self._heap)
        return new


# ============================================================
# Mutable construction state
# ============================================================


@dataclass(frozen=True)
class ShareOptions:
    """How the SGC/RDI planners treat bin sharing (all off = original behaviour).

    extend_window: a shared pick may start up to the order's own best new-fetch pick
        for that SKU (instead of at most rt[k] after the picker is ready).
    reward: lambda; plan score -= lambda * sum over shared picks of (rt[k] + rt_ret[k])
        (robot time saved by not fetching a new bin).
    legacy_rt_p: weight of the older RDI bonus, score -= w * sum over shared picks of (rt[k] + p[k]).
    """

    extend_window: bool = False
    reward: float = 0.0
    legacy_rt_p: float = 0.0


@dataclass
class HeuristicState:
    """Committed partial schedule + instance/run parameters."""

    inst: Instance
    horizon: int
    pick_cap: int  # max simultaneous picks from one presented bin
    move_cap: int  # global MoveCap
    lane_free: dict[tuple[int, int], int]  # (s, ln) -> earliest free time
    pickface_free: dict[int, int]  # s -> earliest time pickface is free
    move_da: DifferenceArray  # global move profile
    bin_pools: dict[int, BinCopyPool]  # k -> BinCopyPool
    station_bin_events: dict[int, list[BinEvent]]  # s -> BinEvents
    pickface_intervals: dict[int, list[tuple[int, int]]]  # s -> [(presence_start, presence_end)]
    lane_intervals: dict[tuple[int, int], list[tuple[int, int]]]  # (s, ln) -> [(start, end)]
    first_pick: dict[int, int]  # s -> earliest pick start at s
    last_pick: dict[int, int]  # s -> latest pick end at s
    share: ShareOptions = field(default_factory=ShareOptions)


def init_state(
    inst: Instance,
    horizon: int,
    exo: ExoState | None = None,
    share: ShareOptions | None = None,
) -> HeuristicState:
    """Build the initial state, respecting exogenous resources and batch start time."""
    S, L, K, N = inst.S, inst.L, inst.K, inst.N
    exo = exo or ExoState()
    t0 = exo.batch_start_time

    move_da = DifferenceArray(horizon)
    if exo.moves:
        for st, en in exo.moves:
            st = max(0, min(horizon - 1, int(st)))
            en = max(0, min(horizon - 1, int(en)))
            if st < en:
                move_da.add_move(st, en)
    move_da.build_prefix()

    bin_pools = {k: BinCopyPool(k, N[k]) for k in K}
    for pool in bin_pools.values():
        pool._heap = [(t0, i) for i in range(pool.n_copies)]
    if exo.blocks:
        for k, blocks in exo.blocks.items():
            if k in bin_pools:
                pool = bin_pools[k]
                for st, en in sorted(blocks):
                    avail, cid = heapq.heappop(pool._heap)
                    heapq.heappush(pool._heap, (max(avail, en), cid))

    lane_intervals = {(s, ln): [] for s in S for ln in L}
    if exo.lanes:
        for (s, ln), intervals in exo.lanes.items():
            if (s, ln) in lane_intervals:
                lane_intervals[(s, ln)] = sorted([(st, en) for st, en in intervals if st < horizon])

    return HeuristicState(
        inst=inst,
        horizon=horizon,
        pick_cap=inst.pickcap,
        move_cap=inst.movecap,
        lane_free={(s, ln): t0 for s in S for ln in L},
        pickface_free={s: t0 for s in S},
        move_da=move_da,
        bin_pools=bin_pools,
        station_bin_events={s: [] for s in S},
        pickface_intervals={s: [] for s in S},
        lane_intervals=lane_intervals,
        first_pick={s: horizon + 1 for s in S},
        last_pick={s: 0 for s in S},
        share=share or ShareOptions(),
    )


def earliest_lane(state: HeuristicState, s: int) -> tuple[int, int] | None:
    """(lane, free_time) of the earliest-free lane at station s, or None."""
    L_at_s = [ln for (ss, ln) in state.lane_free if ss == s]
    if not L_at_s:
        return None
    best_lane = min(L_at_s, key=lambda ln: state.lane_free[(s, ln)])
    return best_lane, state.lane_free[(s, best_lane)]


# ============================================================
# Per-order tentative bookkeeping
# ============================================================


@dataclass
class PlanScratch:
    """Tentative changes while planning ONE order at ONE station."""

    local_intervals: list[tuple[int, int]]  # station pickface presence intervals incl. tentative
    new_bin_events: list[BinEvent] = field(default_factory=list)
    # (sku, bin_event, pick_start, pick_end, new_return_start, new_return_end)
    shared_picks: list[tuple[int, BinEvent, int, int, int, int]] = field(default_factory=list)
    pending_moves: list[tuple[int, int]] = field(default_factory=list)
    removed_moves: list[tuple[int, int]] = field(default_factory=list)
    pending_copies: dict[int, list[tuple[int, int]]] = field(default_factory=lambda: defaultdict(list))
    pick_times: dict[int, tuple[int, int]] = field(default_factory=dict)
    temp_pick_intervals: dict[int, list[tuple[int, int]]] = field(default_factory=dict)  # id(ev) -> picks
    temp_bounds: dict[int, tuple[int, int]] = field(default_factory=dict)  # id(ev) -> (pres_end, ret_end)

    @classmethod
    def for_station(cls, state: HeuristicState, s: int) -> "PlanScratch":
        return cls(local_intervals=list(state.pickface_intervals[s]))

    def delay_move(self, state: HeuristicState, t: int, duration: int) -> int:
        """Earliest start >= t for a robot move of `duration` under MoveCap."""
        return state.move_da.delay_for_movecap_with_pending(
            t, duration, state.move_cap, self.pending_moves, removed_moves=self.removed_moves
        )

    def apply_shared_pick(
        self,
        state: HeuristicState,
        s: int,
        k: int,
        shared: BinEvent,
        pick_start: int,
        new_ret_start: int,
        new_ret_end: int,
    ) -> tuple[int, bool]:
        """Record a pick from an existing/tentative bin. Returns (pick_end, presence_extended)."""
        pick_end = pick_start + state.inst.p[k]
        ev_id = id(shared)
        if ev_id not in self.temp_pick_intervals:
            self.temp_pick_intervals[ev_id] = list(shared.pick_intervals)
        self.temp_pick_intervals[ev_id].append((pick_start, pick_end))

        old_pres_end, old_ret_end = self.temp_bounds.get(ev_id, (shared.presence_end, shared.return_end))
        self.temp_bounds[ev_id] = (new_ret_start, new_ret_end)

        extended = new_ret_start > old_pres_end
        if extended:
            for idx_iv, (iv_s, iv_e) in enumerate(self.local_intervals):
                if iv_s == shared.presence_start and iv_e == old_pres_end:
                    self.local_intervals[idx_iv] = (shared.presence_start, new_ret_start)
                    break
            self.local_intervals.sort(key=lambda x: x[0])

            if (old_pres_end, old_ret_end) in self.pending_moves:
                self.pending_moves.remove((old_pres_end, old_ret_end))
            elif shared in state.station_bin_events[s]:
                orig_ret = (shared.return_start, shared.return_end)
                if orig_ret not in self.removed_moves:
                    self.removed_moves.append(orig_ret)
            self.pending_moves.append((new_ret_start, new_ret_end))

        self.pick_times[k] = (pick_start, pick_end)
        self.shared_picks.append((k, shared, pick_start, pick_end, new_ret_start, new_ret_end))
        return pick_end, extended

    def add_new_bin(self, ev: BinEvent) -> None:
        """Record a tentative new bin visit (one pick by the current order)."""
        self.new_bin_events.append(ev)
        self.local_intervals.append((ev.presence_start, ev.presence_end))
        self.local_intervals.sort(key=lambda x: x[0])
        self.pending_moves.append((ev.fetch_start, ev.fetch_end))
        self.pending_moves.append((ev.return_start, ev.return_end))
        self.pending_copies[ev.sku].append((ev.copy_id, ev.return_end))
        self.pick_times[ev.sku] = ev.pick_intervals[0]

    def score(self, state: HeuristicState, alpha: float, beta: float, start: int, end: int) -> float:
        """alpha*end + beta*(end-start), minus the sharing bonuses configured in state.share."""
        score = alpha * end + beta * (end - start)
        opt = state.share
        if opt.legacy_rt_p or opt.reward:
            rt, rt_ret, p = state.inst.rt, state.inst.rt_ret, state.inst.p
            ks = [k for (k, *_rest) in self.shared_picks]
            bonus = opt.legacy_rt_p * sum(rt[k] + p[k] for k in ks) + opt.reward * sum(rt[k] + rt_ret[k] for k in ks)
            score = score - bonus
        return score

    def window(self, default: int) -> tuple[int, int]:
        """(order_start, order_end) spanned by this order's picks."""
        start = min((ps for ps, pe in self.pick_times.values()), default=default)
        end = max((pe for ps, pe in self.pick_times.values()), default=default)
        return start, end

    def to_plan(self, o: int, s: int, lane: int, start: int, end: int, score: float) -> OrderPlan:
        return OrderPlan(
            order=o,
            station=s,
            lane=lane,
            start=start,
            end=end,
            bin_events=self.new_bin_events,
            shared_picks=self.shared_picks,
            score=score,
            pick_times=self.pick_times,
        )


# ============================================================
# Feasibility helpers
# ============================================================


def is_capacity_available(
    existing_intervals: list[tuple[int, int]], p_start: int, p_end: int, pick_cap: int
) -> bool:
    """Would adding [p_start, p_end) exceed pick_cap concurrent picks?"""
    events = [(p_start, 1), (p_end, -1)]
    for s, e in existing_intervals:
        if max(s, p_start) < min(e, p_end):
            events.append((s, 1))
            events.append((e, -1))
    # ends (-1) before starts (+1) at equal times
    events.sort(key=lambda x: (x[0], x[1]))
    concurrent = 0
    for _t, change in events:
        concurrent += change
        if concurrent > pick_cap:
            return False
    return True


def find_pick_slot(
    existing_intervals: list[tuple[int, int]],
    earliest_start: int,
    duration: int,
    presence_end: int,
    pick_cap: int,
) -> int | None:
    """Earliest start >= earliest_start where a pick fits before presence_end under pick_cap."""
    candidate = earliest_start
    while candidate + duration <= presence_end:
        if is_capacity_available(existing_intervals, candidate, candidate + duration, pick_cap):
            return candidate
        next_candidate = presence_end
        for s, e in existing_intervals:
            if max(s, candidate) < min(e, candidate + duration):
                next_candidate = min(next_candidate, e)
        if next_candidate <= candidate:
            break  # no progress — should not happen
        candidate = next_candidate
    return None


def _can_extend_bin_event(
    ev: BinEvent,
    s: int,
    req_presence_end: int,
    state: HeuristicState,
    scratch: PlanScratch,
    curr_pres_end: int,
    curr_ret_end: int,
) -> tuple[int, int] | None:
    """Can `ev` stay at the pickface until at least req_presence_end?

    Returns (new_return_start, new_return_end) or None if infeasible.
    """
    k = ev.sku
    ret_dur = state.inst.rt_ret[k]

    # effective pending/removed moves excluding ev's current return move
    eff_pending = [
        (ms, me) for (ms, me) in scratch.pending_moves if not (ms == curr_pres_end and me == curr_ret_end)
    ]
    eff_removed = list(scratch.removed_moves) if scratch.removed_moves else []
    if ev in state.station_bin_events[s]:
        orig_ret = (ev.return_start, ev.return_end)
        if orig_ret not in eff_removed:
            eff_removed.append(orig_ret)

    new_ret_start = state.move_da.delay_for_movecap_with_pending(
        req_presence_end, ret_dur, state.move_cap, eff_pending, eff_removed
    )
    new_ret_end = new_ret_start + ret_dur
    if new_ret_end > state.horizon:
        return None

    # pickface no-overlap for the extended presence [ev.presence_start, new_ret_start)
    for iv_s, iv_e in scratch.local_intervals:
        if iv_s == ev.presence_start and iv_e == curr_pres_end:
            continue
        if max(ev.presence_start, iv_s) < min(new_ret_start, iv_e):
            return None

    # no later visit of the same physical copy may start fetching before new_ret_end
    for ev_list in state.station_bin_events.values():
        for other in ev_list:
            if other is not ev and other.sku == k and other.copy_id == ev.copy_id:
                if other.fetch_start >= ev.return_end and other.fetch_start < new_ret_end:
                    return None
    for pcid, _ in scratch.pending_copies.get(k, []):
        if pcid == ev.copy_id:
            return None

    return (new_ret_start, new_ret_end)


def find_shared_bin(
    candidates: list[BinEvent],
    k: int,
    t_cursor: int,
    state: HeuristicState,
    s: int,
    scratch: PlanScratch,
    max_pick_start: float | None = None,
) -> tuple[BinEvent | None, int | None, int | None, int | None]:
    """Find a bin event of SKU k (committed or tentative) whose presence can cover a pick.

    A shared pick may start at most at ``max_pick_start`` (default: t_cursor + rt[k]).
    The presence window may be extended if the picker is ready by its current end
    (no dead air). Returns (bin_event, pick_start, new_return_start, new_return_end) or all-None.
    """
    p, rt, rt_ret = state.inst.p, state.inst.rt, state.inst.rt_ret
    horizon, pick_cap = state.horizon, state.pick_cap
    best_ev = None
    best_start = float("inf")
    best_ret_start = None
    best_ret_end = None

    limit = t_cursor + rt[k] if max_pick_start is None else max_pick_start

    for ev in candidates:
        if ev.sku != k:
            continue
        ev_id = id(ev)
        ev_picks = scratch.temp_pick_intervals.get(ev_id, ev.pick_intervals)
        curr_pres_end, curr_ret_end = scratch.temp_bounds.get(ev_id, (ev.presence_end, ev.return_end))

        earliest = max(t_cursor, ev.presence_start)
        if earliest > limit:
            continue

        # 1. fit inside the current presence window
        if earliest + p[k] <= curr_pres_end:
            slot = find_pick_slot(ev_picks, earliest, p[k], curr_pres_end, pick_cap)
            if slot is not None and slot <= limit and slot < best_start:
                best_ev = ev
                best_start = slot
                best_ret_start = curr_pres_end
                best_ret_end = curr_ret_end
                continue

        # 2. extend the presence window if the picker is ready by curr_pres_end
        if earliest <= curr_pres_end:
            ext_limit = min(horizon - rt_ret[k], curr_pres_end + p[k])
            if earliest + p[k] <= ext_limit:
                slot = find_pick_slot(ev_picks, earliest, p[k], ext_limit, pick_cap)
                if slot is not None and slot <= curr_pres_end and slot < best_start:
                    req_pres_end = max(curr_pres_end, slot + p[k])
                    ext = _can_extend_bin_event(ev, s, req_pres_end, state, scratch, curr_pres_end, curr_ret_end)
                    if ext is not None:
                        best_ev = ev
                        best_start = slot
                        best_ret_start, best_ret_end = ext

    if best_ev is None:
        return (None, None, None, None)
    return (best_ev, best_start, best_ret_start, best_ret_end)


def share_pick_limit(
    candidates: list[BinEvent],
    k: int,
    t_cursor: int,
    state: HeuristicState,
    new_fetch: Callable[[], BinEvent | None],
) -> float | None:
    """Upper bound on a shared pick's start for find_shared_bin (None = default rt[k] window).

    With ShareOptions.extend_window, waiting for an existing bin of SKU k is allowed as long as the
    pick starts no later than the order's own best new fetch of k would (``new_fetch()``, evaluated
    lazily and only if a candidate bin of SKU k exists; infeasible fetch -> no limit).
    """
    if not state.share.extend_window or not any(ev.sku == k for ev in candidates):
        return None
    alt = new_fetch()
    return max(t_cursor + state.inst.rt[k], alt.presence_start if alt is not None else float("inf"))


def earliest_feasible_fetch(
    k: int, desired_start: int, state: HeuristicState, scratch: PlanScratch
) -> tuple[int, int]:
    """Earliest fetch start for SKU k (copy availability + MoveCap).

    Returns (fetch_start, copy_id). Raises ValueError if it cannot finish within the horizon.
    """
    rt, rt_ret = state.inst.rt, state.inst.rt_ret
    pool = state.bin_pools[k]
    avail_at, copy_id = pool.get_earliest_free()

    # tentative commits of this order may push copy availability later
    pending_for_k = scratch.pending_copies.get(k, [])
    if pending_for_k:
        effective_times: list[tuple[int, int]] = list(pool._heap)
        for pcid, p_ret_end in pending_for_k:
            for i, (t, cid) in enumerate(effective_times):
                if cid == pcid:
                    effective_times[i] = (max(t, p_ret_end), cid)
                    break
        effective_times.sort()
        avail_at, copy_id = effective_times[0]

    t_start = scratch.delay_move(state, max(desired_start, avail_at), rt[k])

    if t_start + rt[k] + rt_ret[k] > state.horizon:
        raise ValueError(f"SKU {k} infeasible: fetch start {t_start} exceeds horizon {state.horizon}")
    return t_start, copy_id


# ============================================================
# Commit
# ============================================================


def commit_plan(plan: OrderPlan, state: HeuristicState) -> None:
    """Commit a chosen OrderPlan into the mutable HeuristicState."""
    s = plan.station
    ln = plan.lane

    state.lane_free[(s, ln)] = plan.end
    state.pickface_free[s] = max(
        state.pickface_free[s],
        max((ev.presence_end for ev in plan.bin_events), default=state.pickface_free[s]),
    )

    for ev in plan.bin_events:
        state.move_da.add_move(ev.fetch_start, ev.fetch_end)
        state.move_da.add_move(ev.return_start, ev.return_end)
        state.bin_pools[ev.sku].commit(ev.copy_id, ev.return_end)
        state.station_bin_events[s].append(ev)
        state.pickface_intervals[s].append((ev.presence_start, ev.presence_end))

    state.pickface_intervals[s].sort(key=lambda x: x[0])
    state.lane_intervals[(s, ln)].append((plan.start, plan.end))
    state.lane_intervals[(s, ln)].sort(key=lambda x: x[0])

    for k, shared_ev, pick_start, pick_end, new_ret_start, new_ret_end in plan.shared_picks:
        shared_ev.orders_served.append(plan.order)
        shared_ev.pick_intervals.append((pick_start, pick_end))
        shared_ev.pick_intervals.sort(key=lambda x: x[0])

        if new_ret_start > shared_ev.return_start:
            if shared_ev not in plan.bin_events:
                state.move_da.remove_move(shared_ev.return_start, shared_ev.return_end)
                state.move_da.add_move(new_ret_start, new_ret_end)
            shared_ev.presence_end = new_ret_start
            shared_ev.return_start = new_ret_start
            shared_ev.return_end = new_ret_end
            state.bin_pools[k].commit(shared_ev.copy_id, new_ret_end)

    state.pickface_intervals[s] = sorted((ev.presence_start, ev.presence_end) for ev in state.station_bin_events[s])
    state.pickface_free[s] = max(
        (ev.presence_end for ev in state.station_bin_events[s]),
        default=state.pickface_free[s],
    )

    state.move_da.build_prefix()

    if plan.start < state.first_pick.get(s, float("inf")):
        state.first_pick[s] = plan.start
    if plan.end > state.last_pick.get(s, 0):
        state.last_pick[s] = plan.end


def record_plan(plan: OrderPlan, order_assignments: dict, pick_events: dict) -> None:
    """Copy a committed plan into the solution dictionaries."""
    order_assignments[plan.order] = (plan.station, plan.lane, plan.start, plan.end)
    for k, times in plan.pick_times.items():
        pick_events[(plan.order, plan.station, k)] = times
