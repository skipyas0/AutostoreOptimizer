#!/usr/bin/env python3
"""
AutoStore Simple Greedy Constructive (SGC) Heuristic.
Mirrors variable names from the CP model (cp_model.py).
"""
import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
from instance import Instance


# ============================================================
# PHASE 0 — Infrastructure
# ============================================================

# ------ [P0-4] BinEvent dataclass ------

@dataclass
class BinEvent:
    """One bin visit at a station: fetch -> presence -> return."""
    sku: int
    copy_id: int
    fetch_start: int
    fetch_end: int
    presence_start: int
    presence_end: int
    return_start: int
    return_end: int
    orders_served: list[int] = field(default_factory=list)


# ------ [P0-4 cont.] OrderPlan dataclass ------

@dataclass
class OrderPlan:
    """Result of tentatively scheduling one order at one station."""
    order: int
    station: int
    lane: int
    start: int
    end: int
    bin_events: list[BinEvent]  # new bin events created
    shared_picks: list[tuple[int, BinEvent, int]]  # (sku, bin_event, pick_end)
    score: float = float("inf")
    pick_times: dict[int, tuple[int, int]] = field(default_factory=dict)  # sku -> (pick_start, pick_end)


# ------ [P0-2] DifferenceArray class ------

class DifferenceArray:
    """Tracks concurrent moves via a difference array + prefix sum."""

    def __init__(self, horizon: int):
        """Initialise with a time horizon H."""
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

    def range_max(self, t1: int, t2: int) -> int:
        """Max concurrent moves in [t1, t2] using current prefix. O(t2-t1)."""
        if self._dirty:
            self.build_prefix()
        t1 = max(0, t1)
        t2 = min(t2, self.H)
        if t1 > t2:
            return 0
        return max(self.prefix[t1:t2 + 1])

    def delay_for_movecap(self, t_start: int, duration: int, cap: int) -> int:
        """Slide t_start forward until [t_start, t_start+duration) fits under cap."""
        if self._dirty:
            self.build_prefix()
        if cap is None:
            return t_start
        t = t_start
        limit = self.H - duration + 1
        while t <= limit:
            ok = True
            for i in range(t, t + duration):
                if self.prefix[i] >= cap:
                    ok = False
                    t = i + 1
                    break
            if ok:
                return t
        return t  # may exceed horizon — caller must check

    def delay_for_movecap_with_pending(
            self, t_start: int, duration: int, cap: int,
            pending_moves: list[tuple[int, int]]
    ) -> int:
        """Like delay_for_movecap but also accounts for uncommitted pending moves."""
        if cap is None:
            return t_start
        if self._dirty:
            self.build_prefix()

        t = t_start
        limit = self.H - duration + 1
        while t <= limit:
            ok = True
            for i in range(t, t + duration):
                total = self.prefix[i]
                for (ms, me) in pending_moves:
                    if ms <= i < me:
                        total += 1
                if total >= cap:
                    ok = False
                    t = i + 1
                    break
            if ok:
                return t
        return t


# ------ [P0-3] BinCopyPool class ------

class BinCopyPool:
    """Min-heap tracking N[k] physical copy availability times for one SKU."""

    def __init__(self, sku: int, n_copies: int):
        """Initialise pool for SKU k with n_copies copies, all free at t=0."""
        self.sku = sku
        self.n_copies = n_copies
        # Heap entries: (available_at, copy_id)
        self._heap: list[tuple[int, int]] = [(0, i) for i in range(n_copies)]
        heapq.heapify(self._heap)

    def get_earliest_free(self) -> tuple[int, int]:
        """Peek at the earliest-available copy. Returns (available_at, copy_id)."""
        return self._heap[0]

    def commit(self, copy_id: int, return_end: int) -> None:
        """Update copy_id's availability to return_end after committing a bin event."""
        for i, (t, cid) in enumerate(self._heap):
            if cid == copy_id:
                self._heap[i] = (return_end, copy_id)
                heapq.heapify(self._heap)
                return
        raise ValueError(f"copy_id {copy_id} not found in pool for SKU {self.sku}")

    def peek_nth(self, n: int) -> int:
        """Return the availability time of the n-th earliest copy (0-indexed)."""
        sorted_copies = sorted(self._heap)
        return sorted_copies[n][0] if n < len(sorted_copies) else float("inf")

    def __deepcopy__(self, memo):
        """Support deep copy for tentative state snapshots."""
        new = BinCopyPool.__new__(BinCopyPool)
        new.sku = self.sku
        new.n_copies = self.n_copies
        new._heap = list(self._heap)
        return new


# ------ [P0-5] HeuristicState ------

@dataclass
class HeuristicState:
    """Mutable state tracked during the greedy constructive loop."""
    lane_free: dict[tuple[int, int], int]  # (s, ln) -> earliest free time
    pickface_free: dict[int, int]  # s -> earliest time pickface is free
    move_da: DifferenceArray  # global move profile
    bin_pools: dict[int, BinCopyPool]  # k -> BinCopyPool
    station_bin_events: dict[int, list[BinEvent]]  # s -> list of BinEvents
    pickface_intervals: dict[int, list[tuple[int, int]]]
    lane_intervals: dict[tuple[int, int], list[tuple[int, int]]]  # s -> [(start, end)]
    first_pick: dict[int, int]  # s -> earliest pick start at s
    last_pick: dict[int, int]  # s -> latest pick end at s
    move_cap: Optional[int]  # global MoveCap (None = no cap)

    


# ------ [P0-6] compute_U ------

def compute_U(orders_req: dict[int, list[int]], K: list[int],
              L: list[int]) -> dict[int, int]:
    """Compute U[k] = ceil(demand_k / |L|), mirroring the CP model."""
    need_count: dict[int, int] = defaultdict(int)
    for o in orders_req:
        for k in orders_req[o]:
            need_count[k] += 1
    # Lcap = max(1, len(L))
    # return {k: (math.ceil(need_count[k] / Lcap) if need_count[k] > 0 else 0) for k in K}
    # Conservative upper bound: one bin trip per needed item.
    # While Lcap concurrent lanes imply we could share bins efficiently,
    # the heuristic may not perfectly cluster, so we allow up to need_count[k] trips.
    return {k: need_count[k] for k in K}


# ============================================================
# PHASE 1 — Baseline SGC
# ============================================================

# ------ [P1-1] static_priority_sort ------

def static_priority_sort(orders_req: dict[int, list[int]],
                         O: list[int]) -> list[int]:
    """Sort order ids by sharing degree ascending (fewest shared SKUs first)."""
    req_sets = {o: set(req) for o, req in orders_req.items()}

    def sharing_degree(o: int) -> int:
        req = req_sets[o]
        return sum(1 for o2 in O if o2 != o and not req.isdisjoint(req_sets[o2]))

    return sorted(O, key=sharing_degree, reverse=False)


# ------ [P1-2] find_shared_bin ------

def find_shared_bin(station_bin_events: list[BinEvent], k: int,
                    t_cursor: int, p: dict[int, int]) -> Optional[BinEvent]:
    """Find an existing BinEvent at a station for SKU k whose presence covers t_cursor."""
    # We can use a bin even if it arrives AFTER t_cursor (we just wait for it).
    # We prefer the earliest-starting feasible bin that can fit us.
    best_ev = None
    best_start = float("inf")

    for ev in station_bin_events:
        if ev.sku == k:
            # Can we fit a pick of duration? Not passed here, but let's assume caller checks fit.
            start = max(t_cursor, ev.presence_start)
            # Check if the pick fits within the bin presence
            if start + p[k] <= ev.presence_end:
                if start < best_start:
                    best_ev = ev
                    best_start = start
    return best_ev


# ------ [P1-3] earliest_feasible_fetch ------

def earliest_feasible_fetch(
        k: int, s: int, desired_start: int,
        state: HeuristicState,
        rt: dict[int, int], rt_ret: dict[int, int],
        pending_moves: list[tuple[int, int]],
        pending_copies: dict[int, list[tuple[int, int]]],
        horizon: int,
) -> tuple[int, int]:
    """Find earliest fetch start for SKU k at station s, respecting all constraints.

    Returns (fetch_start, copy_id). Raises ValueError if infeasible within horizon.
    """
    pool = state.bin_pools[k]

    # Step 1: earliest available copy, accounting for pending (tentative) commits
    avail_at, copy_id = pool.get_earliest_free()

    # Check if any pending copies for this SKU push availability later
    pending_for_k = pending_copies.get(k, [])
    if pending_for_k:
        effective_times: list[tuple[int, int]] = list(pool._heap)
        for (pcid, p_ret_end) in pending_for_k:
            for i, (t, cid) in enumerate(effective_times):
                if cid == pcid:
                    effective_times[i] = (max(t, p_ret_end), cid)
                    break
        effective_times.sort()
        avail_at, copy_id = effective_times[0]

    # Step 2: basic lower bound
    t_start = max(desired_start, avail_at)

    # Step 3: delay for MoveCap (fetch move occupies [t_start, t_start + rt[k]))
    if state.move_cap is not None:
        t_start = state.move_da.delay_for_movecap_with_pending(
            t_start, rt[k], state.move_cap, pending_moves
        )

    if t_start + rt[k] + rt_ret[k] > horizon:
        raise ValueError(f"SKU {k} infeasible: fetch start {t_start} exceeds horizon {horizon}")

    return t_start, copy_id


# ------ [P1-4] plan_order_at_station ------

def plan_order_at_station(
        o: int, s: int,
        state: HeuristicState,
        orders_req: dict[int, list[int]],
        rt: dict[int, int], rt_ret: dict[int, int], p: dict[int, int],
        N: dict[int, int],
        horizon: int,
        ALPHA: float, BETA: float,
        demand_count: dict[int, int],
) -> Optional[OrderPlan]:
    """Tentatively schedule order o at station s. Returns OrderPlan or None if infeasible."""
    L_at_s = [ln for (ss, ln) in state.lane_free if ss == s]
    if not L_at_s:
        return None

    # Pick the earliest-free lane
    best_lane = min(L_at_s, key=lambda ln: state.lane_free[(s, ln)])
    t_lane = state.lane_free[(s, best_lane)]
    t_order_start = t_lane

    # Sort SKUs by demand descending (most popular first)
    skus_sorted = sorted(orders_req[o], key=lambda k: demand_count.get(k, 0), reverse=True)

    # t_cursor tracks when the ORDER is free (picker availability).
    # We decouple this from pickface availability (current_station_free).
    t_cursor = t_order_start
    current_station_free = state.pickface_free[s]

    new_bin_events: list[BinEvent] = []
    shared_picks: list[tuple[int, int, int]] = []
    pending_moves: list[tuple[int, int]] = []
    pending_copies: dict[int, list[tuple[int, int]]] = defaultdict(list)
    pick_times_dict: dict[int, tuple[int, int]] = {}

    for k in skus_sorted:
        # --- Try shared bin first ---
        # 1. Look in existing bin events from prior orders
        shared = find_shared_bin(state.station_bin_events[s], k, t_cursor, p)

        # 2. Look in bin events created by *this* order
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
            # We want presence_start >= t_cursor (picker ready)
            # AND presence_start >= current_station_free (pickface clear)
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

        # Check return move fits under MoveCap
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


# ------ [P1-6] commit_plan ------

def commit_plan(plan: OrderPlan, state: HeuristicState) -> None:
    """Commit a chosen OrderPlan into the mutable HeuristicState."""
    s = plan.station
    ln = plan.lane

    state.lane_free[(s, ln)] = plan.end
    state.pickface_free[s] = max(
        state.pickface_free[s],
        max((ev.presence_end for ev in plan.bin_events), default=state.pickface_free[s])
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

    for k, shared_ev, pick_end in plan.shared_picks:
        shared_ev.orders_served.append(plan.order)

    state.move_da.build_prefix()

    if plan.start < state.first_pick.get(s, float("inf")):
        state.first_pick[s] = plan.start
    if plan.end > state.last_pick.get(s, 0):
        state.last_pick[s] = plan.end


# ------ [P1-7] run_sgc ------

@dataclass
class Solution:
    """Complete heuristic solution."""
    order_assignments: dict[int, tuple[int, int, int, int]]  # o -> (s, ln, start, end)
    bin_events: dict[int, list[BinEvent]]  # s -> list of BinEvents
    makespan: int
    total_moves: int
    feasible: bool
    pick_events: dict[tuple[int, int, int], tuple[int, int]] = field(default_factory=dict)
    # (o, s, k) -> (pick_start, pick_end)


def init_state(
        S: list[int], L: list[int], K: list[int],
        N: dict[int, int], horizon: int, move_cap: Optional[int],
) -> HeuristicState:
    """Build initial HeuristicState with everything at t=0."""
    lane_free = {(s, ln): 0 for s in S for ln in L}
    pickface_free = {s: 0 for s in S}
    move_da = DifferenceArray(horizon)
    move_da.build_prefix()
    bin_pools = {k: BinCopyPool(k, N[k]) for k in K}
    station_bin_events: dict[int, list[BinEvent]] = {s: [] for s in S}
    first_pick = {s: horizon + 1 for s in S}
    last_pick = {s: 0 for s in S}

    return HeuristicState(
        lane_free=lane_free,
        pickface_free=pickface_free,
        move_da=move_da,
        bin_pools=bin_pools,
        station_bin_events=station_bin_events,
        pickface_intervals={s: [] for s in S},
        lane_intervals={(s, ln): [] for s in S for ln in L},
        first_pick=first_pick,
        last_pick=last_pick,
        move_cap=move_cap,
    )


def run_sgc(
        instance,
        *args,
        **kwargs
) -> Solution:
    """Run the Simple Greedy Constructive heuristic. [P1-7]"""
    if not isinstance(instance, Instance):
        # Backward compatibility fallback
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
        instance = Instance(S, L, K, orders_req, rt, p, N, rt_ret=rt_ret)
    else:
        horizon = kwargs.get('horizon', args[0] if len(args) > 0 else 10000)
        move_cap = kwargs.get('move_cap', args[1] if len(args) > 1 else None)
        ALPHA = kwargs.get('ALPHA', args[2] if len(args) > 2 else 1.0)
        BETA = kwargs.get('BETA', args[3] if len(args) > 3 else 0.0)
        rt_ret = kwargs.get('rt_ret', args[4] if len(args) > 4 else None)

    S, L, K, orders_req, rt, p, N = instance
    O = instance.O
    if 'rt_ret' not in locals() or rt_ret is None:
        rt_ret = instance.rt_ret

    state = init_state(S, L, K, N, horizon, move_cap)

    demand_count: dict[int, int] = defaultdict(int)
    for o_req in orders_req.values():
        for k in o_req:
            demand_count[k] += 1

    sorted_orders = static_priority_sort(orders_req, O)

    order_assignments: dict[int, tuple[int, int, int, int]] = {}
    pick_events_map: dict[tuple[int, int, int], tuple[int, int]] = {}
    failed_orders: list[int] = []

    for o in sorted_orders:
        best_plan: Optional[OrderPlan] = None

        for s in S:
            plan = plan_order_at_station(
                o, s, state, orders_req, rt, rt_ret, p, N, horizon, ALPHA, BETA, demand_count
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
    total_moves = sum(
        len(evts) * 2 for evts in state.station_bin_events.values()
    )
    feasible = len(failed_orders) == 0

    if failed_orders:
        print(f"[SGC] WARNING: {len(failed_orders)} orders could not be scheduled: {failed_orders}")

    return Solution(
        order_assignments=order_assignments,
        bin_events=dict(state.station_bin_events),
        makespan=makespan,
        total_moves=total_moves,
        feasible=feasible,
        pick_events=pick_events_map,
    )


# ============================================================
# Visualisation helpers (bridge to schedule_visualizer2)
# ============================================================

class _MockIV:
    """Placeholder for a CP interval variable — used only as a unique dict key."""
    __slots__ = ()


@dataclass
class _MockIVSol:
    """Lightweight stand-in for a CP interval-variable solution."""
    present: bool
    start: int = 0
    end: int = 0

    def is_present(self) -> bool:
        return self.present

    def get_start(self) -> int:
        return self.start

    def get_end(self) -> int:
        return self.end


class _MockCPSol:
    """Stand-in for a docplex CpoSolveResult, backed by plain dicts."""

    def __init__(self) -> None:
        self._d: dict[int, _MockIVSol] = {}

    def _set(self, iv: _MockIV, **kw) -> None:
        self._d[id(iv)] = _MockIVSol(**kw)

    def get_var_solution(self, iv: _MockIV) -> _MockIVSol:
        return self._d.get(id(iv), _MockIVSol(present=False))


def build_viz_handles(
        solution: "Solution",
        instance,
        *args,
        **kwargs
) -> tuple["_MockCPSol", dict]:
    """Convert a heuristic Solution to (mock_cp_sol, handles) for schedule_visualizer2.

    The returned pair can be passed directly to ``plot_schedule(mock_sol, handles)``
    without any docplex dependency.
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
        instance = Instance(S, L, K, orders_req, rt, p, {}, rt_ret=rt_ret)
    else:
        rt_ret = kwargs.pop('rt_ret', None)

    S, L, K, orders_req, rt, p, N = instance
    O = instance.O
    if rt_ret is None:
        rt_ret = instance.rt_ret
    sol = _MockCPSol()
    I_os: dict = {}
    I_os_lane: dict = {}
    F: dict = {}
    B: dict = {}
    R: dict = {}
    P: dict = {}
    C: dict = {}

    # Assign each BinEvent a unique per-(s, k) visit index so that the same physical
    # copy making multiple visits does not overwrite earlier F/B/R dict entries.
    # (copy_id is reused across visits; we need a collision-free key for the viz dicts.)
    visit_counters: dict[tuple[int, int], int] = defaultdict(int)
    visit_idx_lookup: dict[tuple[int, int, int], int] = {}  # (s, k, presence_start) -> visit_idx

    # Order windows: I_os and I_os_lane
    for o in O:
        assignment = solution.order_assignments.get(o)
        s_sel, ln_sel, t_start, t_end = assignment if assignment else (None, None, 0, 0)
        for s in S:
            iv = _MockIV()
            I_os[(o, s)] = iv
            sol._set(iv, present=(s == s_sel), start=t_start, end=t_end)
            for ln in L:
                iv2 = _MockIV()
                I_os_lane[(o, s, ln)] = iv2
                sol._set(iv2, present=(s == s_sel and ln == ln_sel),
                         start=t_start, end=t_end)

    # Bin fetch / presence / return — keyed by visit index, not physical copy_id
    for s, evts in solution.bin_events.items():
        for be in evts:
            k = be.sku
            e = visit_counters[(s, k)]
            visit_counters[(s, k)] += 1
            visit_idx_lookup[(s, k, be.presence_start)] = e
            iv_f = _MockIV()
            F[(s, k, e)] = iv_f
            sol._set(iv_f, present=True, start=be.fetch_start, end=be.fetch_end)
            iv_b = _MockIV()
            B[(s, k, e)] = iv_b
            sol._set(iv_b, present=True, start=be.presence_start, end=be.presence_end)
            iv_r = _MockIV()
            R[(s, k, e)] = iv_r
            sol._set(iv_r, present=True, start=be.return_start, end=be.return_end)

    # U[k] = max visits per SKU across stations (drives visualizer loop bounds)
    U: dict[int, int] = {k: 0 for k in K}
    for (s, k), count in visit_counters.items():
        U[k] = max(U[k], count)

    # Pick and consumption intervals
    for (o, s, k), (ps, pe) in solution.pick_events.items():
        matched_be = next(
            (be for be in solution.bin_events.get(s, [])
             if be.sku == k and be.presence_start <= ps < be.presence_end),
            None,
        )
        if matched_be is not None:
            copy_e = visit_idx_lookup.get((s, k, matched_be.presence_start))
            if copy_e is not None:
                iv_p = _MockIV()
                P[(o, s, k, copy_e)] = iv_p
                sol._set(iv_p, present=True, start=ps, end=pe)
        iv_c = _MockIV()
        C[(o, k, s)] = iv_c
        sol._set(iv_c, present=True, start=ps, end=pe)

    handles = {
        "I_os": I_os, "I_os_lane": I_os_lane,
        "C": C, "P": P, "F": F, "R": R, "B": B,
        "U": U, "orders_req": orders_req,
        "rt": rt, "rt_return": rt_ret, "p": p,
        "S": S, "L": L, "K": K, "O": O,
    }
    handles.update(kwargs)
    return sol, handles


# ============================================================
# Benchmarking API (mirrors solve_instance in CP model)
# ============================================================

def solve_heuristic_instance(config: dict, return_raw: bool = False):
    """Run one heuristic instance described by *config*.

    Accepts the same config keys as the CP model's ``solve_instance``.
    Returns a dict with the same top-level keys so benchmark scripts can
    treat both interchangeably.

    Returns::

        {
            'status':          'Feasible' | 'Infeasible',
            'solve_time':      float,          # wall-clock seconds
            'objective_value': float | None,   # makespan (None if infeasible)
            'num_vars':        0,              # not applicable for heuristic
            'progress':        [],             # heuristic is non-iterative
            'total_moves':     int,
        }
    """
    import sys
    import os
    import time

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
    sol = run_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N,
                  horizon=horizon, move_cap=move_cap, ALPHA=alpha, BETA=beta)
    elapsed = time.perf_counter() - t0

    res = {
        "status": "Feasible" if sol.feasible else "Infeasible",
        "solve_time": elapsed,
        "objective_value": float(sol.makespan) if sol.feasible else None,
        "num_vars": 0,
        "progress": [],
        "total_moves": sol.total_moves,
    }
    return (res, sol) if return_raw else res


# ============================================================
# [P1-9] Validator
# ============================================================

def validate_solution(
        solution: Solution,
        instance,
        *args,
        **kwargs
) -> list[str]:
    """Validate a heuristic solution. Returns list of violation strings (empty = valid)."""
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
        instance = Instance(S, L, K, orders_req, rt, p, N, rt_ret=rt_ret)
    else:
        horizon = kwargs.get('horizon', args[0] if len(args) > 0 else 10000)
        move_cap = kwargs.get('move_cap', args[1] if len(args) > 1 else None)
        rt_ret = kwargs.get('rt_ret', args[2] if len(args) > 2 else None)

    S, L, K, orders_req, rt, p, N = instance
    O = instance.O
    if rt_ret is None:
        rt_ret = instance.rt_ret
    violations: list[str] = []

    # 1. Every order assigned exactly once
    for o in O:
        if o not in solution.order_assignments:
            violations.append(f"Order {o} not assigned")

    # 2. Lane capacity: no two orders overlap on the same (s, ln)
    lane_usage: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for o, (s, ln, start, end) in solution.order_assignments.items():
        lane_usage[(s, ln)].append((start, end, o))
    for (s, ln), intervals in lane_usage.items():
        intervals.sort()
        for i in range(len(intervals) - 1):
            _, end_i, oi = intervals[i]
            start_j, _, oj = intervals[i + 1]
            if end_i > start_j:
                violations.append(
                    f"Lane overlap at S{s} L{ln}: order {oi} ends {end_i} > order {oj} starts {start_j}"
                )

    # 3. Pickface: at most one bin present at each station at any time
    for s in S:
        events = solution.bin_events.get(s, [])
        presence_intervals = [(ev.presence_start, ev.presence_end, ev.sku) for ev in events]
        presence_intervals.sort()
        for i in range(len(presence_intervals) - 1):
            _, end_i, k_i = presence_intervals[i]
            start_j, _, k_j = presence_intervals[i + 1]
            if end_i > start_j:
                violations.append(
                    f"Pickface overlap at S{s}: SKU {k_i} presence ends {end_i} > SKU {k_j} starts {start_j}"
                )

    # 4. Bin Block concurrency <= N[k] for each SKU
    for k in K:
        all_blocks: list[tuple[int, int]] = []
        for s in S:
            for ev in solution.bin_events.get(s, []):
                if ev.sku == k:
                    all_blocks.append((ev.fetch_start, ev.return_end))
        if len(all_blocks) <= N[k]:
            continue
        events_sweep: list[tuple[int, int]] = []
        for (bs, be) in all_blocks:
            events_sweep.append((bs, +1))
            events_sweep.append((be, -1))
        events_sweep.sort(key=lambda x: (x[0], x[1]))
        concurrent = 0
        for t, delta in events_sweep:
            concurrent += delta
            if concurrent > N[k]:
                violations.append(
                    f"SKU {k} Block concurrency {concurrent} > N[{k}]={N[k]} at t={t}"
                )
                break

    # 5. MoveCap not exceeded
    if move_cap is not None:
        move_events: list[tuple[int, int]] = []
        for s in S:
            for ev in solution.bin_events.get(s, []):
                move_events.append((ev.fetch_start, ev.fetch_end))
                move_events.append((ev.return_start, ev.return_end))
        sweep: list[tuple[int, int]] = []
        for (ms, me) in move_events:
            sweep.append((ms, +1))
            sweep.append((me, -1))
        sweep.sort(key=lambda x: (x[0], x[1]))
        concurrent = 0
        for t, delta in sweep:
            concurrent += delta
            if concurrent > move_cap:
                violations.append(
                    f"MoveCap violation: {concurrent} > {move_cap} at t={t}"
                )
                break

    # 6. Timing consistency for bin events
    for s in S:
        for ev in solution.bin_events.get(s, []):
            k = ev.sku
            if ev.fetch_end - ev.fetch_start != rt[k]:
                violations.append(
                    f"S{s} SKU {k}: fetch duration {ev.fetch_end - ev.fetch_start} != rt[{k}]={rt[k]}"
                )
            if ev.return_end - ev.return_start != rt_ret[k]:
                violations.append(
                    f"S{s} SKU {k}: return duration {ev.return_end - ev.return_start} != rt_ret[{k}]={rt_ret[k]}"
                )
            if ev.presence_start != ev.fetch_end:
                violations.append(
                    f"S{s} SKU {k}: presence_start {ev.presence_start} != fetch_end {ev.fetch_end}"
                )
            if ev.presence_end != ev.return_start:
                violations.append(
                    f"S{s} SKU {k}: presence_end {ev.presence_end} != return_start {ev.return_start}"
                )

    # 7. Horizon
    for o, (s, ln, start, end) in solution.order_assignments.items():
        if end > horizon:
            violations.append(f"Order {o} end {end} exceeds horizon {horizon}")

    # 8. Pick completeness
    for o, (s, ln, start, end) in solution.order_assignments.items():
        for k in orders_req[o]:
            covered = False
            for ev in solution.bin_events.get(s, []):
                if ev.sku == k and o in ev.orders_served:
                    covered = True
                    break
            if not covered:
                violations.append(f"Order {o} missing pick for SKU {k} at station {s}")

    return violations


# ============================================================
# Standalone entry point (mirrors CP model CLI)
# ============================================================

def main():
    """CLI entry point for running the SGC heuristic standalone."""
    import argparse
    import time
    import sys
    import os

    # Add project dir to path so we can import from the CP model
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from datagen import generate_data

    ap = argparse.ArgumentParser(description="SGC Heuristic for AutoStore Task B (v5)")
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
    ap.add_argument("--no_vis", action="store_true", help="Skip HTML schedule visualisation")
    args = ap.parse_args()
    

    print("Generating data...")
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

    print(f"Stations={len(S)}, Lanes={len(L)}, SKUs={len(K)}, Orders={len(O)}")
    # print(f"N (bins per SKU): { {k: N[k] for k in K} }")
    print("Orders:")
    for o in O:
        print(f"  o={o}: {orders_req[o]}")

    print(f"\nRunning SGC heuristic (alpha={args.alpha}, beta={args.beta})...")
    t0 = time.perf_counter()
    sol = run_sgc(
        instance,
        horizon=args.horizon, move_cap=args.movecap,
        ALPHA=args.alpha, BETA=args.beta,
    )
    elapsed = time.perf_counter() - t0

    print(f"\n=== SGC Result ===")
    print(f"Feasible: {sol.feasible}")
    print(f"Makespan: {sol.makespan}")
    print(f"Total bin events (moves/2): {sol.total_moves // 2}")
    print(f"Time: {elapsed:.4f}s")

    print(f"\nOrder assignments:")
    for o in O:
        if o in sol.order_assignments:
            s, ln, start, end = sol.order_assignments[o]
            print(f"  Order {o:>2} -> S{s} L{ln} [{start}, {end})")

    print(f"\nStation timelines:")
    for s in S:
        events = sol.bin_events.get(s, [])
        if not events:
            print(f"  Station {s}: (no events)")
            continue
        print(f"  Station {s}:")
        for ev in sorted(events, key=lambda e: e.fetch_start):
            print(f"    SKU {ev.sku} copy={ev.copy_id}: "
                  f"F[{ev.fetch_start},{ev.fetch_end}) "
                  f"B[{ev.presence_start},{ev.presence_end}) "
                  f"R[{ev.return_start},{ev.return_end}) "
                  f"orders={ev.orders_served}")

    violations = validate_solution(
        sol, instance, horizon=args.horizon, move_cap=args.movecap
    )
    if violations:
        print(f"\n=== VALIDATION FAILED ({len(violations)} violations) ===")
        for v in violations:
            print(f"  - {v}")
    else:
        print("\n=== Validation PASSED ===")

    if not args.no_vis:
        try:
            from schedule_visualizer import plot_schedule, write_html
            mock_sol, handles = build_viz_handles(sol, instance)
            fig = plot_schedule(mock_sol, handles, show=True)
            html_file = "./autostore_heuristic_solution.html"
            write_html(fig, html_file)
            print(f"\nWrote visualisation to {html_file}")
        except ImportError:
            print("\nschedule_visualizer2 not found. Skipping visualisation.")


if __name__ == "__main__":
    main()
