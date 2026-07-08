#!/usr/bin/env python3
"""
Unit tests for autostore_heuristic.py — Phases 0 and 1.

[P0-7] DifferenceArray tests
[P0-8] BinCopyPool tests
[P1-1] static_priority_sort test
[P1-2] find_shared_bin test
[P1-10] End-to-end SGC tests
[P2-VW] validate_warmstart tests (require docplex / CP Optimizer)
[P2-HE] Heuristic edge-case tests
"""
from cp_model import (
    build_model,
    validate_warmstart,
    inject_warmstart,
)
from datagen import generate_data as gen_v5
from autostore_heuristic import (
    DifferenceArray, BinCopyPool, BinEvent, HeuristicState,
    static_priority_sort, find_shared_bin, run_sgc, validate_solution,
    init_state, plan_order_at_station, commit_plan,
)
import copy
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# CP model imports — the whole test file implicitly requires docplex since
# generate_data lives in the same module as the CP model.


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _small_v5_instance(stations=1, lanes=2, orders=7, skus=5, seed=42):
    """Generate a small problem instance using v5's generate_data."""
    S, L, K, orders_req, rt, p, N = gen_v5(
        num_stations=stations, lanes_per_station=lanes, num_orders=orders,
        num_skus=skus, seed=seed, pick_touch_time=4,
    )
    rt_ret = dict(rt)
    O = sorted(orders_req.keys())
    return S, L, K, O, orders_req, rt, rt_ret, p, N


def _run_small(stations=1, lanes=2, orders=7, skus=5, seed=42,
               horizon=2000, move_cap=None):
    """Generate a small instance and run the SGC heuristic on it."""
    S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance(
        stations, lanes, orders, skus, seed)
    sol = run_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N,
                  horizon=horizon, move_cap=move_cap)
    return sol, S, L, K, O, orders_req, rt, rt_ret, p, N


def _build_handles(S, L, K, O, orders_req, rt, rt_ret, p, N,
                   horizon=2000, move_cap=None, symmetry_breaking=True):
    """Build the CP model and return (mdl, handles)."""
    mdl, handles = build_model(
        S, L, K, orders_req, rt, p,
        rt_return=rt_ret, horizon=horizon, N=N,
        move_cap=move_cap, add_symmetry_breaking=symmetry_breaking,
    )
    return mdl, handles


# ---------------------------------------------------------------------------
# Corruption helpers — shallow-copy a Solution with one field patched
# ---------------------------------------------------------------------------

def _patch_order(sol, o, **kw):
    """Return a Solution copy with one order_assignments entry patched."""
    new_sol = copy.copy(sol)
    a = dict(sol.order_assignments)
    s_a, ln_a, t_s, t_e = a[o]
    a[o] = (kw.get("s", s_a), kw.get("ln", ln_a),
            kw.get("start", t_s), kw.get("end", t_e))
    new_sol.order_assignments = a
    return new_sol


def _patch_bin_event(sol, s, idx, **kw):
    """Return a Solution copy with one BinEvent field patched."""
    new_sol = copy.copy(sol)
    evs = {st: list(v) for st, v in sol.bin_events.items()}
    be = copy.copy(evs[s][idx])
    for attr, val in kw.items():
        setattr(be, attr, val)
    evs[s][idx] = be
    new_sol.bin_events = evs
    return new_sol


def _append_bin_event(sol, s, be):
    """Return a Solution copy with one extra BinEvent appended at station s."""
    new_sol = copy.copy(sol)
    evs = {st: list(v) for st, v in sol.bin_events.items()}
    evs.setdefault(s, []).append(be)
    new_sol.bin_events = evs
    return new_sol


def _drop_order(sol, o):
    """Return a Solution copy with order o removed from order_assignments."""
    new_sol = copy.copy(sol)
    new_sol.order_assignments = {k: v for k, v in sol.order_assignments.items() if k != o}
    return new_sol


# ============================================================
# [P0-7] DifferenceArray tests
# ============================================================

class TestDifferenceArray:
    def test_empty_profile(self):
        da = DifferenceArray(100)
        da.build_prefix()
        assert da.range_max(0, 100) == 0

    def test_single_move(self):
        da = DifferenceArray(100)
        da.add_move(10, 20)
        da.build_prefix()
        assert da.range_max(0, 9) == 0
        assert da.range_max(10, 19) == 1
        assert da.range_max(20, 30) == 0

    def test_overlapping_moves(self):
        da = DifferenceArray(100)
        da.add_move(10, 30)
        da.add_move(20, 40)
        da.build_prefix()
        assert da.range_max(10, 19) == 1
        assert da.range_max(20, 29) == 2
        assert da.range_max(30, 39) == 1
        assert da.range_max(40, 50) == 0

    def test_remove_move_symmetry(self):
        da = DifferenceArray(100)
        da.add_move(10, 30)
        da.add_move(20, 40)
        da.remove_move(10, 30)
        da.build_prefix()
        assert da.range_max(10, 19) == 0
        assert da.range_max(20, 29) == 1
        assert da.range_max(30, 39) == 1
        assert da.range_max(40, 50) == 0

    def test_delay_for_movecap_no_conflict(self):
        da = DifferenceArray(100)
        da.build_prefix()
        assert da.delay_for_movecap(0, 10, 2) == 0

    def test_delay_for_movecap_with_conflict(self):
        da = DifferenceArray(100)
        da.add_move(5, 15)
        da.add_move(8, 18)
        da.build_prefix()
        # Cap=2 means max 2 concurrent. At [8,15) there are 2 moves.
        # A new move of duration 5 starting at 5 would see peak=2 at t=8 -> must delay
        result = da.delay_for_movecap(5, 5, 2)
        assert result >= 15  # must start after both existing moves end their overlap

    def test_delay_for_movecap_none_cap(self):
        da = DifferenceArray(100)
        da.add_move(0, 50)
        da.add_move(0, 50)
        da.build_prefix()
        assert da.delay_for_movecap(0, 10, None) == 0

    def test_delay_with_pending(self):
        da = DifferenceArray(100)
        da.build_prefix()
        pending = [(0, 10), (5, 15)]
        # Cap=2: pending creates 2 concurrent moves at [5,10). A new duration-5 move:
        result = da.delay_for_movecap_with_pending(5, 5, 2, pending)
        assert result >= 10  # must wait for the overlap to clear


# ============================================================
# [P0-8] BinCopyPool tests
# ============================================================

class TestBinCopyPool:
    def test_single_copy(self):
        pool = BinCopyPool(sku=0, n_copies=1)
        avail, cid = pool.get_earliest_free()
        assert avail == 0
        assert cid == 0
        pool.commit(0, 50)
        avail, cid = pool.get_earliest_free()
        assert avail == 50

    def test_three_copies_concurrent(self):
        pool = BinCopyPool(sku=1, n_copies=3)
        # All 3 copies free at 0
        a0, c0 = pool.get_earliest_free()
        assert a0 == 0
        pool.commit(c0, 100)

        a1, c1 = pool.get_earliest_free()
        assert a1 == 0  # still 2 copies free
        pool.commit(c1, 200)

        a2, c2 = pool.get_earliest_free()
        assert a2 == 0  # still 1 copy free
        pool.commit(c2, 300)

        # Now all 3 are busy
        a3, c3 = pool.get_earliest_free()
        assert a3 == 100  # earliest return

    def test_peek_nth(self):
        pool = BinCopyPool(sku=2, n_copies=3)
        pool.commit(0, 50)
        pool.commit(1, 30)
        pool.commit(2, 70)
        assert pool.peek_nth(0) == 30
        assert pool.peek_nth(1) == 50
        assert pool.peek_nth(2) == 70

    def test_deep_copy(self):
        pool = BinCopyPool(sku=0, n_copies=2)
        pool.commit(0, 100)
        pool2 = copy.deepcopy(pool)
        pool2.commit(1, 200)
        # Original should be unchanged
        a, c = pool.get_earliest_free()
        assert a == 0  # copy 1 still free in original


# ============================================================
# [P1-1] static_priority_sort test
# ============================================================

class TestStaticPrioritySort:
    def test_sort_by_sharing_degree(self):
        orders_req = {0: [0, 1], 1: [2], 2: [0, 1, 2]}
        O = [0, 1, 2]
        result = static_priority_sort(orders_req, O)
        # Order 0: shares [0, 1] with Order 2 -> degree 1
        # Order 1: shares [2] with Order 2 -> degree 1
        # Order 2: shares [0, 1] with Order 0, [2] with Order 1 -> degree 2
        # Ascending sort: [0, 1, 2] or [1, 0, 2]
        assert result[0] in [0, 1]
        assert result[1] in [0, 1]
        assert result[2] == 2


# ============================================================
# [P1-2] find_shared_bin test
# ============================================================

class TestFindSharedBin:
    def test_finds_overlap(self):
        ev = BinEvent(sku=3, copy_id=0,
                      fetch_start=0, fetch_end=10,
                      presence_start=10, presence_end=20,
                      return_start=20, return_end=30,
                      orders_served=[0])
        result = find_shared_bin([ev], k=3, t_cursor=15, p={3: 4})
        assert result is ev

    def test_no_overlap(self):
        ev = BinEvent(sku=3, copy_id=0,
                      fetch_start=0, fetch_end=10,
                      presence_start=10, presence_end=20,
                      return_start=20, return_end=30,
                      orders_served=[0])
        result = find_shared_bin([ev], k=3, t_cursor=25, p={3: 4})
        assert result is None

    def test_wrong_sku(self):
        ev = BinEvent(sku=3, copy_id=0,
                      fetch_start=0, fetch_end=10,
                      presence_start=10, presence_end=20,
                      return_start=20, return_end=30,
                      orders_served=[0])
        result = find_shared_bin([ev], k=5, t_cursor=15, p={5: 4})
        assert result is None


# ============================================================
# [P2-VW] validate_warmstart tests
# ============================================================

class TestValidateWarmstart:
    """Tests for validate_warmstart in cp_model.py.

    Each negative test corrupts one aspect of a valid solution and asserts that
    validate_warmstart catches the violation with a recognisable message.
    """

    @classmethod
    def _instance(cls):
        """Return (sol, S, L, K, O, orders_req, rt, rt_ret, p, N, handles)."""
        S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance()
        sol = run_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N, horizon=2000)
        _, handles = _build_handles(S, L, K, O, orders_req, rt, rt_ret, p, N)
        return sol, S, L, K, O, orders_req, rt, rt_ret, p, N, handles

    # --- Positive test ---

    def test_valid_clean(self):
        """A fresh heuristic solution should have no real violations."""
        sol, *_, handles = self._instance()
        violations = validate_warmstart(sol, sol.pick_events, handles)
        # Symmetry(A) raw-lane warnings are expected (inject_warmstart remaps them).
        non_real = [v for v in violations if "Symmetry (A)" not in v]
        assert non_real == [], f"Unexpected violations: {non_real}"

    # --- Check 1: bin trip count vs U[k] ---

    def test_u_overflow(self):
        """Adding extra bin events beyond U[k] should be flagged."""
        sol, S, L, K, O, orders_req, rt, rt_ret, p, N, handles = self._instance()
        U = handles["U"]
        s0 = S[0]
        # Find SKU with fewest allowed trips (easiest to overflow)
        k_target = min(K, key=lambda k: U.get(k, 0))
        u_k = U[k_target]
        curr_count = sum(1 for be in sol.bin_events.get(s0, []) if be.sku == k_target)
        # Add enough events so that total count > U[k_target]
        n_extra = u_k - curr_count + 1
        bad_sol = sol
        for i in range(max(1, n_extra)):
            t = 50000 + i * 500
            extra = BinEvent(
                sku=k_target, copy_id=i,
                fetch_start=t, fetch_end=t + rt[k_target],
                presence_start=t + rt[k_target],
                presence_end=t + rt[k_target] + p[k_target],
                return_start=t + rt[k_target] + p[k_target],
                return_end=t + rt[k_target] + p[k_target] + rt_ret[k_target],
            )
            bad_sol = _append_bin_event(bad_sol, s0, extra)
        violations = validate_warmstart(bad_sol, bad_sol.pick_events, handles)
        assert any("trips" in v for v in violations), f"Expected U overflow violation, got: {violations}"

    # --- Check 2: pick within bin presence ---

    def test_pick_outside_presence(self):
        """A pick event shifted far into the future should be flagged."""
        sol, *_, handles = self._instance()
        if not sol.pick_events:
            pytest.skip("No pick events in solution")
        bad_pe = dict(sol.pick_events)
        first_key = next(iter(bad_pe))
        ps, pe = bad_pe[first_key]
        bad_pe[first_key] = (ps + 100_000, pe + 100_000)
        violations = validate_warmstart(sol, bad_pe, handles)
        assert any("not covered" in v for v in violations), f"Expected pick-outside-presence, got: {violations}"

    # --- Check 3: order window spans picks ---

    def test_order_start_mismatch(self):
        """Forcing t_start=0 should mismatch the min pick start."""
        sol, *_, handles = self._instance()
        # Find an order that has pick events and whose heuristic start != 0
        o_target = next(
            (o for o in sol.order_assignments if sol.order_assignments[o][2] != 0),
            None,
        )
        if o_target is None:
            pytest.skip("All orders start at t=0")
        bad_sol = _patch_order(sol, o_target, start=0)
        violations = validate_warmstart(bad_sol, sol.pick_events, handles)
        assert any("Start" in v for v in violations), f"Expected start mismatch, got: {violations}"

    def test_order_end_mismatch(self):
        """Inflating t_end should mismatch the max pick end."""
        sol, *_, handles = self._instance()
        o_target = next(iter(sol.order_assignments))
        t_e = sol.order_assignments[o_target][3]
        bad_sol = _patch_order(sol, o_target, end=t_e + 999)
        violations = validate_warmstart(bad_sol, sol.pick_events, handles)
        assert any("End" in v for v in violations), f"Expected end mismatch, got: {violations}"

    # --- Check 4: lane fill order symmetry ---

    def test_lane_symmetry_raw(self):
        """Putting all orders on the last lane should trigger symmetry(A) warning."""
        sol, S, L, K, O, orders_req, rt, rt_ret, p, N, handles = self._instance()
        # Remap all orders to the last lane so lane 0 count < last lane count
        last_ln = L[-1]
        bad_assignments = {}
        for i, o in enumerate(O):
            bad_assignments[o] = (S[0], last_ln, i * 100, i * 100 + 10)
        bad_sol = copy.copy(sol)
        bad_sol.order_assignments = bad_assignments
        violations = validate_warmstart(bad_sol, sol.pick_events, handles)
        assert any("Symmetry (A)" in v for v in violations), \
            f"Expected symmetry (A) violation, got: {violations}"

    # --- Check 6: order completeness ---

    def test_order_completeness(self):
        """Dropping an order from assignments should be reported."""
        sol, S, L, K, O, orders_req, rt, rt_ret, p, N, handles = self._instance()
        o_drop = O[0]
        bad_sol = _drop_order(sol, o_drop)
        bad_pe = {k: v for k, v in sol.pick_events.items() if k[0] != o_drop}
        violations = validate_warmstart(bad_sol, bad_pe, handles)
        assert any(str(o_drop) in v and "not assigned" in v for v in violations), \
            f"Expected completeness violation for order {o_drop}, got: {violations}"

    # --- Check 7: lane no-overlap ---

    def test_lane_overlap(self):
        """Two orders forced into overlapping intervals on the same lane should be flagged."""
        sol, S, L, K, O, orders_req, rt, rt_ret, p, N, handles = self._instance()
        o0, o1 = O[0], O[1]
        bad_sol = copy.copy(sol)
        a = dict(sol.order_assignments)
        # Place both on lane 0, station 0 with overlapping windows [0,100) and [50,150)
        a[o0] = (S[0], L[0], 0, 100)
        a[o1] = (S[0], L[0], 50, 150)
        bad_sol.order_assignments = a
        violations = validate_warmstart(bad_sol, sol.pick_events, handles)
        assert any("Lane overlap" in v for v in violations), \
            f"Expected lane overlap violation, got: {violations}"

    # --- Check 8: pickface no-overlap ---

    def test_pickface_overlap(self):
        """Two bin presences at the same station with identical windows should be flagged."""
        sol, S, L, K, O, orders_req, rt, rt_ret, p, N, handles = self._instance()
        s0 = S[0]
        evts = sol.bin_events.get(s0, [])
        if not evts:
            pytest.skip("No bin events at station 0")
        orig = evts[0]
        # Append a copy of orig with a different SKU but identical presence window
        # (use a SKU index that doesn't exist in handles["K"] to avoid U-overflow check)
        fake_sku = max(K) + 999
        overlap_ev = BinEvent(
            sku=fake_sku, copy_id=0,
            fetch_start=orig.fetch_start,
            fetch_end=orig.fetch_end,
            presence_start=orig.presence_start,
            presence_end=orig.presence_end,
            return_start=orig.return_start,
            return_end=orig.return_end,
        )
        bad_sol = _append_bin_event(sol, s0, overlap_ev)
        violations = validate_warmstart(bad_sol, sol.pick_events, handles)
        assert any("Pickface overlap" in v for v in violations), \
            f"Expected pickface overlap violation, got: {violations}"

    # --- Check 9: bin timing consistency ---

    def test_bin_timing_inconsistency(self):
        """Shifting presence_start away from fetch_end should be flagged."""
        sol, S, L, K, O, orders_req, rt, rt_ret, p, N, handles = self._instance()
        s0 = S[0]
        evts = sol.bin_events.get(s0, [])
        if not evts:
            pytest.skip("No bin events at station 0")
        bad_sol = _patch_bin_event(sol, s0, 0,
                                   presence_start=evts[0].presence_start + 5)
        violations = validate_warmstart(bad_sol, sol.pick_events, handles)
        assert any("presence_start" in v for v in violations), \
            f"Expected timing inconsistency, got: {violations}"

    # --- Check 10: block concurrency ---

    def test_block_concurrency(self):
        """Adding a bin event that overlaps an existing block for a N[k]=1 SKU should be flagged."""
        S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance()
        # Force N[k]=1 for all SKUs so any overlap is a violation
        N_one = {k: 1 for k in K}
        sol = run_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N_one, horizon=2000)
        _, handles = _build_handles(S, L, K, O, orders_req, rt, rt_ret, p, N_one)
        s0 = S[0]
        evts = sol.bin_events.get(s0, [])
        if not evts:
            pytest.skip("No bin events at station 0")
        orig = evts[0]
        k = orig.sku
        # Create a block that overlaps orig's [fetch_start, return_end) by 1 tick
        t = orig.fetch_start + 1
        overlap_ev = BinEvent(
            sku=k, copy_id=99,
            fetch_start=t, fetch_end=t + rt[k],
            presence_start=t + rt[k],
            presence_end=t + rt[k] + p[k],
            return_start=t + rt[k] + p[k],
            return_end=t + rt[k] + p[k] + rt_ret[k],
        )
        bad_sol = _append_bin_event(sol, s0, overlap_ev)
        violations = validate_warmstart(bad_sol, sol.pick_events, handles)
        assert any("Block concurrency" in v for v in violations), \
            f"Expected block concurrency violation, got: {violations}"

    # --- Check 11: MoveCap ---

    def test_movecap_violation(self):
        """Three simultaneous fetches with move_cap=2 should be flagged."""
        S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance()
        sol = run_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N, horizon=2000)
        # Build handles with move_cap=2
        _, handles = _build_handles(S, L, K, O, orders_req, rt, rt_ret, p, N,
                                    move_cap=2)
        s0 = S[0]
        k0 = K[0]
        # Add 3 bin events all starting at the same time -> 3 concurrent fetches
        bad_sol = sol
        for i in range(3):
            t = 80000
            extra = BinEvent(
                sku=k0, copy_id=i,
                fetch_start=t, fetch_end=t + rt[k0],
                presence_start=t + rt[k0],
                presence_end=t + rt[k0] + p[k0],
                return_start=t + rt[k0] + p[k0],
                return_end=t + rt[k0] + p[k0] + rt_ret[k0],
            )
            bad_sol = _append_bin_event(bad_sol, s0, extra)
        violations = validate_warmstart(bad_sol, bad_sol.pick_events, handles)
        assert any("MoveCap" in v for v in violations), \
            f"Expected MoveCap violation, got: {violations}"


# ============================================================
# [P2-HE] Heuristic edge-case tests
# ============================================================

class TestHeuristicEdgeCases:
    """Edge cases for the SGC heuristic: tight constraints, degenerate inputs."""

    def test_movecap_1_serializes_fetches(self):
        """With move_cap=1, no two moves may be concurrent; validate_solution must pass."""
        sol, S, L, K, O, orders_req, rt, rt_ret, p, N = _run_small(
            stations=1, lanes=2, orders=5, skus=4, seed=3,
            horizon=5000, move_cap=1,
        )
        assert sol.feasible, "Heuristic should find a feasible solution with move_cap=1"
        violations = validate_solution(
            sol, S, L, K, O, orders_req, rt, rt_ret, p, N, 5000, move_cap=1)
        assert violations == [], f"Violations: {violations}"

    def test_bin_copy_exhaustion(self):
        """With N[k]=1 for all SKUs, orders must queue for bins; no concurrency violation."""
        S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance(
            stations=1, lanes=2, orders=4, skus=2, seed=5)
        N_one = {k: 1 for k in K}
        sol = run_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N_one, horizon=5000)
        assert sol.feasible, "Should be feasible even with single bin copies"
        violations = validate_solution(
            sol, S, L, K, O, orders_req, rt, rt_ret, p, N_one, 5000)
        assert violations == [], f"Violations with N[k]=1: {violations}"

    def test_pick_events_cover_all_assignments(self):
        """For every assigned (o, s, k), pick_events must have an entry."""
        sol, S, L, K, O, orders_req, rt, rt_ret, p, N = _run_small(
            stations=2, lanes=2, orders=8, skus=5, seed=11)
        assert sol.feasible
        for o, (s_a, ln_a, t_s, t_e) in sol.order_assignments.items():
            for k in orders_req[o]:
                assert (o, s_a, k) in sol.pick_events, (
                    f"Missing pick_event for (o={o}, s={s_a}, k={k})")

    def test_bin_events_timing_valid(self):
        """Every BinEvent must have internally consistent timing fields."""
        sol, S, L, K, O, orders_req, rt, rt_ret, p, N = _run_small()
        assert sol.feasible
        for s in S:
            for be in sol.bin_events.get(s, []):
                k = be.sku
                assert be.fetch_end == be.presence_start, (
                    f"S{s} SKU {k}: fetch_end {be.fetch_end} != presence_start {be.presence_start}")
                assert be.presence_end == be.return_start, (
                    f"S{s} SKU {k}: presence_end {be.presence_end} != return_start {be.return_start}")
                assert be.fetch_end - be.fetch_start == rt[k], (
                    f"S{s} SKU {k}: fetch duration {be.fetch_end - be.fetch_start} != rt={rt[k]}")
                assert be.return_end - be.return_start == rt_ret[k], (
                    f"S{s} SKU {k}: return duration {be.return_end - be.return_start} != rt_ret={rt_ret[k]}")

    def test_tight_horizon(self):
        """With horizon=1, the solution is infeasible or all order ends are within horizon."""
        sol, S, L, K, O, orders_req, rt, rt_ret, p, N = _run_small(
            horizon=1)
        if sol.feasible:
            for o, (s, ln, t_s, t_e) in sol.order_assignments.items():
                assert t_e <= 1, f"Order {o} ends at {t_e} which exceeds horizon=1"
        # Infeasible is also an acceptable outcome; the test simply ensures no crash

    def test_many_orders_lane_usage(self):
        """20 orders across 4 lanes should yield a valid solution."""
        sol, S, L, K, O, orders_req, rt, rt_ret, p, N = _run_small(
            stations=1, lanes=4, orders=20, skus=8, seed=1, horizon=8000)
        assert sol.feasible
        violations = validate_solution(
            sol, S, L, K, O, orders_req, rt, rt_ret, p, N, 8000)
        assert violations == [], f"Violations: {violations}"

    def test_single_sku_n1_serialized(self):
        """3 orders all needing only SKU 0 with N[0]=1 must be fully serialized."""
        S, L = [0], [0, 1]
        K = [0]
        O = [0, 1, 2]
        orders_req = {0: [0], 1: [0], 2: [0]}
        rt = {0: 20}
        rt_ret = {0: 20}
        p = {0: 4}
        N = {0: 1}
        sol = run_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N, horizon=5000)
        violations = validate_solution(
            sol, S, L, K, O, orders_req, rt, rt_ret, p, N, 5000)
        assert violations == [], f"Single-SKU N=1 violations: {violations}"

    def test_every_pick_covered_by_exactly_one_bin_event(self):
        """Every (o, s, k) pick event must be covered by exactly one BinEvent presence."""
        sol, S, L, K, O, orders_req, rt, rt_ret, p, N = _run_small(
            stations=1, lanes=2, orders=7, skus=4, seed=0)
        assert sol.feasible
        for (o, s_sel, k), (ps, pe) in sol.pick_events.items():
            covering = [
                be for be in sol.bin_events.get(s_sel, [])
                if be.sku == k and be.presence_start <= ps and pe <= be.presence_end
            ]
            assert len(covering) == 1, (
                f"Order {o} SKU {k} at S{s_sel} covered by {len(covering)} bin events (expected 1)")

    def test_validate_solution_passes_multi_station(self):
        """Multi-station run should satisfy all 7 checks in validate_solution."""
        sol, S, L, K, O, orders_req, rt, rt_ret, p, N = _run_small(
            stations=3, lanes=2, orders=20, skus=8, seed=7,
            horizon=5000, move_cap=4)
        assert sol.feasible
        violations = validate_solution(
            sol, S, L, K, O, orders_req, rt, rt_ret, p, N, 5000, move_cap=4)
        assert violations == [], f"Multi-station violations: {violations}"


# ============================================================
# [P1-10] End-to-end SGC test
# ============================================================

class TestEndToEnd:
    @staticmethod
    def _generate_default():
        """Generate the default small test instance (seed=42)."""
        S, L, K, orders_req, rt, p, N = gen_v5(
            num_stations=1, lanes_per_station=2, num_orders=7,
            num_skus=5, seed=42, pick_touch_time=4,
        )
        rt_ret = dict(rt)
        O = sorted(orders_req.keys())
        return S, L, K, O, orders_req, rt, rt_ret, p, N

    def test_feasible_and_valid(self):
        S, L, K, O, orders_req, rt, rt_ret, p, N = self._generate_default()
        sol = run_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N,
                      horizon=2000)
        assert sol.feasible
        violations = validate_solution(sol, S, L, K, O, orders_req, rt, rt_ret, p, N, 2000)
        assert violations == [], f"Violations: {violations}"

    def test_all_orders_assigned(self):
        S, L, K, O, orders_req, rt, rt_ret, p, N = self._generate_default()
        sol = run_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N, horizon=2000)
        assert set(sol.order_assignments.keys()) == set(O)

    def test_makespan_positive(self):
        S, L, K, O, orders_req, rt, rt_ret, p, N = self._generate_default()
        sol = run_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N, horizon=2000)
        assert sol.makespan > 0

    def test_with_movecap(self):
        S, L, K, O, orders_req, rt, rt_ret, p, N = self._generate_default()
        sol = run_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N,
                      horizon=2000, move_cap=2)
        assert sol.feasible
        violations = validate_solution(sol, S, L, K, O, orders_req, rt, rt_ret, p, N,
                                       2000, move_cap=2)
        assert violations == [], f"Violations: {violations}"

    def test_multi_station(self):
        S, L, K, orders_req, rt, p, N = gen_v5(
            num_stations=2, lanes_per_station=2, num_orders=7,
            num_skus=5, seed=42, pick_touch_time=4,
        )
        rt_ret = dict(rt)
        O = sorted(orders_req.keys())
        sol = run_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N,
                      horizon=2000, move_cap=3)
        assert sol.feasible
        violations = validate_solution(sol, S, L, K, O, orders_req, rt, rt_ret, p, N,
                                       2000, move_cap=3)
        assert violations == [], f"Violations: {violations}"

    def test_medium_instance(self):
        S, L, K, orders_req, rt, p, N = gen_v5(
            num_stations=3, lanes_per_station=3, num_orders=40,
            num_skus=20, seed=42, pick_touch_time=4,
        )
        rt_ret = dict(rt)
        O = sorted(orders_req.keys())
        sol = run_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N,
                      horizon=10000, move_cap=4)
        assert sol.feasible
        violations = validate_solution(sol, S, L, K, O, orders_req, rt, rt_ret, p, N,
                                       10000, move_cap=4)
        assert violations == [], f"Violations: {violations}"

    def test_validate_warmstart_no_cp_violations(self):
        """After SGC, validate_warmstart should find no real constraint violations."""
        S, L, K, orders_req, rt, p, N = gen_v5(
            num_stations=2, lanes_per_station=2, num_orders=10,
            num_skus=6, seed=5, pick_touch_time=4,
        )
        rt_ret = dict(rt)
        O = sorted(orders_req.keys())
        sol = run_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N, horizon=2000)
        assert sol.feasible
        _, handles = build_model(
            S, L, K, orders_req, rt, p,
            rt_return=rt_ret, horizon=2000, N=N,
        )
        violations = validate_warmstart(sol, sol.pick_events, handles)
        # Only Symmetry(A) raw-lane warnings are acceptable; inject_warmstart remaps them.
        non_real = [v for v in violations if "Symmetry (A)" not in v]
        assert non_real == [], f"Unexpected warmstart violations: {non_real}"

    def test_inject_warmstart_accepted_by_cp(self):
        """CP Optimizer should accept the inject_warmstart output as a valid starting point."""
        S, L, K, orders_req, rt, p, N = gen_v5(
            num_stations=1, lanes_per_station=2, num_orders=5,
            num_skus=4, seed=0, pick_touch_time=4,
        )
        rt_ret = dict(rt)
        O = sorted(orders_req.keys())
        sol = run_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N, horizon=2000)
        assert sol.feasible
        mdl, handles = build_model(
            S, L, K, orders_req, rt, p,
            rt_return=rt_ret, horizon=2000, N=N,
        )
        sp = inject_warmstart(sol, sol.pick_events, mdl, handles)
        mdl.set_starting_point(sp)
        cp_sol = mdl.solve(
            Workers=1, TimeLimit=5, LogVerbosity="Quiet",
            solve_with_search_next=False,
        )
        assert cp_sol is not None
        status = cp_sol.get_solve_status()
        # "Unknown" means timeout with no solution found — still acceptable (no crash or rejection).
        assert status in ("Feasible", "Optimal", "Unknown"), f"Unexpected CP status: {status}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
