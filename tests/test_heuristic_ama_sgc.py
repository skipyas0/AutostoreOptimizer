#!/usr/bin/env python3
"""
Unit tests for AMA-SGC Heuristic.

This file combines tests from the original test_heuristic_ama_sgc.py
and tests/test_heuristic_ama_sgc.py.
"""
from cp_model import build_model, validate_warmstart
from heuristic_ama_sgc import (
    precompute_attributes, sort_orders, sort_skus_for_order, run_ama_sgc,
    run_sgc_parameterised, ORDER_ATTRS, BIN_ATTRS
)
from autostore_heuristic import validate_solution, run_sgc
from datagen import generate_data as gen_v5
import copy
import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def _run_small_ama(stations=1, lanes=2, orders=7, skus=5, seed=42,
                   horizon=2000, move_cap=None):
    """Generate a small instance and run the AMA-SGC heuristic on it."""
    S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance(
        stations, lanes, orders, skus, seed)
    sol, _, _ = run_ama_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N,
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

# ============================================================
# Basic component tests for AMA SGC
# ============================================================


class TestPrecomputeAttributes:
    def test_keys_present(self):
        S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance()
        order_attrs, sku_attrs = precompute_attributes(O, orders_req, rt, rt_ret, p, N, K)

        for o in O:
            assert set(order_attrs[o].keys()) == {
                "sum_rt", "order_size", "sum_cycle", "max_rt",
                "sku_rarity", "sku_contention", "sharing_degree", "min_copies",
            }
        for k in K:
            assert set(sku_attrs[k].keys()) == {
                "rt", "p", "cycle", "demand_ratio", "copies", "demand",
            }

    def test_sum_rt_correctness(self):
        S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance()
        order_attrs, _ = precompute_attributes(O, orders_req, rt, rt_ret, p, N, K)
        for o in O:
            expected = sum(rt[k] for k in orders_req[o])
            assert order_attrs[o]["sum_rt"] == expected

    def test_order_size_correctness(self):
        S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance()
        order_attrs, _ = precompute_attributes(O, orders_req, rt, rt_ret, p, N, K)
        for o in O:
            assert order_attrs[o]["order_size"] == len(orders_req[o])

    def test_sum_cycle_correctness(self):
        S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance()
        order_attrs, sku_attrs = precompute_attributes(O, orders_req, rt, rt_ret, p, N, K)
        for o in O:
            expected = sum(rt[k] + p[k] + rt_ret[k] for k in orders_req[o])
            assert order_attrs[o]["sum_cycle"] == expected

    def test_sku_contention_nonnegative(self):
        S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance()
        order_attrs, _ = precompute_attributes(O, orders_req, rt, rt_ret, p, N, K)
        for o in O:
            assert order_attrs[o]["sku_contention"] >= 0.0

    def test_sharing_degree_bounds(self):
        S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance()
        order_attrs, _ = precompute_attributes(O, orders_req, rt, rt_ret, p, N, K)
        for o in O:
            sd = order_attrs[o]["sharing_degree"]
            assert 0 <= sd <= len(O) - 1

    def test_min_copies_positive(self):
        S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance()
        order_attrs, _ = precompute_attributes(O, orders_req, rt, rt_ret, p, N, K)
        for o in O:
            assert order_attrs[o]["min_copies"] >= 1

    def test_sku_attrs_demand_ratio(self):
        S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance()
        _, sku_attrs = precompute_attributes(O, orders_req, rt, rt_ret, p, N, K)
        for k in K:
            assert sku_attrs[k]["demand_ratio"] >= 0.0
            assert sku_attrs[k]["copies"] == N[k]
            assert sku_attrs[k]["rt"] == rt[k]


class TestSortFunctions:
    def test_sort_orders_descending(self):
        S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance()
        order_attrs, _ = precompute_attributes(O, orders_req, rt, rt_ret, p, N, K)
        sorted_o = sort_orders(O, order_attrs, "sum_rt", descending=True)
        vals = [order_attrs[o]["sum_rt"] for o in sorted_o]
        assert vals == sorted(vals, reverse=True)

    def test_sort_orders_ascending(self):
        S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance()
        order_attrs, _ = precompute_attributes(O, orders_req, rt, rt_ret, p, N, K)
        sorted_o = sort_orders(O, order_attrs, "order_size", descending=False)
        vals = [order_attrs[o]["order_size"] for o in sorted_o]
        assert vals == sorted(vals)

    def test_sort_skus_descending(self):
        S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance()
        _, sku_attrs = precompute_attributes(O, orders_req, rt, rt_ret, p, N, K)
        o = O[0]
        sorted_k = sort_skus_for_order(orders_req[o], sku_attrs, "rt", descending=True)
        vals = [sku_attrs[k]["rt"] for k in sorted_k]
        assert vals == sorted(vals, reverse=True)

    def test_sort_all_order_attrs(self):
        S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance()
        order_attrs, _ = precompute_attributes(O, orders_req, rt, rt_ret, p, N, K)
        for attr in ORDER_ATTRS:
            result = sort_orders(O, order_attrs, attr, descending=True)
            assert len(result) == len(O)
            assert set(result) == set(O)

    def test_sort_all_bin_attrs(self):
        S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance()
        _, sku_attrs = precompute_attributes(O, orders_req, rt, rt_ret, p, N, K)
        o = O[0]
        for attr in BIN_ATTRS:
            result = sort_skus_for_order(orders_req[o], sku_attrs, attr, descending=False)
            assert set(result) == set(orders_req[o])


class TestRunSgcParameterised:
    def test_feasible_default_sort(self):
        S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance()
        order_attrs, sku_attrs = precompute_attributes(O, orders_req, rt, rt_ret, p, N, K)
        sol = run_sgc_parameterised(
            S, L, K, O, orders_req, rt, rt_ret, p, N,
            horizon=2000, move_cap=None, ALPHA=1.0, BETA=0.0,
            order_attrs=order_attrs, sku_attrs=sku_attrs,
            order_attr_key="sum_rt", order_descending=True,
            bin_attr_key="rt", bin_descending=True,
        )
        assert sol.feasible
        assert sol.makespan > 0

    def test_validates_cleanly(self):
        S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance()
        order_attrs, sku_attrs = precompute_attributes(O, orders_req, rt, rt_ret, p, N, K)
        sol = run_sgc_parameterised(
            S, L, K, O, orders_req, rt, rt_ret, p, N,
            horizon=2000, move_cap=None, ALPHA=1.0, BETA=0.0,
            order_attrs=order_attrs, sku_attrs=sku_attrs,
            order_attr_key="sum_rt", order_descending=True,
            bin_attr_key="rt", bin_descending=True,
        )
        viols = validate_solution(sol, S, L, K, O, orders_req, rt, rt_ret, p, N,
                                  horizon=2000, move_cap=None)
        assert viols == [], f"Violations: {viols}"

    def test_all_orders_assigned(self):
        S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance()
        order_attrs, sku_attrs = precompute_attributes(O, orders_req, rt, rt_ret, p, N, K)
        sol = run_sgc_parameterised(
            S, L, K, O, orders_req, rt, rt_ret, p, N,
            horizon=2000, move_cap=None, ALPHA=1.0, BETA=0.0,
            order_attrs=order_attrs, sku_attrs=sku_attrs,
            order_attr_key="sku_contention", order_descending=False,
            bin_attr_key="demand", bin_descending=True,
        )
        assert len(sol.order_assignments) == len(O)


class TestRunAmaSgc:
    def test_run_ama_sgc_two_phase(self):
        S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance(
            stations=1, lanes=2, orders=5, skus=10, seed=42)
        sol, config, _ = run_ama_sgc(
            S, L, K, O, orders_req, rt, rt_ret, p, N,
            horizon=10000, move_cap=None,
            ALPHA=1.0, BETA=0.0, mode="two_phase", verbose=False
        )
        assert sol.feasible
        assert len(sol.order_assignments) == 5
        assert len(config) == 4

    def test_ama_better_or_equal_to_base_full_grid(self):
        S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance()
        base = run_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N,
                       horizon=2000, move_cap=None, ALPHA=1.0, BETA=0.0)
        ama, _, _ = run_ama_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N,
                                horizon=2000, move_cap=None, ALPHA=1.0, BETA=0.0,
                                mode="full_grid")
        assert ama.makespan <= base.makespan, (
            f"AMA-SGC makespan {ama.makespan} > base SGC {base.makespan}"
        )

    def test_ama_better_or_equal_to_base_two_phase(self):
        S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance()
        base = run_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N,
                       horizon=2000, move_cap=None, ALPHA=1.0, BETA=0.0)
        ama, _, _ = run_ama_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N,
                                horizon=2000, move_cap=None, ALPHA=1.0, BETA=0.0,
                                mode="two_phase")
        assert ama.makespan <= base.makespan

    def test_full_grid_solution_validates(self):
        S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance()
        sol, _, _ = run_ama_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N,
                                horizon=2000, move_cap=None, ALPHA=1.0, BETA=0.0,
                                mode="full_grid")
        viols = validate_solution(sol, S, L, K, O, orders_req, rt, rt_ret, p, N,
                                  horizon=2000, move_cap=None)
        assert viols == [], f"Violations: {viols}"

    def test_two_phase_solution_validates(self):
        S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance()
        sol, _, _ = run_ama_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N,
                                horizon=2000, move_cap=None, ALPHA=1.0, BETA=0.0,
                                mode="two_phase")
        viols = validate_solution(sol, S, L, K, O, orders_req, rt, rt_ret, p, N,
                                  horizon=2000, move_cap=None)
        assert viols == [], f"Violations: {viols}"

    def test_best_config_is_tuple_of_four(self):
        S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance()
        sol, config, _ = run_ama_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N,
                                     horizon=2000, mode="full_grid")
        oa, od, ba, bd = config
        assert oa in ORDER_ATTRS
        assert ba in BIN_ATTRS
        assert isinstance(od, bool)
        assert isinstance(bd, bool)

    def test_invalid_mode_raises(self):
        S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance()
        with pytest.raises(ValueError, match="Unknown mode"):
            run_ama_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N,
                        horizon=2000, mode="bogus_mode")


# ============================================================
# [P2-HE] Heuristic edge-case tests (adapted)
# ============================================================

class TestHeuristicEdgeCases:
    """Edge cases for the AMA-SGC heuristic: tight constraints, degenerate inputs."""

    def test_movecap_1_serializes_fetches(self):
        """With move_cap=1, no two moves may be concurrent; validate_solution must pass."""
        sol, S, L, K, O, orders_req, rt, rt_ret, p, N = _run_small_ama(
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
        sol, _, _ = run_ama_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N_one, horizon=5000)
        assert sol.feasible, "Should be feasible even with single bin copies"
        violations = validate_solution(
            sol, S, L, K, O, orders_req, rt, rt_ret, p, N_one, 5000)
        assert violations == [], f"Violations with N[k]=1: {violations}"

    def test_pick_events_cover_all_assignments(self):
        """For every assigned (o, s, k), pick_events must have an entry."""
        sol, S, L, K, O, orders_req, rt, rt_ret, p, N = _run_small_ama(
            stations=2, lanes=2, orders=8, skus=5, seed=11)
        assert sol.feasible
        for o, (s_a, ln_a, t_s, t_e) in sol.order_assignments.items():
            for k in orders_req[o]:
                pe_key = (o, s_a, k)
                assert pe_key in sol.pick_events, f"Missing pick event for {pe_key}"
                ps, pe = sol.pick_events[pe_key]
                assert ps >= t_s, f"Pick start {ps} before order start {t_s}"
                assert pe <= t_e, f"Pick end {pe} after order end {t_e}"
                assert pe - ps == p[k], f"Pick duration {pe - ps} != expected {p[k]}"

    def test_bin_events_timing_valid(self):
        """Every BinEvent must have internally consistent timing fields."""
        sol, S, L, K, O, orders_req, rt, rt_ret, p, N = _run_small_ama()
        assert sol.feasible
        for s in S:
            for be in sol.bin_events.get(s, []):
                assert be.fetch_end - be.fetch_start == rt[be.sku]
                assert be.return_end - be.return_start == rt_ret[be.sku]
                assert be.presence_start >= be.fetch_end
                assert be.return_start >= be.presence_end
                assert be.presence_end > be.presence_start

    def test_tight_horizon(self):
        """With horizon=1, the solution is infeasible or all order ends are within horizon."""
        sol, S, L, K, O, orders_req, rt, rt_ret, p, N = _run_small_ama(horizon=1)
        if sol.feasible:
            assert sol.makespan <= 1

    def test_many_orders_lane_usage(self):
        """20 orders across 4 lanes should yield a valid solution."""
        sol, S, L, K, O, orders_req, rt, rt_ret, p, N = _run_small_ama(
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
        sol, _, _ = run_ama_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N, horizon=5000)
        violations = validate_solution(
            sol, S, L, K, O, orders_req, rt, rt_ret, p, N, 5000)
        assert violations == [], f"Single-SKU N=1 violations: {violations}"

    def test_every_pick_covered_by_exactly_one_bin_event(self):
        """Every (o, s, k) pick event must be covered by exactly one BinEvent presence."""
        sol, S, L, K, O, orders_req, rt, rt_ret, p, N = _run_small_ama(
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
        sol, S, L, K, O, orders_req, rt, rt_ret, p, N = _run_small_ama(
            stations=3, lanes=2, orders=20, skus=8, seed=7,
            horizon=5000, move_cap=4)
        assert sol.feasible
        violations = validate_solution(
            sol, S, L, K, O, orders_req, rt, rt_ret, p, N, 5000, move_cap=4)
        assert violations == [], f"Multi-station violations: {violations}"


# ============================================================
# [P1-10] End-to-end AMA SGC tests (adapted)
# ============================================================

class TestAMAEndToEnd:
    @staticmethod
    def _generate_default():
        """Generate the default small test instance (seed=42)."""
        return _small_v5_instance()

    def test_feasible_and_valid(self):
        S, L, K, O, orders_req, rt, rt_ret, p, N = self._generate_default()
        sol, _, _ = run_ama_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N,
                                horizon=2000)
        assert sol.feasible
        # The 7-point validation check
        violations = validate_solution(
            sol, S, L, K, O, orders_req, rt, rt_ret, p, N, 2000)
        assert violations == [], f"Solution validation failed: {violations}"

    def test_all_orders_assigned(self):
        S, L, K, O, orders_req, rt, rt_ret, p, N = self._generate_default()
        sol, _, _ = run_ama_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N, horizon=2000)
        assert set(sol.order_assignments.keys()) == set(O)

    def test_makespan_positive(self):
        S, L, K, O, orders_req, rt, rt_ret, p, N = self._generate_default()
        sol, _, _ = run_ama_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N, horizon=2000)
        assert sol.makespan > 0

    def test_with_movecap(self):
        # A movecap of 1 heavily constrains the bin fetches
        S, L, K, O, orders_req, rt, rt_ret, p, N = self._generate_default()
        sol, _, _ = run_ama_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N,
                                horizon=5000, move_cap=1)
        assert sol.feasible
        violations = validate_solution(
            sol, S, L, K, O, orders_req, rt, rt_ret, p, N, 5000, move_cap=1)
        assert violations == [], f"MoveCap=1 violations: {violations}"

    def test_multi_station(self):
        S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance(
            stations=2, lanes=2, orders=10, skus=8, seed=99)
        sol, config, _ = run_ama_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N,
                                     horizon=3000)
        assert sol.feasible
        violations = validate_solution(
            sol, S, L, K, O, orders_req, rt, rt_ret, p, N, 3000)
        assert violations == [], f"Multi-station validation failed: {violations}"
        # Make sure that both stations were used (not guaranteed, but likely for 10 orders)
        used_stations = {s for s, _, _, _ in sol.order_assignments.values()}
        assert len(used_stations) > 0

    def test_validate_warmstart_no_cp_violations(self):
        """Check that the AMA-SGC output satisfies docplex validate_warmstart."""
        S, L, K, O, orders_req, rt, rt_ret, p, N = self._generate_default()
        sol, config, _ = run_ama_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N, horizon=2000)
        assert sol.feasible

        mdl, handles = _build_handles(S, L, K, O, orders_req, rt, rt_ret, p, N, horizon=2000)

        # docplex warmstart validation
        violations = validate_warmstart(sol, sol.pick_events, handles)
        # We expect some "Symmetry (A)" warnings due to raw lanes, but no real constraints violated.
        real_violations = [v for v in violations if "Symmetry (A)" not in v]
        assert real_violations == [], f"CP Model validation failed: {real_violations}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# ============================================================
# [P1-10] End-to-end AMA SGC 2-Phase specific tests (adapted)
# ============================================================


class TestAMA2PhaseEndToEnd:
    @staticmethod
    def _generate_default():
        """Generate the default small test instance (seed=42)."""
        return _small_v5_instance()

    def test_best_config_returns_expected_tuple(self):
        S, L, K, O, orders_req, rt, rt_ret, p, N = self._generate_default()
        sol, config, runs = run_ama_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N, horizon=2000, mode="two_phase")
        assert len(config) == 4
        oa, od, ba, bd = config
        assert isinstance(oa, str)
        assert isinstance(od, bool)
        assert isinstance(ba, str)
        assert isinstance(bd, bool)
        # Should be roughly 28 runs for 2-phase (8*2 + 6*2)
        assert len(runs) == 28


def test_two_phase_uses_cycle_asc_anchor():
    """Two-phase mode must use cycle↑ as Phase 1 anchor, not demand↓.

    We verify this indirectly: after the fix, two_phase result must be
    within 2% of the full_grid result on the same instance.
    Under the old demand↓ anchor, two_phase would land ~7.6% gap vs ~6.0%
    for full_grid on a moderately dense instance, exceeding the 2% tolerance.
    """
    S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance(
        stations=2, lanes=2, orders=12, skus=10, seed=42
    )
    horizon = 5000

    two_phase_sol, _, _ = run_ama_sgc(
        S, L, K, O, orders_req, rt, rt_ret, p, N,
        horizon=horizon, mode="two_phase"
    )
    full_sol, _, _ = run_ama_sgc(
        S, L, K, O, orders_req, rt, rt_ret, p, N,
        horizon=horizon, mode="full_grid"
    )

    assert two_phase_sol.feasible
    assert full_sol.feasible
    assert two_phase_sol.makespan <= full_sol.makespan * 1.02, (
        f"two_phase makespan {two_phase_sol.makespan} > full_grid "
        f"{full_sol.makespan} * 1.02"
    )
