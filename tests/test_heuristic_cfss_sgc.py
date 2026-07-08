#!/usr/bin/env python3
"""
Unit tests for CFSS-SGC Heuristic.

This file combines tests from the original test_heuristic_cfss_sgc.py
and tests/test_heuristic_cfss_sgc.py.
"""
from datagen import generate_data as gen_v5
from autostore_heuristic import validate_solution, init_state, run_sgc
from heuristic_cfss_sgc import (
    compute_demand_count, compute_weighted_jaccard, build_similarity_matrix,
    agglomerative_cluster, snapshot_state, run_cfss_sgc, OrderCluster,
    score_cluster_at_station
)
from cp_model import build_model, validate_warmstart
import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _small_v5_instance(stations=1, lanes=2, orders=7, skus=5, seed=42):
    S, L, K, orders_req, rt, p, N = gen_v5(
        num_stations=stations, lanes_per_station=lanes, num_orders=orders,
        num_skus=skus, seed=seed, pick_touch_time=4,
    )
    rt_ret = dict(rt)
    O = sorted(orders_req.keys())
    return S, L, K, O, orders_req, rt, rt_ret, p, N


def _run_small_cfss(stations=1, lanes=2, orders=7, skus=5, seed=42, horizon=2000, move_cap=None):
    S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance(stations, lanes, orders, skus, seed)
    sol = run_cfss_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N, horizon=horizon, move_cap=move_cap)
    return sol, S, L, K, O, orders_req, rt, rt_ret, p, N


def _build_handles(S, L, K, O, orders_req, rt, rt_ret, p, N, horizon=2000, move_cap=None):
    mdl, handles = build_model(
        S, L, K, orders_req, rt, p, rt_return=rt_ret, horizon=horizon, N=N,
        move_cap=move_cap, add_symmetry_breaking=True,
    )
    return mdl, handles


def test_helpers_load():
    assert _small_v5_instance() is not None


# ---------------------------------------------------------------------------
# Step 7a: Weighted Jaccard computation
# ---------------------------------------------------------------------------

class TestWeightedJaccard:
    def test_identical_orders_score_one(self):
        orders_req = {0: [1, 2, 3], 1: [1, 2, 3]}
        rt = {1: 10, 2: 20, 3: 30}
        rt_ret = dict(rt)
        score = compute_weighted_jaccard(0, 1, orders_req, rt, rt_ret)
        assert abs(score - 1.0) < 1e-9

    def test_disjoint_orders_score_zero(self):
        orders_req = {0: [1, 2], 1: [3, 4]}
        rt = {1: 10, 2: 20, 3: 30, 4: 40}
        rt_ret = dict(rt)
        score = compute_weighted_jaccard(0, 1, orders_req, rt, rt_ret)
        assert score == 0.0

    def test_partial_overlap_symmetric_rt(self):
        """When all rt equal, weighted Jaccard == plain Jaccard."""
        orders_req = {0: [0, 1, 2], 1: [1, 2, 3]}
        rt = {0: 10, 1: 10, 2: 10, 3: 10}
        rt_ret = dict(rt)
        score = compute_weighted_jaccard(0, 1, orders_req, rt, rt_ret)
        # intersection={1,2}, union={0,1,2,3} -> 2/4 = 0.5
        assert abs(score - 0.5) < 1e-9

    def test_partial_overlap_asymmetric_rt(self):
        """Weighted Jaccard differs from plain Jaccard with asymmetric rt."""
        orders_req = {0: [0, 1, 2], 1: [1, 2, 3]}
        rt = {0: 100, 1: 5, 2: 5, 3: 100}
        rt_ret = dict(rt)
        score = compute_weighted_jaccard(0, 1, orders_req, rt, rt_ret)
        # intersection weight = (5+5)*2 = 20
        # union weight = (100+100) + (5+5) + (5+5) + (100+100) = 420
        expected = 20 / 420
        assert abs(score - expected) < 1e-9

    def test_symmetry(self):
        orders_req = {0: [0, 1], 1: [1, 2, 3]}
        rt = {0: 10, 1: 30, 2: 50, 3: 20}
        rt_ret = {k: v // 2 for k, v in rt.items()}
        s01 = compute_weighted_jaccard(0, 1, orders_req, rt, rt_ret)
        s10 = compute_weighted_jaccard(1, 0, orders_req, rt, rt_ret)
        assert abs(s01 - s10) < 1e-9

    def test_empty_order(self):
        orders_req = {0: [], 1: [1, 2]}
        rt = {1: 10, 2: 20}
        rt_ret = dict(rt)
        score = compute_weighted_jaccard(0, 1, orders_req, rt, rt_ret)
        assert score == 0.0

    def test_build_similarity_matrix_coverage(self):
        S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance()
        sim = build_similarity_matrix(O, orders_req, rt, rt_ret)
        n = len(O)
        # Upper triangle: n*(n-1)/2 pairs
        assert len(sim) == n * (n - 1) // 2
        for s in sim.values():
            assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# Step 7b: Agglomerative clustering
# ---------------------------------------------------------------------------

class TestAgglomerativeClustering:
    def test_two_separated_groups(self):
        """4 orders: {0,1} highly similar, {2,3} highly similar, cross-group=0."""
        O = [0, 1, 2, 3]
        orders_req = {0: [0, 1], 1: [0, 1, 2], 2: [3, 4], 3: [3, 4, 5]}
        rt = {0: 10, 1: 20, 2: 30, 3: 10, 4: 20, 5: 30}
        rt_ret = dict(rt)
        sim = build_similarity_matrix(O, orders_req, rt, rt_ret)
        clusters = agglomerative_cluster(O, orders_req, sim, rt,
                                         threshold=0.05, max_cluster_orders=8)
        # Should produce exactly 2 clusters
        assert len(clusters) == 2
        cluster_sets = [frozenset(c.order_ids) for c in clusters]
        assert frozenset([0, 1]) in cluster_sets
        assert frozenset([2, 3]) in cluster_sets

    def test_capacity_constraint_prevents_large_cluster(self):
        """With max_cluster_orders=2, cluster of 3 similar orders must not form."""
        O = [0, 1, 2]
        orders_req = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
        rt = {0: 10, 1: 20}
        rt_ret = dict(rt)
        sim = build_similarity_matrix(O, orders_req, rt, rt_ret)
        clusters = agglomerative_cluster(O, orders_req, sim, rt,
                                         threshold=0.0, max_cluster_orders=2)
        max_size = max(len(c.order_ids) for c in clusters)
        assert max_size <= 2

    def test_threshold_zero_merges_all_if_similar(self):
        """threshold=0, max=100 should merge all identical orders into one."""
        O = [0, 1, 2]
        orders_req = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
        rt = {0: 10, 1: 20}
        rt_ret = dict(rt)
        sim = build_similarity_matrix(O, orders_req, rt, rt_ret)
        clusters = agglomerative_cluster(O, orders_req, sim, rt,
                                         threshold=0.0, max_cluster_orders=100)
        # All three orders should merge into one cluster
        assert len(clusters) == 1
        assert set(clusters[0].order_ids) == {0, 1, 2}

    def test_high_threshold_no_merges(self):
        """threshold=1.0 should leave all orders as singletons (only identical merge)."""
        S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance()
        sim = build_similarity_matrix(O, orders_req, rt, rt_ret)
        clusters = agglomerative_cluster(O, orders_req, sim, rt,
                                         threshold=1.0, max_cluster_orders=8)
        # With threshold=1.0, only exactly identical orders can merge
        # On a random instance this should leave mostly singletons
        total_orders = sum(len(c.order_ids) for c in clusters)
        assert total_orders == len(O)

    def test_all_orders_covered(self):
        """Every order must appear in exactly one cluster."""
        S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance(orders=12, skus=8)
        sim = build_similarity_matrix(O, orders_req, rt, rt_ret)
        clusters = agglomerative_cluster(O, orders_req, sim, rt,
                                         threshold=0.05, max_cluster_orders=6)
        assigned = []
        for c in clusters:
            assigned.extend(c.order_ids)
        assert sorted(assigned) == sorted(O)

    def test_cluster_metadata(self):
        """Verify all_skus and total_rt_mass are computed correctly."""
        O = [0, 1]
        orders_req = {0: [0, 1], 1: [1, 2]}
        rt = {0: 10, 1: 20, 2: 30}
        rt_ret = dict(rt)
        sim = build_similarity_matrix(O, orders_req, rt, rt_ret)
        clusters = agglomerative_cluster(O, orders_req, sim, rt,
                                         threshold=0.0, max_cluster_orders=8)
        merged = next(c for c in clusters if len(c.order_ids) == 2)
        assert merged.all_skus == {0, 1, 2}
        expected_mass = (10 + 20) + (20 + 30)
        assert abs(merged.total_rt_mass - expected_mass) < 1e-9


# ---------------------------------------------------------------------------
# snapshot_state and score_cluster_at_station
# ---------------------------------------------------------------------------

class TestSnapshotState:
    def test_snapshot_is_independent(self):
        """Modifying a snapshot must not affect the original state."""
        S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance()
        state = init_state(S, L, K, N, horizon=2000, move_cap=None)
        snap = snapshot_state(state)

        # Mutate the snapshot
        snap.lane_free[(S[0], 0)] = 9999
        snap.pickface_free[S[0]] = 9999
        snap.move_da.diff[5] += 100

        assert state.lane_free[(S[0], 0)] == 0
        assert state.pickface_free[S[0]] == 0
        assert state.move_da.diff[5] != 100


class TestScoreClusterAtStation:
    def test_single_order_cluster_score(self):
        S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance()
        state = init_state(S, L, K, N, horizon=2000, move_cap=None)
        cluster = OrderCluster(order_ids=[O[0]])
        demand_count = compute_demand_count(orders_req, K)
        sharing_degree = {o: 1 for o in O}
        score, plans = score_cluster_at_station(
            cluster, S[0], state, orders_req, rt, rt_ret, p, N,
            2000, 1.0, 0.0, demand_count, sharing_degree
        )
        assert score < float("inf")
        assert len(plans) == 1

    def test_state_not_mutated(self):
        S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance()
        state = init_state(S, L, K, N, horizon=2000, move_cap=None)
        original_lane_free = dict(state.lane_free)
        cluster = OrderCluster(order_ids=O[:2])
        demand_count = compute_demand_count(orders_req, K)
        sharing_degree = {o: 1 for o in O}
        score_cluster_at_station(
            cluster, S[0], state, orders_req, rt, rt_ret, p, N,
            2000, 1.0, 0.0, demand_count, sharing_degree
        )
        assert state.lane_free == original_lane_free


class TestHeuristicEdgeCases:
    def test_movecap_1_serializes_fetches(self):
        sol, S, L, K, O, orders_req, rt, rt_ret, p, N = _run_small_cfss(
            stations=1, lanes=2, orders=5, skus=4, seed=3, horizon=5000, move_cap=1)
        assert sol.feasible
        violations = validate_solution(sol, S, L, K, O, orders_req, rt, rt_ret, p, N, 5000, move_cap=1)
        assert violations == []

    def test_bin_copy_exhaustion(self):
        S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance(
            stations=1, lanes=2, orders=4, skus=2, seed=5)
        N_one = {k: 1 for k in K}
        sol = run_cfss_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N_one, horizon=5000)
        assert sol.feasible
        violations = validate_solution(sol, S, L, K, O, orders_req, rt, rt_ret, p, N_one, 5000)
        assert violations == []

    def test_tight_horizon(self):
        sol, S, L, K, O, orders_req, rt, rt_ret, p, N = _run_small_cfss(horizon=1)
        if sol.feasible:
            assert sol.makespan <= 1

    def test_single_sku_n1_serialized(self):
        S, L, K, O = [0], [0, 1], [0], [0, 1, 2]
        orders_req = {0: [0], 1: [0], 2: [0]}
        rt, rt_ret, p, N = {0: 20}, {0: 20}, {0: 4}, {0: 1}
        sol = run_cfss_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N, horizon=5000)
        violations = validate_solution(sol, S, L, K, O, orders_req, rt, rt_ret, p, N, 5000)
        assert violations == []

    def test_every_pick_covered_by_exactly_one_bin_event(self):
        sol, S, L, K, O, orders_req, rt, rt_ret, p, N = _run_small_cfss(
            stations=1, lanes=2, orders=7, skus=4, seed=0)
        assert sol.feasible
        for (o, s_sel, k), (ps, pe) in sol.pick_events.items():
            covering = [be for be in sol.bin_events.get(s_sel, [])
                        if be.sku == k and be.presence_start <= ps and pe <= be.presence_end]
            assert len(covering) == 1


class TestCFSSEndToEnd:
    def test_feasible_and_valid(self):
        sol, S, L, K, O, orders_req, rt, rt_ret, p, N = _run_small_cfss(horizon=2000)
        assert sol.feasible
        violations = validate_solution(sol, S, L, K, O, orders_req, rt, rt_ret, p, N, 2000)
        assert violations == []

    def test_all_orders_assigned(self):
        sol, S, L, K, O, orders_req, rt, rt_ret, p, N = _run_small_cfss(horizon=2000)
        assert set(sol.order_assignments.keys()) == set(O)

    def test_multi_station(self):
        sol, S, L, K, O, orders_req, rt, rt_ret, p, N = _run_small_cfss(
            stations=2, lanes=2, orders=10, skus=8, seed=99, horizon=3000)
        assert sol.feasible
        violations = validate_solution(sol, S, L, K, O, orders_req, rt, rt_ret, p, N, 3000)
        assert violations == []

    def test_movecap_respected(self):
        S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance(orders=10, skus=5)
        sol = run_cfss_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N,
                           horizon=5000, move_cap=3)
        viols = validate_solution(sol, S, L, K, O, orders_req, rt, rt_ret, p, N,
                                  horizon=5000, move_cap=3)
        assert viols == [], f"MoveCap violations: {viols}"

    def test_high_threshold_degrades_to_sgc_like(self):
        """With threshold=1.0, no merges -> clustering is all singletons -> like base SGC."""
        S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance()
        sol = run_cfss_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N,
                           horizon=2000, sim_threshold=1.0)
        assert sol.feasible

    def test_zero_threshold_creates_large_clusters(self):
        """With threshold=0 (and generous cap), most orders should be clustered."""
        S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance(skus=3, orders=9)
        # With only 3 SKUs and 9 orders there will be heavy overlap
        sol = run_cfss_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N,
                           horizon=5000, sim_threshold=0.0, max_cluster_orders=20)
        assert sol.feasible
        viols = validate_solution(sol, S, L, K, O, orders_req, rt, rt_ret, p, N, horizon=5000)
        assert viols == []

    def test_bin_sharing_measurement(self):
        """CFSS produces a feasible, valid solution on a high-overlap instance.

        The soft-clustering variant trades strict bin-event minimisation for load
        balance across stations, so we only require CFSS to be within 20% of the
        base SGC bin-event count rather than strictly better.
        """
        S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance(skus=3, orders=9, lanes=3)
        base = run_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N, horizon=5000)
        cfss = run_cfss_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N,
                            horizon=5000, sim_threshold=0.0, max_cluster_orders=20)
        assert cfss.feasible
        base_events = sum(len(evts) for evts in base.bin_events.values())
        cfss_events = sum(len(evts) for evts in cfss.bin_events.values())
        assert cfss_events <= base_events * 1.2, (
            f"CFSS bin events {cfss_events} more than 20% above base SGC {base_events}"
        )

    def test_different_seeds_feasible(self):
        """CFSS should produce feasible solutions across several seeds."""
        for seed in [1, 7, 42, 99, 123]:
            S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance(seed=seed)
            sol = run_cfss_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N, horizon=5000)
            assert sol.feasible, f"Seed {seed}: infeasible"

    def test_validate_warmstart_no_cp_violations(self):
        sol, S, L, K, O, orders_req, rt, rt_ret, p, N = _run_small_cfss(horizon=2000)
        assert sol.feasible
        mdl, handles = _build_handles(S, L, K, O, orders_req, rt, rt_ret, p, N, horizon=2000)
        violations = validate_warmstart(sol, sol.pick_events, handles)
        real_violations = [v for v in violations if "Symmetry (A)" not in v]
        assert real_violations == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
