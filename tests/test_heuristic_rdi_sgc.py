import os
import sys

from cp_model import build_model, validate_warmstart
from heuristic_rdi_sgc import (
    build_sku_order_index, compute_regret_k, compute_dirty_set,
    InsertionScores, run_rdi_sgc
)
from autostore_heuristic import validate_solution
from datagen import generate_data as gen_v5


def _small_v5_instance(stations=1, lanes=2, orders=7, skus=5, seed=42):
    S, L, K, orders_req, rt, p, N = gen_v5(
        num_stations=stations, lanes_per_station=lanes, num_orders=orders,
        num_skus=skus, seed=seed, pick_touch_time=4,
    )
    rt_ret = dict(rt)
    O = sorted(orders_req.keys())
    return S, L, K, O, orders_req, rt, rt_ret, p, N


def _run_small_rdi(stations=1, lanes=2, orders=7, skus=5, seed=42, horizon=2000, move_cap=None, **kwargs):
    S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance(stations, lanes, orders, skus, seed)
    sol = run_rdi_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N, horizon=horizon, move_cap=move_cap, **kwargs)
    return sol, S, L, K, O, orders_req, rt, rt_ret, p, N


def _build_handles(S, L, K, O, orders_req, rt, rt_ret, p, N, horizon=2000, move_cap=None):
    mdl, handles = build_model(
        S, L, K, orders_req, rt, p, rt_return=rt_ret, horizon=horizon, N=N,
        move_cap=move_cap, add_symmetry_breaking=True,
    )
    return mdl, handles


def test_helpers_load():
    assert _small_v5_instance() is not None


class TestRDIComponents:
    def test_build_sku_order_index(self):
        orders_req = {1: [10, 20], 2: [20, 30], 3: [10]}
        idx = build_sku_order_index(orders_req)
        assert idx[10] == {1, 3}
        assert idx[20] == {1, 2}
        assert idx[30] == {2}

    def test_compute_regret_k(self):
        scores = {0: 10.0, 1: 15.0, 2: 20.0, 3: float('inf')}
        assert compute_regret_k(scores, 2) == 5.0   # 15 - 10
        assert compute_regret_k(scores, 3) == 15.0  # (15-10) + (20-10)
        assert compute_regret_k({0: 10.0}, 2) == float('inf') # Only 1 option

    def test_compute_dirty_set(self):
        orders_req = {1: [10], 2: [10], 3: [20], 4: [30]}
        idx = {10: {1, 2}, 20: {3}, 30: {4}}
        unscheduled = {2, 3, 4}
        
        cached = {
            2: InsertionScores(2, {}, {}, 1.0, 2.0, 1, None, 1.0),
            3: InsertionScores(3, {}, {}, 1.0, 2.0, 0, None, 1.0),
            4: InsertionScores(4, {}, {}, 1.0, 2.0, 1, None, 1.0)
        }
        
        # Commit order 1 at station 0.
        # Order 2 is dirty (shares SKU 10). Order 3 is dirty (best_station = 0). Order 4 is clean.
        dirty = compute_dirty_set(1, 0, orders_req, idx, cached, unscheduled)
        assert dirty == {2, 3}


class TestRDIEdgeCases:
    def test_movecap_1_serializes_fetches(self):
        sol, S, L, K, O, orders_req, rt, rt_ret, p, N = _run_small_rdi(
            stations=1, lanes=2, orders=5, skus=4, seed=3, horizon=5000, move_cap=1)
        assert sol.feasible
        violations = validate_solution(sol, S, L, K, O, orders_req, rt, rt_ret, p, N, 5000, move_cap=1)
        assert violations == []

    def test_bin_copy_exhaustion(self):
        S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance(
            stations=1, lanes=2, orders=4, skus=2, seed=5)
        N_one = {k: 1 for k in K}
        sol = run_rdi_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N_one, horizon=5000)
        assert sol.feasible
        violations = validate_solution(sol, S, L, K, O, orders_req, rt, rt_ret, p, N_one, 5000)
        assert violations == []

    def test_tight_horizon(self):
        sol, S, L, K, O, orders_req, rt, rt_ret, p, N = _run_small_rdi(horizon=1)
        if sol.feasible:
            assert sol.makespan <= 1

    def test_single_sku_n1_serialized(self):
        S, L, K, O = [0], [0, 1], [0], [0, 1, 2]
        orders_req = {0: [0], 1: [0], 2: [0]}
        rt, rt_ret, p, N = {0: 20}, {0: 20}, {0: 4}, {0: 1}
        sol = run_rdi_sgc(S, L, K, O, orders_req, rt, rt_ret, p, N, horizon=5000)
        violations = validate_solution(sol, S, L, K, O, orders_req, rt, rt_ret, p, N, 5000)
        assert violations == []
        
    def test_lazy_vs_full_rescore(self):
        # Ensure lazy re-scoring yields identical solutions to full re-scoring
        sol_lazy, *_ = _run_small_rdi(stations=1, lanes=3, orders=8, skus=10, seed=42, use_lazy=True)
        sol_full, *_ = _run_small_rdi(stations=1, lanes=3, orders=8, skus=10, seed=42, use_lazy=False)
        
        assert sol_lazy.feasible == sol_full.feasible
        assert sol_lazy.makespan == sol_full.makespan
        assert sol_lazy.total_moves == sol_full.total_moves
        
    def test_tiebreakers_produce_valid_results(self):
        sol_sd, *_ = _run_small_rdi(tiebreaker="sharing_degree")
        assert sol_sd.feasible
        sol_bs, *_ = _run_small_rdi(tiebreaker="best_score")
        assert sol_bs.feasible


class TestRDIEndToEnd:
    def test_feasible_and_valid(self):
        sol, S, L, K, O, orders_req, rt, rt_ret, p, N = _run_small_rdi(horizon=2000)
        assert sol.feasible
        violations = validate_solution(sol, S, L, K, O, orders_req, rt, rt_ret, p, N, 2000)
        assert violations == []

    def test_all_orders_assigned(self):
        sol, S, L, K, O, orders_req, rt, rt_ret, p, N = _run_small_rdi(horizon=2000)
        assert set(sol.order_assignments.keys()) == set(O)

    def test_multi_station(self):
        sol, S, L, K, O, orders_req, rt, rt_ret, p, N = _run_small_rdi(
            stations=2, lanes=2, orders=10, skus=8, seed=99, horizon=3000)
        assert sol.feasible
        violations = validate_solution(sol, S, L, K, O, orders_req, rt, rt_ret, p, N, 3000)
        assert violations == []

    def test_validate_warmstart_no_cp_violations(self):
        sol, S, L, K, O, orders_req, rt, rt_ret, p, N = _run_small_rdi(horizon=2000)
        assert sol.feasible
        mdl, handles = _build_handles(S, L, K, O, orders_req, rt, rt_ret, p, N, horizon=2000)
        violations = validate_warmstart(sol, sol.pick_events, handles)
        real_violations = [v for v in violations if "Symmetry (A)" not in v]
        assert real_violations == []
