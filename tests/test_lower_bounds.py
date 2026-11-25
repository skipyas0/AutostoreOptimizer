"""Validity of lower_bounds.py: no feasible heuristic schedule may beat the W or C bounds.

Small random instances with few lanes and skewed SKU popularity, so hot SKUs (more than L
multi-line orders) exist and the lane bound is above the global one.
"""

import contextlib
import io
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datagen import generate_data  # noqa: E402
from heuristics import run_heuristic  # noqa: E402
from heuristics.evolve import robot_work  # noqa: E402
from lower_bounds import makespan_lb, work_lb_global, work_lb_lane  # noqa: E402

H = 10000
HEURS = ["lb_sgc", "lb_sgc_share", "gbs_max_sharing", "gbs_cluster_lpt", "rdi"]
CASES = [(2, 1, 30, 400, 20), (3, 2, 40, 300, 10), (2, 2, 30, 200, 4), (4, 1, 50, 600, 30)]


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("S,L,O,K,mc", CASES)
def test_bounds_below_heuristics(S, L, O, K, mc, seed):
    inst = generate_data(num_stations=S, lanes_per_station=L, num_orders=O, num_skus=K, seed=seed,
                         sku_popularity_skew=1.5, movecap=mc)
    wb = work_lb_lane(inst, time_limit=10, workers=1)
    assert wb.global_lb == work_lb_global(inst) <= wb.value
    cb = makespan_lb(inst, work_lb=wb.value)
    for h in HEURS:
        with contextlib.redirect_stdout(io.StringIO()):
            sol = run_heuristic(h, inst, horizon=H)
        if not sol.feasible:
            continue
        assert robot_work(sol, inst) >= wb.value, h
        assert sol.makespan >= cb.value, h


def test_lane_bound_bites_with_one_lane():
    inst = generate_data(num_stations=2, lanes_per_station=1, num_orders=40, num_skus=200, seed=7,
                         sku_popularity_skew=1.5, movecap=20)
    wb = work_lb_lane(inst, time_limit=10, workers=1)
    assert wb.hot_skus > 0 and wb.value > wb.global_lb


@pytest.mark.parametrize("seed", [1, 2])
def test_sequence_relaxations_below_heuristics(seed):
    from lower_bounds import coupled_relaxation_lb, seq_relaxation_lb

    inst = generate_data(num_stations=2, lanes_per_station=1, num_orders=8, num_skus=30, seed=seed,
                         sku_popularity_skew=1.5, movecap=4)
    sols = []
    for h in HEURS:
        with contextlib.redirect_stdout(io.StringIO()):
            sol = run_heuristic(h, inst, horizon=H)
        if sol.feasible:
            sols.append(sol)
    sq = seq_relaxation_lb(inst, time_limit=10, workers=1, hint=sols[0])
    assert sq.value >= work_lb_lane(inst, time_limit=10, workers=1).value or not sq.optimal
    cr = coupled_relaxation_lb(inst, time_limit=10, workers=1)
    for sol in sols:
        assert robot_work(sol, inst) >= sq.value
        assert sol.makespan >= cr.value
    if sq.upper is not None:  # a relaxed sequence is runnable on one station: never below the bound
        assert sq.upper >= sq.value
