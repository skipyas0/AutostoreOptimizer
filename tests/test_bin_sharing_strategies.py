"""Bin-sharing neighbourhood strategies (neighborhood_selection.py)."""

import os
import random
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cp_model_utils import CppSolveResult  # noqa: E402
from cpp_cp_model import build_handles  # noqa: E402
from datagen import generate_data  # noqa: E402
from heuristics import run_heuristic  # noqa: E402
from matheuristic import prepare_model_subproblem  # noqa: E402
from neighborhood_selection import (  # noqa: E402
    StrategyManager,
    _refetch_counts,
    fetch_pick_counts,
    strategy_fetch_pick_ratio_timeslice,
    strategy_refetched_sku_fetch_density,
    strategy_refetched_sku_pick_slices,
)


def _iv(start, end):
    return {"present": True, "start": start, "end": end, "size": end - start}


class _Inst:
    """Tiny instance: 1 station, 4 orders, SKUs 0 and 1."""
    S, L, K = [0], [0], [0, 1]
    orders_req = {0: [0], 1: [0], 2: [0], 3: [1]}
    rt = {0: 5, 1: 5}
    rt_ret = rt
    p = {0: 1, 1: 1}
    N = {0: 1, 1: 1}
    movecap, pickcap = 2, 1

    def __iter__(self):
        yield from (self.S, self.L, self.K, self.orders_req, self.rt, self.p, self.N)


def _hand_solution():
    """SKU 0 fetched three times (2 refetches), SKU 1 once.

    t:   0-10 order 0 | 10-20 order 1 | 100-110 order 2 | 200-210 order 3
    SKU 0 visit 0 serves orders 0 and 1 (shared), visits 1 and 2 serve order 2 and
    nothing; SKU 1 visit 0 serves order 3.
    """
    handles = build_handles(_Inst(), horizon=1000)
    vs = {}
    for o, (start, end) in {0: (0, 10), 1: (10, 20), 2: (100, 110), 3: (200, 210)}.items():
        vs[f"I_os[{o},0]"] = _iv(start, end)
    vs["F[0,0,0]"], vs["P[0,0,0,0]"], vs["P[1,0,0,0]"] = _iv(0, 5), _iv(5, 6), _iv(15, 16)
    vs["F[0,0,1]"], vs["P[2,0,0,1]"] = _iv(95, 100), _iv(100, 101)
    vs["F[0,0,2]"] = _iv(150, 155)  # a visit without picks
    vs["F[0,1,0]"], vs["P[3,0,1,0]"] = _iv(195, 200), _iv(200, 201)
    return handles, CppSolveResult({"status": "Feasible", "var_solutions": vs})


class TestBinSharingStrategies(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        self.handles, self.sol = _hand_solution()

    def test_refetch_counts(self):
        self.assertEqual(_refetch_counts(self.handles, self.sol), {0: 2})

    def test_pick_slices(self):
        # tiny slices (width 2) around SKU 0's picks hit exactly the picking orders
        res = strategy_refetched_sku_pick_slices(self.handles, self.sol, 10, 2)
        self.assertEqual(res.seed_orders, {0, 1, 2})
        res = strategy_refetched_sku_pick_slices(self.handles, self.sol, 1, 2)
        self.assertEqual(len(res.seed_orders), 1)

    def test_fetch_density(self):
        # width 60: windows over [95, 100) and [150, 155) hold two SKU-0 fetches
        res = strategy_refetched_sku_fetch_density(self.handles, self.sol, 60)
        self.assertEqual(res.seed_orders, {2})

    def test_fetch_pick_counts(self):
        visits = [(0, 0, (0, 5), [(5, 6), (15, 16)]), (0, 1, (30, 35), [(80, 81)])]
        fetches, picks = fetch_pick_counts(visits, 100, 12)
        # window [0, 12): fetch 0 and its first pick
        self.assertEqual((fetches[0], picks[0]), (1, 1))
        # window [5, 17) leaves the fetch [0, 5): nothing counts
        self.assertEqual((fetches[5], picks[5]), (0, 0))
        # window [4, 16): fetch 0 and both picks
        self.assertEqual((fetches[4], picks[4]), (1, 2))
        # fetch [30, 35) and pick [80, 81) never fit in one window of 12
        self.assertEqual(int(fetches[30:].sum() + picks[30:].sum()), 0)

    def test_fetch_pick_ratio(self):
        # width 12: every visit with a pick reaches ratio 1.0; ties are random, but the
        # window never lies where nothing counts (the pick-less visit at 150-155)
        seen = set()
        for _ in range(30):
            res = strategy_fetch_pick_ratio_timeslice(self.handles, self.sol, 12)
            self.assertIn(frozenset(res.seed_orders), {frozenset({0, 1}), frozenset({2}),
                                                       frozenset({3})})
            seen.add(frozenset(res.seed_orders))
        self.assertGreater(len(seen), 1)  # ties broken at random

    def test_fetch_pick_ratio_prefers_unshared(self):
        # make picks of the shared visit both fall in every counting window: ratio 1/2
        vs = dict(self.sol.var_sols)
        vs["P[1,0,0,0]"] = _iv(5, 6)  # same time as order 0's pick
        vs["F[0,0,1]"], vs["P[2,0,0,1]"] = _iv(97, 100), _iv(100, 101)
        sol = CppSolveResult({"status": "Feasible", "var_solutions": vs})
        del vs["F[0,1,0]"], vs["P[3,0,1,0]"]
        for _ in range(20):
            res = strategy_fetch_pick_ratio_timeslice(self.handles, sol, 4)
            self.assertEqual(res.seed_orders, {2})

    def test_no_refetches(self):
        vs = dict(self.sol.var_sols)
        del vs["F[0,0,1]"], vs["F[0,0,2]"]
        sol = CppSolveResult({"status": "Feasible", "var_solutions": vs})
        self.assertEqual(strategy_refetched_sku_pick_slices(self.handles, sol, 10, 2).seed_orders, set())
        self.assertEqual(strategy_refetched_sku_fetch_density(self.handles, sol, 60).seed_orders, set())

    def test_preset_on_real_instance(self):
        inst = generate_data(num_stations=2, lanes_per_station=2, num_orders=30, num_skus=40,
                             seed=2, movecap=4, pickcap=1)
        heur = run_heuristic("load_balanced_greedy", inst, horizon=10000)
        _, handles, sol = prepare_model_subproblem({"horizon": 10000}, inst, heur)
        mgr = StrategyManager(handles, sol, None, {}, "bin_sharing", default_timeslice_width=100)
        self.assertEqual(set(mgr.active_strats), {
            "refetched_sku_pick_slices", "refetched_sku_fetch_density",
            "fetch_pick_ratio_timeslice"})
        for name, strat in mgr.active_strats.items():
            for sev in (1, 3):
                res = strat(sev)
                self.assertTrue(res.seed_orders <= set(handles["O"]), name)
                self.assertTrue(res.seed_orders, (name, sev))


if __name__ == "__main__":
    unittest.main()
