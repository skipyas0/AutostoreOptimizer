import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neighborhood_selection import (
    StrategyManager,
    strategy_cycle_time_weighted_refetched_skus,
    strategy_frequently_refetched_skus,
    strategy_longest_orders,
)


class MockVarSolution:
    def __init__(self, start=0, end=10, present=True):
        self._start = start
        self._end = end
        self._present = present

    def is_present(self):
        return self._present

    def get_start(self):
        return self._start

    def get_end(self):
        return self._end


class MockSolution:
    def __init__(self, intervals):
        self.intervals = intervals

    def get_var_solution(self, x):
        return self.intervals.get(x)


class TestNeighborhoodSelection(unittest.TestCase):
    def setUp(self):
        # 3 SKUs: 1, 2, 3
        # SKU 1 has 1 fetch
        # SKU 2 has 3 fetches
        # SKU 3 has 2 fetches
        self.handles = {
            "O": [101, 102, 103],
            "S": [0, 1],
            "L": [0],
            "K": [1, 2, 3],
            "active_K": [1, 2, 3],
            "I_os": {
                (101, 0): "I_101_0",
                (101, 1): "I_101_1",
                (102, 0): "I_102_0",
                (102, 1): "I_102_1",
                (103, 0): "I_103_0",
                (103, 1): "I_103_1",
            },
            "F": {
                (0, 1, 0): "F_0_1_0",
                (0, 2, 0): "F_0_2_0",
                (0, 2, 1): "F_0_2_1",
                (1, 2, 0): "F_1_2_0",
                (0, 3, 0): "F_0_3_0",
                (1, 3, 0): "F_1_3_0",
            },
            "rt": {1: 10, 2: 2, 3: 50},
            "p": {1: 2, 2: 1, 3: 5},
            "rt_return": {1: 10, 2: 2, 3: 50},
        }

        # Order 101 flow time: 100 - 10 = 90
        # Order 102 flow time: 30 - 0 = 30
        # Order 103 flow time: 200 - 50 = 150
        self.solution = MockSolution({
            "I_101_0": MockVarSolution(10, 100, present=True),
            "I_101_1": MockVarSolution(0, 0, present=False),
            "I_102_0": MockVarSolution(0, 30, present=True),
            "I_102_1": MockVarSolution(0, 0, present=False),
            "I_103_1": MockVarSolution(50, 200, present=True),
            "I_103_0": MockVarSolution(0, 0, present=False),
            "F_0_1_0": MockVarSolution(0, 10, present=True),
            "F_0_2_0": MockVarSolution(0, 2, present=True),
            "F_0_2_1": MockVarSolution(20, 22, present=True),
            "F_1_2_0": MockVarSolution(10, 12, present=True),
            "F_0_3_0": MockVarSolution(0, 50, present=True),
            "F_1_3_0": MockVarSolution(50, 100, present=True),
        })

    def test_longest_orders(self):
        # 103 (150) > 101 (90) > 102 (30)
        res_k1 = strategy_longest_orders(self.handles, self.solution, k=1)
        self.assertEqual(res_k1.seed_orders, {103})

        res_k2 = strategy_longest_orders(self.handles, self.solution, k=2)
        self.assertEqual(res_k2.seed_orders, {103, 101})

        res_p = strategy_longest_orders(self.handles, self.solution, p=0.4)  # 3 * 0.4 = 1
        self.assertEqual(res_p.seed_orders, {103})

    def test_frequently_refetched_skus_unweighted(self):
        # Fetches count: SKU 2 has 3, SKU 3 has 2, SKU 1 has 1
        res_k1 = strategy_frequently_refetched_skus(self.handles, self.solution, k=1, weighted=False)
        self.assertEqual(res_k1.seed_skus, {2})

        res_k2 = strategy_frequently_refetched_skus(self.handles, self.solution, k=2, weighted=False)
        self.assertEqual(res_k2.seed_skus, {2, 3})

    def test_cycle_time_weighted_refetched_skus(self):
        # Cycle time:
        # SKU 1: (10 + 2 + 10) * 1 = 22
        # SKU 2: (2 + 1 + 2) * 3 = 15
        # SKU 3: (50 + 5 + 50) * 2 = 210
        # Weighted order: SKU 3 (210) > SKU 1 (22) > SKU 2 (15)
        res_k1 = strategy_cycle_time_weighted_refetched_skus(self.handles, self.solution, k=1)
        self.assertEqual(res_k1.seed_skus, {3})

        res_k2 = strategy_cycle_time_weighted_refetched_skus(self.handles, self.solution, k=2)
        self.assertEqual(res_k2.seed_skus, {3, 1})

    def test_strategy_manager_registration(self):
        manager = StrategyManager(
            self.handles,
            self.solution,
            weighted_jaccard_matrix={},
            strategy_preset="bottlenecks",
        )
        self.assertIn("frequently_refetched_skus", manager.active_strats)
        self.assertIn("cycle_time_weighted_refetched_skus", manager.active_strats)
        self.assertIn("longest_orders", manager.active_strats)

        res_refetched = manager.active_strats["frequently_refetched_skus"](1)
        self.assertTrue(len(res_refetched.seed_skus) > 0)

        res_longest = manager.active_strats["longest_orders"](1)
        self.assertTrue(len(res_longest.seed_orders) > 0)

    def test_edge_cases(self):
        # None solution
        res_none = strategy_longest_orders(self.handles, None, k=1)
        self.assertEqual(res_none.seed_orders, set())

        res_none_sku = strategy_frequently_refetched_skus(self.handles, None, k=1)
        self.assertEqual(res_none_sku.seed_skus, set())


if __name__ == "__main__":
    unittest.main()
