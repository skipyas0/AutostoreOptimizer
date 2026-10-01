import os
import sys
import unittest
import numpy as np
import pandas as pd
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matheuristic_plots import (
    VisualLogger,
    Status,
    get_distinct_colors,
)


class TestMatheuristicPlotsColorPalettes(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_get_distinct_colors(self):
        # Test various counts of colors
        for n in [1, 5, 10, 11, 15, 20, 25, 45, 65]:
            colors = get_distinct_colors(n)
            self.assertEqual(len(colors), n, f"Failed for n={n}")
            # Ensure elements are valid color tuples or strings
            for c in colors:
                self.assertTrue(isinstance(c, (tuple, list, str)))

    def test_plots_with_many_strategies(self):
        # Simulate an experiment run with 15 strategies (indices 0..14)
        num_strats = 15
        num_iters = 30
        num_vars = 50

        vlg = VisualLogger(
            instance_folder=self.test_dir,
            instance_config={},
            experiment_config={"runs": 1, "iters": num_iters, "time_limit": 1.0},
        )

        var_to_idx = {f"var_{i}": i for i in range(num_vars)}
        vlg.log_run_start(num_vars, var_to_idx, None)

        # Register 15 strategies and iterations
        for i in range(num_iters):
            strat_idx = i % num_strats
            strat_name = f"Strategy_{strat_idx}"
            vlg.log_iteration_start(i)
            vlg.log_strategy(strat_name, strat_idx)
            # Unfreeze some variables
            to_optimize = [f"var_{v}" for v in range(strat_idx, strat_idx + 5)]
            vlg.log_freeze_constr(to_optimize, var_to_idx, strat_idx)
            vlg.log_iteration_end(Status.Optimal_Improve, best=100 - i, cur=100)

        vlg.log_run_end()

        # Test plot_neighborhood_barcode (previously crashed when max_strat >= 10)
        vlg.plot_neighborhood_barcode()
        expected_barcode_path = os.path.join(vlg.path, "neighborhood_barcode_run1.svg")
        self.assertTrue(os.path.exists(expected_barcode_path))

        # Test plot_iter_statuses (previously restricted to colormap="tab10")
        vlg.plot_iter_statuses()
        expected_results_path = os.path.join(vlg.path, "iteration_results.svg")
        self.assertTrue(os.path.exists(expected_results_path))

        # Test plot_strategy_footprints (previously used % 10 color cycling)
        vlg.plot_strategy_footprints()
        expected_footprints_path = os.path.join(vlg.path, "strategy_footprints_run1.svg")
        self.assertTrue(os.path.exists(expected_footprints_path))

        # Test status distribution plots
        vlg.plot_distribution_of_times_per_status()
        vlg.plot_status_time_histogram()
        self.assertTrue(os.path.exists(os.path.join(vlg.path, "status_time_distributions.svg")))
        self.assertTrue(os.path.exists(os.path.join(vlg.path, "status_time_histogram.svg")))


if __name__ == "__main__":
    unittest.main()
