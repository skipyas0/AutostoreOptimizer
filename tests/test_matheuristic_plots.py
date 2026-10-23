import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matheuristic_plots import (
    Status,
    VisualLogger,
    extract_discovered_solutions,
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
        expected_footprints_path = os.path.join(
            vlg.path, "strategy_footprints_run1.svg"
        )
        self.assertTrue(os.path.exists(expected_footprints_path))

        # Test status distribution plots
        vlg.plot_distribution_of_times_per_status()
        vlg.plot_status_time_histogram()
        self.assertTrue(
            os.path.exists(os.path.join(vlg.path, "status_time_distributions.svg"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(vlg.path, "status_time_histogram.svg"))
        )

    def test_extract_discovered_solutions(self):
        # No solutions -> empty list (covers infeasible/unknown subproblems)
        self.assertEqual(extract_discovered_solutions(None), [])
        self.assertEqual(extract_discovered_solutions([]), [])

        # C++ backend raw progress dicts
        records = [
            {"time": 0.01, "best": 100, "kpis": {"makespan": 100, "bin_fetches": 3}},
            {"time": 0.5, "best": 90, "kpis": {"makespan": 90, "bin_fetches": 3}},
        ]
        out = extract_discovered_solutions(records)
        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(out[0]["time"], 0.01)
        self.assertEqual(out[0]["kpis"]["makespan"], 100.0)
        self.assertEqual(out[1]["kpis"]["bin_fetches"], 3.0)

        # docplex backend CpoSolveResult-like objects
        class FakeSolveResult:
            def get_solve_time(self):
                return 0.2

            def get_kpis(self):
                return {"makespan": 55, "idle_time": 12}

        out = extract_discovered_solutions([FakeSolveResult()])
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0]["time"], 0.2)
        self.assertEqual(out[0]["kpis"], {"makespan": 55.0, "idle_time": 12.0})

    def test_discovered_solutions_logging_and_plot(self):
        num_iters = 6
        vlg = VisualLogger(
            instance_folder=self.test_dir,
            instance_config={},
            experiment_config={"runs": 1, "iters": num_iters, "time_limit": 1.0},
        )
        vlg.log_run_start(10, {f"var_{i}": i for i in range(10)}, None)

        for i in range(num_iters):
            vlg.log_iteration_start(i)
            vlg.log_strategy("Strategy_0", 0)
            if i == 0:
                # Infeasible/unknown iteration -> empty list
                vlg.log_discovered_solutions([])
            else:
                records = [
                    {"time": 0.01 * j, "kpis": {"makespan": 100 - j}} for j in range(i)
                ]
                vlg.log_discovered_solutions(records)
            vlg.log_iteration_end(Status.Feasible_New_Best, best=100 - i, cur=100)

        vlg.log_run_end()

        self.assertEqual(
            vlg.df["num_discovered_solutions"].tolist(), [0, 1, 2, 3, 4, 5]
        )
        self.assertEqual(vlg.df["discovered_solutions"].iloc[0], [])

        vlg.plot_discovered_solutions_distribution()
        self.assertTrue(
            os.path.exists(
                os.path.join(vlg.path, "discovered_solutions_distribution.svg")
            )
        )
        self.assertTrue(
            os.path.exists(
                os.path.join(vlg.path, "discovered_solutions_vs_iter_time.svg")
            )
        )

    def test_upper_bound_and_intermediate_barcodes_on_disk(self):
        # When iters is None and time_limit is set, upper bound should be calculated reasonably
        time_limit = 10.0
        iter_time_limit = 2.0
        vlg = VisualLogger(
            instance_folder=self.test_dir,
            instance_config={},
            experiment_config={
                "runs": 2,
                "iters": None,
                "time_limit": time_limit,
                "iter_time_limit": iter_time_limit,
            },
        )
        num_vars = 20
        var_to_idx = {f"var_{i}": i for i in range(num_vars)}

        vlg.log_run_start(num_vars, var_to_idx, None)
        # Upper bound should not be the massive default of 100,000
        self.assertLess(vlg.current_capacity, 1000)
        self.assertGreaterEqual(vlg.current_capacity, 100)

        for i in range(5):
            vlg.log_iteration_start(i)
            vlg.log_strategy("Strategy_0", 0)
            vlg.log_freeze_constr([f"var_{i}"], var_to_idx, 0)
            vlg.log_iteration_end(Status.Optimal_Improve, best=100 - i, cur=100)

        vlg.log_run_end()

        # Intermediate barcode must be saved on disk as .npy
        expected_npy = os.path.join(vlg.path, "barcode_run_1.npy")
        self.assertTrue(os.path.exists(expected_npy))

        # In-memory current_barcode must be released
        self.assertIsNone(vlg.current_barcode)

        # Barcode access via LazyBarcodeList should dynamically load the array
        self.assertEqual(len(vlg.barcodes), 1)
        matrix = vlg.barcodes[0]
        self.assertEqual(matrix.shape, (5, num_vars))
        self.assertEqual(matrix[0, 0], 0)
        self.assertEqual(matrix[0, 1], -1)

        # Run 2
        vlg.log_run_start(num_vars, var_to_idx, None)
        for i in range(3):
            vlg.log_iteration_start(i)
            vlg.log_strategy("Strategy_0", 0)
            vlg.log_freeze_constr([f"var_{i}"], var_to_idx, 0)
            vlg.log_iteration_end(Status.Optimal_Improve, best=90 - i, cur=90)
        vlg.log_run_end()

        expected_npy_2 = os.path.join(vlg.path, "barcode_run_2.npy")
        self.assertTrue(os.path.exists(expected_npy_2))
        self.assertEqual(len(vlg.barcodes), 2)
        self.assertEqual(vlg.barcodes[1].shape, (3, num_vars))

    def test_dynamic_capacity_expansion(self):
        # Start with a very small initial capacity (3 iters) and log 10 iterations
        vlg = VisualLogger(
            instance_folder=self.test_dir,
            instance_config={},
            experiment_config={"runs": 1, "iters": 3, "time_limit": None},
        )
        num_vars = 10
        var_to_idx = {f"var_{i}": i for i in range(num_vars)}

        vlg.log_run_start(num_vars, var_to_idx, None)
        self.assertEqual(vlg.current_capacity, 3)

        # Log 10 iterations (exceeding initial capacity 3)
        for i in range(10):
            vlg.log_iteration_start(i)
            vlg.log_strategy("Strategy_0", 0)
            vlg.log_freeze_constr([f"var_{i % num_vars}"], var_to_idx, 0)
            vlg.log_iteration_end(Status.Optimal_Improve, best=100 - i, cur=100)

        # Capacity should have expanded automatically
        self.assertGreaterEqual(vlg.current_capacity, 10)
        vlg.log_run_end()

        # Final saved barcode should have 10 rows
        matrix = vlg.barcodes[0]
        self.assertEqual(matrix.shape, (10, num_vars))


if __name__ == "__main__":
    unittest.main()

