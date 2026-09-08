import json
import os
import pickle
import time
from datetime import datetime
from enum import IntEnum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import ticker
from matplotlib.colors import BoundaryNorm, ListedColormap

from schedule_visualizer import plot_schedule, write_html


class Status(IntEnum):
    Unknown = 1
    Optimal_No_Improve = 2
    Optimal_New_Best = 3
    Optimal_Improve = 4
    Feasible_No_Improve = 5
    Feasible_New_Best = 6
    Feasible_Improve = 7
    Feasible_Degradation = 8
    Infeasible = 9


class LoguruStream:
    def __init__(self, level="INFO"):
        self.level = level

    def write(self, message):
        cleaned = message.strip()
        if cleaned:
            logger.log(self.level, cleaned)

    def flush(self):
        pass


class VisualLogger:
    def __init__(self, instance_folder, instance_config, experiment_config):
        self.experiment_timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

        self.instance_folder = instance_folder
        self.instance_config = instance_config
        self.experiment_config = experiment_config
        self.path = f"{self.instance_folder}/experiments/{self.experiment_timestamp}"
        os.makedirs(self.path, exist_ok=True)

        self.logger_start_time = time.perf_counter()
        self.last_time = self.logger_start_time

        self.current_iteration = 0
        self.current_run = 0
        self.active_run_start = None
        self.active_iteration_start = None
        self.all_run_solutions = []

        # Data storage
        self.df = pd.DataFrame()
        self.barcodes = []
        self.current_run_data = {}
        self.current_barcode = None
        self.num_variables = 0

        # Strategy Tracking
        self.strategy_map = {}

        self.open_log = None
        logger.info("Started VisualLogger.")

    def time_from_here(self):
        self.last_time = time.perf_counter()

    def time_diff_with_overwrite(self):
        curr_time = time.perf_counter()
        diff = curr_time - self.last_time
        self.last_time = curr_time
        return diff

    def log_run_start(self, num_variables, var_to_idx, sp):
        self.current_run += 1
        self.all_run_solutions.append([])
        self.num_variables = num_variables
        self.open_log = logger.add(
            f"{self.path}/run_{self.current_run}.log",
            mode="w",
        )
        logger.info(f"Starting run #{self.current_run}")

        iters = self.experiment_config["iters"]

        # Pre-allocate run data
        self.current_run_data = {
            "run_id": np.full(iters, self.current_run, dtype=np.int32),
            "iteration": np.arange(iters, dtype=np.int32),
            "best": np.full(iters, -1, dtype=np.float64),
            "current": np.full(iters, -1, dtype=np.float64),
            "num_optimized": np.full(iters, -1, dtype=np.int32),
            "total_iter_time": np.full(iters, -1.0, dtype=np.float64),
            "lns_strategy_time": np.full(iters, -1.0, dtype=np.float64),
            "constr_update_time": np.full(iters, -1.0, dtype=np.float64),
            "constr_generate_time": np.full(iters, -1.0, dtype=np.float64),
            "solve_time": np.full(iters, -1.0, dtype=np.float64),
            "solve_api_overhead": np.full(iters, -1.0, dtype=np.float64),
            "solve_cpp_presolve": np.full(iters, -1.0, dtype=np.float64),
            "solve_cpp_search": np.full(iters, -1.0, dtype=np.float64),
            "statuses": np.full(iters, -1, dtype=np.int32),
            "severity_increases": np.zeros(iters, dtype=np.int8),
            "strategy": np.full(iters, -1, dtype=np.int32),
        }

        self.current_barcode = np.full((iters, num_variables), -1, dtype=np.int8)
        self.current_iteration = 0
        self.active_run_start = time.perf_counter()
        self.last_time = self.active_run_start

        # --- Active / Inactive Boundary Detection ---
        self.var_names = [""] * num_variables
        self.inactive_start_idx = num_variables

        for var, idx in var_to_idx.items():
            self.var_names[idx] = var.name if hasattr(var, "name") else var.get_name()

            if hasattr(sp, "BooleanValue"):
                is_active = sp.BooleanValue(var.pres)
            else:
                var_sol = sp.get_var_solution(var)
                is_active = var_sol is not None and var_sol.is_present()

            # The first absent variable marks the start of the inactive block
            if not is_active and idx < self.inactive_start_idx:
                self.inactive_start_idx = idx

        self.var_name_to_idx = {name: idx for idx, name in enumerate(self.var_names)}

        # --- Zebra Stripe Groupings ---
        self.inactive_groups = []
        if self.inactive_start_idx < num_variables:
            current_prefix = self.var_names[self.inactive_start_idx].split("[")[0]
            self.inactive_groups.append((current_prefix, self.inactive_start_idx))

            for i in range(self.inactive_start_idx + 1, num_variables):
                prefix = self.var_names[i].split("[")[0]

                if prefix != current_prefix:
                    # logger.debug(
                    # f"{i}: Var name {self.var_names[i]} prefix {prefix} current {current_prefix}"
                    # )
                    current_prefix = prefix
                    self.inactive_groups.append((current_prefix, i))
        # logger.debug(
        # f"Inactive groups: {len(self.inactive_groups)}, {random.sample(self.inactive_groups, 5)} {random.sample(self.var_names, 5)}"
        # )

    def log_run_end(self, solution, handles):
        # 1. Store run data
        run_df = pd.DataFrame(self.current_run_data)
        self.df = pd.concat([self.df, run_df], ignore_index=True)
        self.barcodes.append(self.current_barcode)

        # 2. Extract metrics for logging
        run_time = time.perf_counter() - self.active_run_start
        best_makespan = self.current_run_data["best"][-1]

        logger.info("Exporting visualization...")
        fig = plot_schedule(solution, handles, show=False)
        html_file = f"{self.path}/run{self.current_run}-schedule.html"
        write_html(fig, html_file)
        logger.info(f"\nWrote visualization to {html_file}")

        sol_dict = {}
        for var_sol in solution.get_all_var_solutions():
            val = var_sol.get_value()
            if hasattr(val, "is_present"):
                sol_dict[var_sol.get_name()] = {
                    "present": val.is_present(),
                    "start": val.get_start() if val.is_present() else None,
                    "end": val.get_end() if val.is_present() else None,
                }
            else:
                sol_dict[var_sol.get_name()] = val
        with open(f"{self.path}/run{self.current_run}-solution.pkl", "wb") as f:
            pickle.dump(sol_dict, f)

        # Attempt CP lookup — load all seeds and compute mean final best
        import glob

        cp_pattern = os.path.join(self.instance_folder, "cp_intermediate_records_*.pkl")
        cp_files = sorted(glob.glob(cp_pattern))
        cp_final_bests = []
        for fpath in cp_files:
            try:
                with open(fpath, "rb") as f:
                    records = pickle.load(f)
                    cp_final_bests.append(records[-1]["best"])
            except (FileNotFoundError, IndexError, pickle.UnpicklingError):
                continue

        if cp_final_bests:
            cp_1hour_best = float(np.mean(cp_final_bests))
        else:
            cp_1hour_best = 0.0

        gap_to_cp = (
            ((best_makespan - cp_1hour_best) / cp_1hour_best) * 100
            if cp_1hour_best > 0
            else 0
        )

        # 3. Clean Print
        logger.info(
            f"\n"
            f"=================================================\n"
            f"             RUN {self.current_run} COMPLETED\n"
            f"=================================================\n"
            f"Total Run Time : {run_time:.2f}s\n"
            f"Best Makespan  : {best_makespan:.1f}\n"
            f"CP Seeds Found : {len(cp_final_bests)}\n"
            f"Mean CP Best   : {cp_1hour_best:.1f}\n"
            f"Gap to CP      : {gap_to_cp:+.2f}%\n"
            f"=================================================\n"
        )

        # 4. Reset states
        self.active_iteration_start = None
        self.last_time = None
        logger.remove(self.open_log)

    def log_iteration_start(self, iteration):
        self.active_iteration_start = time.perf_counter()
        self.current_iteration = iteration

    def log_iteration(self):
        curr_time = time.perf_counter()

        if self.active_iteration_start is not None:
            self.add_stat_to_current(
                "total_iter_time", curr_time - self.active_iteration_start
            )

        # --- ITERATION LOGGING ---
        strat_time = self.current_run_data["lns_strategy_time"][self.current_iteration]
        constr_update_time = self.current_run_data["constr_update_time"][
            self.current_iteration
        ]
        constr_generate_time = self.current_run_data["constr_generate_time"][
            self.current_iteration
        ]
        solve_time = self.current_run_data["solve_time"][self.current_iteration]
        solve_api_overhead = self.current_run_data["solve_api_overhead"][
            self.current_iteration
        ]
        solve_cpp_presolve = self.current_run_data["solve_cpp_presolve"][
            self.current_iteration
        ]
        solve_cpp_search = self.current_run_data["solve_cpp_search"][
            self.current_iteration
        ]

        total_time = self.current_run_data["total_iter_time"][self.current_iteration]
        status_val = self.current_run_data["statuses"][self.current_iteration]
        best_res = self.current_run_data["best"][self.current_iteration]
        curr_res = self.current_run_data["current"][self.current_iteration]
        num_optimized = self.current_run_data["num_optimized"][self.current_iteration]
        strategy = self.current_run_data["strategy"][self.current_iteration]
        status_name = Status(status_val).name if status_val > 0 else "Unknown"

        logger.info(
            f"Run {self.current_run}/{self.experiment_config['runs']} | "
            f"Iter {self.current_iteration:03d}/{self.experiment_config['iters']} | "
            f"Current Severity: {np.sum(self.current_run_data['severity_increases'])} | "
            f"Optimized {num_optimized:06d}/{self.num_variables:06d} ({100 * num_optimized / self.num_variables:.3f})% | "
            f"Status: {status_name:22s} | "
            f"Best: {best_res:6.1f} | "
            f"Curr: {curr_res:6.1f} | "
            f"Strategy: {self.strategy_map[strategy]:20s}({strategy}) | "
            f"Times: Strat {strat_time:.3f}s, "
            f"Constr Gen {constr_generate_time:.3f}s, "
            f"Constr Update {constr_update_time:.3f}s, "
            f"Solve {solve_time:.3f}s, (API {solve_api_overhead:.3f}, Presolve {solve_cpp_presolve:.3f}, Search {solve_cpp_search:.3f})"
            f"Total {total_time:.3f}s"
        )

    def log_strategy(self, strategy_name: str, strategy_idx: int):
        self.add_stat_to_current("lns_strategy_time", self.time_diff_with_overwrite())
        self.strategy_map[strategy_idx] = strategy_name
        self.add_stat_to_current("strategy", strategy_idx)

    def log_freeze_constr(self, to_optimize, var_to_idx, strategy_idx):
        self.add_stat_to_current("constr_update_time", self.time_diff_with_overwrite())
        self.add_stat_to_current("num_optimized", len(to_optimize))
        for var in to_optimize:
            var_name = var.name if hasattr(var, "name") else var.get_name()
            if var_name in self.var_name_to_idx:
                var_idx = self.var_name_to_idx[var_name]
                self.current_barcode[self.current_iteration, var_idx] = strategy_idx

    def log_solve_time(self, sol, status=None):
        wall_solve_time = self.time_diff_with_overwrite()

        engine_total = sol.get_info("TotalTime") or sol.get_solve_time() or 0.0
        engine_search = sol.get_solve_time() or 0.0
        engine_extraction = sol.get_info("ExtractionTime") or max(
            0.0, engine_total - engine_search
        )

        # 2. API overhead is whatever time the C++ engine wasn't running
        python_overhead = max(0.0, wall_solve_time - engine_total)

        self.add_stat_to_current("solve_api_overhead", python_overhead)
        self.add_stat_to_current("solve_cpp_presolve", engine_extraction)
        self.add_stat_to_current("solve_cpp_search", engine_search)

        # Keep the total for legacy compatibility
        self.add_stat_to_current("solve_time", wall_solve_time)

    def add_stat_to_current(self, stat_name: str, value):
        # Explicit .value extraction prevents Enum Object type-casting failures in arrays
        if isinstance(value, IntEnum):
            value = value.value

        self.current_run_data[stat_name][self.current_iteration] = value

    def get_cp_performance(self, mth_mean_ticks):
        """Load all cp_intermediate_records_{seed}.pkl files and compute
        mean ± std CP trajectory (time-aligned) and mean ± std 1-hour best."""
        import glob

        pattern = os.path.join(self.instance_folder, "cp_intermediate_records_*.pkl")
        cp_files = sorted(glob.glob(pattern))

        if not cp_files:
            logger.warning("No CP records found. Skipping CP performance mapping.")
            return None, None, None, None

        # Load all seed records
        all_records = []
        for fpath in cp_files:
            try:
                with open(fpath, "rb") as f:
                    all_records.append(pickle.load(f))
            except (FileNotFoundError, pickle.UnpicklingError, IndexError) as e:
                logger.warning(f"Could not load {fpath}: {e}")
                continue

        if not all_records:
            logger.warning(
                "No valid CP records loaded. Skipping CP performance mapping."
            )
            return None, None, None, None

        # For each seed, interpolate the best value at each matheuristic time tick
        n_seeds = len(all_records)
        n_ticks = len(mth_mean_ticks)
        trajectory = np.full((n_seeds, n_ticks), np.nan, dtype=np.float64)
        final_bests = np.zeros(n_seeds, dtype=np.float64)

        for s, records in enumerate(all_records):
            bests = np.array([r["best"] for r in records], dtype=np.float64)
            final_bests[s] = bests[-1]

            # Interpolate: for each tick, find the last record whose time <= tick
            last_idx = 0
            for t_idx, tick in enumerate(mth_mean_ticks):
                while (
                    last_idx < len(records) - 1 and records[last_idx + 1]["time"] < tick
                ):
                    last_idx += 1
                trajectory[s, t_idx] = records[last_idx]["best"]

        # Compute mean and std across seeds
        mean_traj = np.nanmean(trajectory, axis=0)
        std_traj = np.nanstd(trajectory, axis=0, ddof=1)  # sample std
        mean_final = np.mean(final_bests)
        std_final = np.std(final_bests, ddof=1)

        return mean_traj, std_traj, mean_final, std_final

    def save_experiment(self):
        # 1. Save Configs and Data
        with open(f"{self.path}/experiment_config.json", "w+") as f:
            json.dump(self.experiment_config, f, indent=4)

        solutions_df = self.get_solution_history_df(self.all_run_solutions)
        solutions_df.to_pickle(f"{self.path}/solutions_dataframe.pkl")
        self.df.to_pickle(f"{self.path}/experiment_dataframe.pkl")

        with open(f"{self.path}/barcodes.pkl", "wb") as f:
            pickle.dump(self.barcodes, f)

        # 2. Generate Summary Text
        self._write_summary_file()

        # 3. Generate Plots
        self.plot_solver_progress()
        self.plot_unfrozen_ratio()
        self.plot_iter_statuses()
        self.plot_and_save_durations_hist()
        self.plot_neighborhood_barcode()
        self.plot_chronological_durations()
        self.plot_binned_durations_composition()
        self.plot_status_time_histogram()
        self.plot_solution_stability(self.all_run_solutions)
        self.plot_strategy_footprints()
        self.plot_status_time_distributions()
        logger.info(f"Experiment saved successfully to {self.path}")

    def get_solution_history_df(self, solution_history):
        """
        Converts the nested list[list[dict]] of solution states into a long-format DataFrame.
        """
        records = []
        for r_idx, run_sols in enumerate(solution_history):
            run_id = r_idx + 1
            for iter_idx, sol_dict in enumerate(run_sols):
                for var_name, state in sol_dict.items():
                    if isinstance(state, dict):
                        records.append(
                            (
                                run_id,
                                iter_idx,
                                var_name,
                                state.get("present"),
                                state.get("start"),
                                state.get("end"),
                                None,
                            )
                        )
                    else:
                        records.append(
                            (run_id, iter_idx, var_name, None, None, None, state)
                        )

        df = pd.DataFrame(
            records,
            columns=[
                "run_id",
                "iteration",
                "var_name",
                "present",
                "start",
                "end",
                "value",
            ],
        )

        # Optimize memory footprint
        df["run_id"] = df["run_id"].astype("int32")
        df["iteration"] = df["iteration"].astype("int32")

        # Robustly handle the float NaN to nullable Boolean conversion
        df["present"] = (
            df["present"]
            .apply(lambda x: pd.NA if pd.isna(x) else bool(x))
            .astype("boolean")
        )

        return df

    def _plot_severity_lines(self):
        """Helper to plot vertical lines wherever severity increased across any run."""
        severity_iters = self.df[self.df["severity_increases"] > 0][
            "iteration"
        ].unique()
        for i, siter in enumerate(severity_iters):
            label = "Severity Increase" if i == 0 else ""
            plt.axvline(x=siter, color="gray", linestyle=":", alpha=0.3, label=label)

    def plot_status_time_distributions(self):
        # Filter valid statuses
        valid_df = self.df[self.df["statuses"] > 0].copy()

        if valid_df.empty:
            return

        # Group the total_iter_time arrays by their status enum values
        groups = valid_df.groupby("statuses")

        data_to_plot = []
        labels = []

        for status_val, group in sorted(groups):
            data_to_plot.append(group["total_iter_time"].values)
            labels.append(Status(status_val).name)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Create the boxplot
        bplot = ax.boxplot(
            data_to_plot,
            labels=labels,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 2},
            boxprops={"edgecolor": "black"},
            whiskerprops={"linestyle": "--", "color": "black"},
            flierprops={
                "marker": "o",
                "markeredgecolor": "black",
                "markerfacecolor": "none",
            },
        )

        # Apply a distinct color map to the boxes
        colors = plt.cm.tab10.colors[: len(labels)]
        for patch, color in zip(bplot["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        ax.set_title(
            "Distribution of Total Iteration Times by Resulting Status", fontsize=14
        )
        ax.set_xlabel("Iteration Status", fontsize=12)
        ax.set_ylabel("Total Iteration Time (seconds)", fontsize=12)

        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g s"))

        # Rotate x-labels to prevent overlapping long enum names
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, alpha=0.3, axis="y", linestyle=":")

        fig.set_layout_engine("constrained")

        plt.savefig(
            f"{self.path}/status_time_distributions.svg",
            format="svg",
            bbox_inches="tight",
        )
        plt.close()

    def plot_solver_progress(self):
        plt.figure(figsize=(10, 6))

        gb = self.df.groupby("iteration")
        iters = list(gb.groups.keys())

        mean_best = gb["best"].mean()
        std_best = gb["best"].std().fillna(0)
        mean_time = gb["total_iter_time"].mean().cumsum()

        # Plot Best Makespan on Iteration Axis
        plt.plot(iters, mean_best, color="blue", label="Mean Best Solution")
        plt.fill_between(
            iters, mean_best - std_best, mean_best + std_best, color="blue", alpha=0.2
        )

        # CP Performance Comparison mapped to Mean Time ticks
        cp_mean, cp_std, cp_final_mean, cp_final_std = self.get_cp_performance(
            mean_time.values
        )

        if cp_mean is not None:
            # Time-aligned CP trajectory with std envelope
            plt.plot(
                iters,
                cp_mean,
                color="green",
                linestyle="--",
                label="CP Trajectory (Time-Aligned, Mean ± 1σ)",
            )
            plt.fill_between(
                iters,
                cp_mean - cp_std,
                cp_mean + cp_std,
                color="green",
                alpha=0.15,
            )

            # 1-hour baseline with std envelope
            if cp_final_mean > 0:
                plt.axhline(
                    y=cp_final_mean,
                    color="red",
                    linestyle=":",
                    label="CP 1-Hour Best (Mean ± 1σ)",
                )
                plt.axhline(
                    y=cp_final_mean - cp_final_std,
                    color="red",
                    linestyle=":",
                    alpha=0.3,
                )
                plt.axhline(
                    y=cp_final_mean + cp_final_std,
                    color="red",
                    linestyle=":",
                    alpha=0.3,
                )

        # Overlay Severity Lines
        self._plot_severity_lines()

        plt.title("Warmstarted Matheuristic vs CP Performance")
        plt.xlabel("Iteration")
        plt.ylabel("Makespan")
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)

        plt.savefig(
            f"{self.path}/solver_progress.svg", format="svg", bbox_inches="tight"
        )
        plt.close()

    def plot_unfrozen_ratio(self):
        plt.figure(figsize=(10, 6))

        self.df["unfrozen_ratio"] = self.df["num_optimized"] / self.num_variables

        gb = self.df.groupby("iteration")
        iters = list(gb.groups.keys())
        mean_ratio = gb["unfrozen_ratio"].mean()
        std_ratio = gb["unfrozen_ratio"].std().fillna(0)

        plt.plot(iters, mean_ratio, color="purple", label="Mean Unfrozen Ratio")
        plt.fill_between(
            iters,
            mean_ratio - std_ratio,
            mean_ratio + std_ratio,
            color="purple",
            alpha=0.2,
        )

        # 50-iteration simple moving average (min_periods=1 so the line
        # starts at iteration 0 with the partial window)
        sma_50 = mean_ratio.rolling(window=50, min_periods=1).mean()
        plt.plot(iters, sma_50, color="green", label="50-Iteration Moving Average")

        # Overlay Severity Lines
        self._plot_severity_lines()

        plt.xlabel("Iteration")
        plt.ylabel("Ratio of Unfrozen Variables")
        plt.title("Variable Optimization Ratio Across Runs")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig(
            f"{self.path}/unfrozen_ratios.svg", format="svg", bbox_inches="tight"
        )
        plt.close()

    def _write_summary_file(self):
        best_overall = self.df["best"].min()
        mean_solve_time = self.df.groupby("run_id")["solve_time"].sum().mean()

        # --- Strategy Statistics ---
        total_iters = len(self.df)
        strat_counts = self.df["strategy"].value_counts()

        # Count New Bests (Status 3 or 6)
        new_best_mask = self.df["statuses"].isin(
            [Status.Optimal_New_Best.value, Status.Feasible_New_Best.value]
        )
        success_counts = self.df[new_best_mask]["strategy"].value_counts()

        strat_stats_str = "\n=== Strategy Statistics ===\n"
        for strat_id, count in strat_counts.items():
            if strat_id == -1:
                continue  # Skip unassigned/frozen

            strat_name = self.strategy_map.get(strat_id, f"Strategy {strat_id}")
            successes = success_counts.get(strat_id, 0)
            usage_pct = (count / total_iters) * 100 if total_iters > 0 else 0
            success_rate = (successes / count) * 100 if count > 0 else 0

            strat_stats_str += (
                f"{strat_name}:\n"
                f"  Usage: {count} times ({usage_pct:.1f}% of all iterations)\n"
                f"  New Bests Found: {successes} ({success_rate:.1f}% success rate)\n"
            )

        summary = (
            f"=== Experiment Summary ===\n"
            f"Date: {self.experiment_timestamp}\n"
            f"Runs: {self.experiment_config['runs']}\n"
            f"Iterations per run: {self.experiment_config['iters']}\n"
            f"Overall Best Makespan: {best_overall}\n"
            f"Average Total Solve Time per run: {mean_solve_time:.2f}s\n"
            f"{strat_stats_str}"
        )

        with open(f"{self.path}/summary.txt", "w") as f:
            f.write(summary)

    def plot_iter_statuses(self):
        # Filter out unknown/uninitialized statuses
        valid_df = self.df[self.df["statuses"] > 0].copy()

        if valid_df.empty:
            return

        # Cross-tabulate statuses and strategies to get stacking data
        crosstab = pd.crosstab(valid_df["statuses"], valid_df["strategy"])

        # Map status values to enum names for the x-axis
        status_names = [Status(val).name for val in crosstab.index]
        crosstab.index = status_names

        # Map strategy IDs to names for the legend
        strat_names = [
            self.strategy_map.get(col, f"Strategy {col}") for col in crosstab.columns
        ]
        crosstab.columns = strat_names

        # Plot stacked bar chart directly from Pandas
        ax = crosstab.plot(
            kind="bar",
            stacked=True,
            figsize=(12, 7),
            colormap="tab10",
            edgecolor="black",
        )

        plt.xticks(rotation=45, ha="right")

        # Calculate totals per status to annotate the total percentage at the top of the stack
        totals = crosstab.sum(axis=1)
        total_iters = len(self.df)

        for i, (idx, total) in enumerate(totals.items()):
            if total > 0:
                pct = (total / total_iters) * 100
                ax.annotate(
                    f"{pct:.1f}%",
                    xy=(i, total),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        plt.title("Iteration Results by Strategy")
        plt.ylabel("Count")
        plt.legend(title="Strategies", bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        plt.savefig(
            f"{self.path}/iteration_results.svg", format="svg", bbox_inches="tight"
        )
        plt.close()

    def plot_neighborhood_barcode(self):
        if not self.barcodes:
            return

        for r in range(self.experiment_config["runs"]):
            matrix = self.barcodes[r]
            matrix_t = matrix.T

            # Build dynamic colormap mapping -1 to Grey, and Strategy Indices to Tab10 Colors
            max_strat = max(self.strategy_map.keys()) if self.strategy_map else 0
            colors = ["#e0e0e0"] + list(plt.cm.tab10.colors[: max_strat + 1])
            cmap = ListedColormap(colors)

            bounds = np.arange(-1.5, max_strat + 1.5, 1.0)
            norm = BoundaryNorm(bounds, cmap.N)

            fig, ax = plt.subplots(figsize=(12, 8))
            cax = ax.imshow(
                matrix_t, cmap=cmap, norm=norm, aspect="auto", interpolation="none"
            )

            # --- 1. Draw Active / Inactive Boundary Line ---
            if hasattr(self, "inactive_start_idx") and self.inactive_start_idx > 0:
                ax.axhline(
                    y=self.inactive_start_idx,
                    color="red",
                    linewidth=1,  # Thinner
                    linestyle="--",  # Dashed/fainter
                    alpha=0.6,  # More transparent
                    label="Active / Inactive Boundary",
                )

            # --- 2. Draw Faint Lines and Spaced Labels for Inactive Groups ---
            if hasattr(self, "inactive_groups") and self.inactive_groups:
                group_starts = [g[1] for g in self.inactive_groups]
                group_starts.append(self.num_variables)

                # Create a secondary y-axis for the prefix labels
                ax_right = ax.twinx()
                ax_right.set_ylim(ax.get_ylim())  # Match limits and inverted direction

                raw_yticks = []
                yticklabels = []

                for i in range(len(self.inactive_groups)):
                    prefix, start_idx = self.inactive_groups[i]
                    end_idx = group_starts[i + 1]

                    # Draw faint dotted line for segment boundaries (skip the first one overlapping red line)
                    if i > 0:
                        ax.axhline(
                            y=start_idx,
                            color="gray",
                            linewidth=0.5,
                            linestyle=":",
                            alpha=0.5,
                        )

                    # Calculate raw vertical center
                    center_idx = (start_idx + end_idx) / 2
                    raw_yticks.append(center_idx)
                    yticklabels.append(prefix)

                # --- Anti-Overlap Logic for Labels ---
                # Forces a minimum vertical gap between labels so 'B' and 'Block' don't overlap
                adjusted_yticks = [raw_yticks[0]]
                min_gap = (
                    self.num_variables * 0.04
                )  # Minimum 4% of total height between labels

                for i in range(1, len(raw_yticks)):
                    prev_y = adjusted_yticks[-1]
                    curr_y = raw_yticks[i]

                    # Since y-axis is inverted (0 at top, max at bottom), values increase going down
                    if curr_y - prev_y < min_gap:
                        adjusted_yticks.append(prev_y + min_gap)
                    else:
                        adjusted_yticks.append(curr_y)

                ax_right.set_yticks(adjusted_yticks)
                ax_right.set_yticklabels(yticklabels, fontsize=11)  # Larger font
                ax_right.set_ylabel("Inactive Variable Groups")
                ax_right.tick_params(
                    axis="y", length=0
                )  # Hide tick marks for a cleaner look

            # Update the title and axis labels
            ax.set_title(f"ALNS Variable Optimization Tracking (Run {r + 1})")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Variable Index")

            # Map y-ticks on colorbar to Strategy Names
            ticks = np.arange(-1, max_strat + 1)
            strategy_names = ["Frozen"] + [
                self.strategy_map.get(i, f"Strategy {i}") for i in range(max_strat + 1)
            ]

            # --- Explicitly attach colorbar to BOTH axes to prevent overlap ---
            axes_to_steal_from = (
                [ax, ax_right]
                if hasattr(self, "inactive_groups") and self.inactive_groups
                else ax
            )
            cbar = fig.colorbar(cax, ax=axes_to_steal_from, ticks=ticks, pad=0.08)
            cbar.ax.set_yticklabels(strategy_names)

            # Use layout engine instead of tight_layout for safer multi-axis bounding
            fig.set_layout_engine("constrained")

            # Save
            plt.savefig(
                f"{self.path}/neighborhood_barcode_run{r + 1}.svg",
                format="svg",
                bbox_inches="tight",
            )
            plt.close()

    def plot_and_save_durations_hist(self):
        plt.figure(figsize=(12, 7))

        # We stack the 6 components that make up the total iteration time
        components = [
            self.df["lns_strategy_time"],
            self.df["constr_generate_time"],
            self.df["constr_update_time"],
            self.df["solve_api_overhead"],
            self.df["solve_cpp_presolve"],
            self.df["solve_cpp_search"],
        ]

        labels = [
            "LNS Strategy Time",
            "Constraint Gen",
            "Constraint Update",
            "Python API Overhead",
            "C++ Engine Presolve",
            "C++ Search Time",
        ]

        # Use a distinct color palette
        colors = ["#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#1f77b4", "#5fffea"]

        plt.hist(
            components,
            bins=20,
            stacked=True,
            label=labels,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
        )

        plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter("%g s"))
        plt.xlabel("Duration (seconds)")
        plt.ylabel("Frequency")
        plt.title("Detailed Iteration Component Durations")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(
            f"{self.path}/detailed_durations_histogram.svg",
            format="svg",
            bbox_inches="tight",
        )
        plt.close()

    def plot_binned_durations_composition(self):
        if self.df.empty:
            return

        components = [
            "lns_strategy_time",
            "constr_generate_time",
            "constr_update_time",
            "solve_api_overhead",
            "solve_cpp_presolve",
            "solve_cpp_search",
        ]
        labels = [
            "LNS Strategy",
            "Constraint Gen",
            "Constraint Update",
            "Python API Overhead",
            "C++ Presolve",
            "C++ Search",
        ]
        colors = ["#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#1f77b4", "#5fffea"]

        # 1. Create bins based on the total iteration time
        max_time = self.df["total_iter_time"].max()
        # Create roughly 10 bins, handling edge cases where max_time is very small
        bins = np.linspace(0, max(max_time, 0.1), 11)

        # 2. Assign each iteration to a bin
        df_binned = self.df.copy()
        df_binned["time_bin"] = pd.cut(df_binned["total_iter_time"], bins=bins)

        # 3. Calculate the mean time of each component for iterations in each bin
        binned_means = (
            df_binned.groupby("time_bin", observed=False)[components].mean().fillna(0)
        )

        # 4. Plot as a stacked bar chart
        ax = binned_means.plot(
            kind="bar",
            stacked=True,
            figsize=(12, 7),
            color=colors,
            edgecolor="black",
            linewidth=0.5,
            width=0.95,
        )

        # Format X-axis labels to show the bin edges clearly
        bin_labels = [f"{b.left:.2f}s - {b.right:.2f}s" for b in binned_means.index]
        ax.set_xticklabels(bin_labels, rotation=45, ha="right")

        plt.xlabel("Total Iteration Duration Bins")
        plt.ylabel("Average Component Duration (seconds)")
        plt.title("Average Component Breakdown by Total Iteration Time")
        plt.legend(labels, bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(
            f"{self.path}/binned_durations_composition.svg",
            format="svg",
            bbox_inches="tight",
        )
        plt.close()

    def plot_status_time_histogram(self):
        # Filter valid statuses to ignore uninitialized (-1) rows
        valid_df = self.df[self.df["statuses"] > 0].copy()

        if valid_df.empty:
            return

        # Group the data by status
        groups = valid_df.groupby("statuses")

        data_to_plot = []
        labels = []

        # Extract the total_iter_time arrays for each status and their enum names
        for status_val, group in sorted(groups):
            data_to_plot.append(group["total_iter_time"].values)
            # Map the integer value back to the Status enum name
            labels.append(Status(status_val).name)

        plt.figure(figsize=(12, 7))

        # Dynamically grab enough colors for the present statuses
        colors = plt.cm.tab10.colors[: len(labels)]

        # Plot the stacked histogram
        plt.hist(
            data_to_plot,
            bins=20,
            stacked=True,
            label=labels,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
        )

        plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter("%g s"))
        plt.xlabel("Total Iteration Duration (seconds)")
        plt.ylabel("Frequency (Count)")
        plt.title("Distribution of Total Iteration Times by Resulting Status")

        # Move legend outside the plot
        plt.legend(title="Iteration Status", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3, axis="y")

        # Use layout engine instead of tight_layout for safer bounding
        plt.gcf().set_layout_engine("constrained")

        plt.savefig(
            f"{self.path}/status_time_histogram.svg",
            format="svg",
            bbox_inches="tight",
        )
        plt.close()

    def plot_chronological_durations(self):
        if self.df.empty:
            return

        components = [
            "lns_strategy_time",
            "constr_generate_time",
            "constr_update_time",
            "solve_api_overhead",
            "solve_cpp_presolve",
            "solve_cpp_search",
        ]
        labels = [
            "LNS Strategy",
            "Constraint Gen",
            "Constraint Update",
            "Python API Overhead",
            "C++ Presolve",
            "C++ Search",
        ]
        colors = ["#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#1f77b4", "#5fffea"]

        plt.figure(figsize=(12, 7))

        # 1. Group by iteration and average across ALL runs
        column_mapping = dict(zip(components, labels))
        gb_means = (
            self.df.groupby("iteration")[components]
            .mean()
            .rename(columns=column_mapping)
        )

        # 2. Plot the mean area trajectory
        ax = gb_means.plot.area(color=colors, figsize=(12, 7), linewidth=0, alpha=0.8)

        plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter("%g s"))
        plt.xlabel("Iteration Number")
        plt.ylabel("Mean Duration Across Runs (seconds)")
        plt.title("Mean Chronological Component Durations Over Time")

        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)

        time_limit = self.experiment_config.get("iter_time_limit", 2.0)
        plt.axhline(
            y=time_limit,
            color="red",
            linestyle="--",
            alpha=0.5,
            label=f"Time Limit ({time_limit}s)",
        )

        plt.tight_layout()
        plt.savefig(
            f"{self.path}/chronological_durations.svg",
            format="svg",
            bbox_inches="tight",
        )
        plt.close()

    def plot_strategy_footprints(self):
        if not self.barcodes:
            return

        for r in range(self.experiment_config["runs"]):
            matrix = self.barcodes[r]

            # Fetch the sequence of strategies used in this run
            run_strategies = self.df[self.df["run_id"] == r + 1]["strategy"].values

            # Identify which strategies were actually used
            used_strats = sorted(list(set(run_strategies) - {-1}))
            k = len(used_strats)

            if k == 0:
                continue

            # Map each strategy to its active iteration indices
            active_iters_dict = {
                s: np.where(run_strategies == s)[0] for s in used_strats
            }

            # Calculate proportional widths for the subplots based on usage count
            width_ratios = [max(1, len(active_iters_dict[s])) for s in used_strats]

            # Create side-by-side subplots (thin vertical subgraphs)
            fig, axes = plt.subplots(
                nrows=1,
                ncols=k,
                figsize=(14, 8),
                sharey=True,
                gridspec_kw={"width_ratios": width_ratios, "wspace": 0.05},
            )

            if k == 1:
                axes = [axes]

            for i, (ax, strat_idx) in enumerate(zip(axes, used_strats)):
                strat_name = self.strategy_map.get(strat_idx, f"Strategy {strat_idx}")
                strat_color = plt.cm.tab10.colors[strat_idx % 10]
                active_iters = active_iters_dict[strat_idx]

                # Slice the matrix to only include this strategy's iterations and transpose
                strat_matrix_t = matrix[active_iters, :].T

                # 0 = Frozen (Grey), 1 = Unfrozen by this strategy (Solid Color)
                footprint = np.zeros_like(strat_matrix_t, dtype=int)
                footprint[strat_matrix_t == strat_idx] = 1

                cmap = ListedColormap(["#e0e0e0", strat_color])
                bounds = [-0.5, 0.5, 1.5]
                norm = BoundaryNorm(bounds, cmap.N)

                ax.imshow(
                    footprint, cmap=cmap, norm=norm, aspect="auto", interpolation="none"
                )

                # --- Context Lines ---
                if hasattr(self, "inactive_start_idx") and self.inactive_start_idx > 0:
                    ax.axhline(
                        y=self.inactive_start_idx,
                        color="red",
                        linewidth=1,
                        linestyle="--",
                        alpha=0.6,
                    )

                if hasattr(self, "inactive_groups") and self.inactive_groups:
                    for j in range(1, len(self.inactive_groups)):
                        start_idx = self.inactive_groups[j][1]
                        ax.axhline(
                            y=start_idx,
                            color="gray",
                            linewidth=0.5,
                            linestyle=":",
                            alpha=0.5,
                        )

                # --- X-Ticks (Original Iteration Numbers) ---
                # Show up to 4 tick marks per subgraph so narrow strips aren't cluttered
                num_ticks = min(4, len(active_iters))
                if num_ticks > 0:
                    tick_positions = np.linspace(
                        0, len(active_iters) - 1, num_ticks, dtype=int
                    )
                    ax.set_xticks(tick_positions)
                    ax.set_xticklabels(
                        [active_iters[pos] for pos in tick_positions],
                        rotation=45,
                        fontsize=8,
                    )
                else:
                    ax.set_xticks([])

                # Title for each vertical subgraph
                ax.set_title(
                    strat_name, fontsize=10, rotation=45, ha="left", va="bottom"
                )

                # Only label the Y-axis on the very first subplot
                if i == 0:
                    ax.set_ylabel("Variable Index")

                # Add the inactive group labels ONLY to the very last subplot
                if (
                    i == k - 1
                    and hasattr(self, "inactive_groups")
                    and self.inactive_groups
                ):
                    group_starts = [g[1] for g in self.inactive_groups]
                    group_starts.append(self.num_variables)

                    ax_right = ax.twinx()
                    ax_right.set_ylim(ax.get_ylim())

                    raw_yticks = []
                    yticklabels = []

                    for j in range(len(self.inactive_groups)):
                        prefix, start_idx = self.inactive_groups[j]
                        end_idx = group_starts[j + 1]
                        center_idx = (start_idx + end_idx) / 2
                        raw_yticks.append(center_idx)
                        yticklabels.append(prefix)

                    adjusted_yticks = [raw_yticks[0]]
                    min_gap = self.num_variables * 0.04

                    for j in range(1, len(raw_yticks)):
                        prev_y = adjusted_yticks[-1]
                        curr_y = raw_yticks[j]
                        if curr_y - prev_y < min_gap:
                            adjusted_yticks.append(prev_y + min_gap)
                        else:
                            adjusted_yticks.append(curr_y)

                    ax_right.set_yticks(adjusted_yticks)
                    ax_right.set_yticklabels(yticklabels, fontsize=11)
                    ax_right.tick_params(axis="y", length=0)
                    ax_right.set_ylabel("Inactive Variable Groups")

            # Main figure labels and layout
            fig.suptitle(
                f"Strategy Footprints (Active Iterations Only) - Run {r + 1}",
                fontsize=14,
                y=1.05,
            )
            fig.text(0.5, -0.05, "Actual Iteration Number", ha="center", fontsize=12)

            fig.set_layout_engine("constrained")

            plt.savefig(
                f"{self.path}/strategy_footprints_run{r + 1}.svg",
                format="svg",
                bbox_inches="tight",
            )
            plt.close()

    def plot_solution_stability(self, solution_history):
        """
        Calculates raw inter-iteration variable changes and plots them,
        marking iterations where a new best objective was found.
        """
        diff_records = []

        # 1. Fast Dictionary Comparison for Raw Diffs
        for r_idx, run_sols in enumerate(solution_history):
            run_id = r_idx + 1
            for iter_idx, curr_sol in enumerate(run_sols):
                if iter_idx == 0:
                    changed_count = 0
                else:
                    prev_sol = run_sols[iter_idx - 1]
                    # Count variables whose state dictionary or value does not perfectly match the previous iteration
                    changed_count = sum(
                        1 for k, v in curr_sol.items() if prev_sol.get(k) != v
                    )

                diff_records.append(
                    {
                        "run_id": run_id,
                        "iteration": iter_idx,
                        "changed_vars": changed_count,
                    }
                )

        diff_df = pd.DataFrame(diff_records)

        # 2. Merge with VisualLogger's internal tracking dataframe
        merged_df = pd.merge(self.df, diff_df, on=["run_id", "iteration"], how="inner")

        # 3. Plot per Run
        for run_id in merged_df["run_id"].unique():
            run_data = merged_df[merged_df["run_id"] == run_id].copy()

            fig, ax = plt.subplots(figsize=(12, 6))

            # Plot Raw Changed Variables
            ax.plot(
                run_data["iteration"],
                run_data["changed_vars"],
                color="#1f77b4",
                label="Changed Variables",
                linewidth=2,
            )
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Number of Changed Variables")

            # Add vertical lines for New Best solutions
            new_best_mask = run_data["statuses"].isin(
                [Status.Optimal_New_Best.value, Status.Feasible_New_Best.value]
            )
            new_best_iters = run_data[new_best_mask]["iteration"]

            for i, iter_val in enumerate(new_best_iters):
                label = "New Best Found" if i == 0 else ""
                ax.axvline(
                    x=iter_val, color="#2ca02c", linestyle="--", alpha=0.7, label=label
                )

            ax.legend(loc="upper right")
            plt.title(f"Solution Stability and Improvements (Run {run_id})")
            plt.grid(True, alpha=0.3)
            fig.set_layout_engine("constrained")

            plt.savefig(
                f"{self.path}/solution_stability_run{run_id}.svg",
                format="svg",
                bbox_inches="tight",
            )
            plt.close()
