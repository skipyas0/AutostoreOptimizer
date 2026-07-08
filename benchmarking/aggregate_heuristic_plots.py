#!/usr/bin/env python3
"""
Aggregates and plots heuristic benchmark results.
Generates:
  - Comparison plots (Objective vs Parameter) for multiple heuristics.
  - Runtime comparison plots.
  - CSV summary tables.
"""

import argparse
import glob
import json
import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt


def load_results(input_dirs):
    """
    Load all JSON results from provided directories into a pandas DataFrame.
    """
    records = []

    files = []
    for d in input_dirs:
        # Recursive search for .json
        files.extend(glob.glob(os.path.join(d, "*.json")))

    if not files:
        print("No JSON files found!")
        return pd.DataFrame()

    print(f"Loading {len(files)} result files...")

    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)

            meta = data.get("meta", {})
            res = data.get("result", {})
            cfg = data.get("config", {})

            # Support both new heuristic runner and old benchmark_v4_single format
            # New runner uses "module" in meta. Old uses "heuristic_module" or "mode".
            module = meta.get("module") or meta.get("heuristic_module") or meta.get("mode")
            if module == "cp":
                module = f"CP-{meta.get('model_version', 'v5')}"

            records.append({
                "module": module,
                "param": meta.get("parameter"),
                "value": meta.get("value"),
                "seed": meta.get("seed"),
                "objective": res.get("objective_value"),
                "time": res.get("solve_time"),
                "status": res.get("status"),
                "filename": os.path.basename(fpath)
            })
            if 1790 > res.get("solve_time") > 400:
                print(f"Warning: {fpath} has unusually long solve time: {res.get('solve_time')}s")
        except Exception as e:
            print(f"Skipping {fpath}: {e}")
            continue

    df = pd.DataFrame(records)
    # Convert types
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["objective"] = pd.to_numeric(df["objective"], errors="coerce")
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    # Ensure seed is numeric to avoid mismatch during merge (e.g. "1" vs 1)
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce")

    return df


def generate_summary_csv(df, output_path):
    """
    Generate a summary CSV with median/min/max stats per (module, param, value).
    """
    if df.empty:
        return

    # Group by
    grouped = df.groupby(["module", "param", "value"])

    summary = grouped.agg(
        n_runs=("seed", "count"),

        obj_median=("objective", "median"),
        obj_min=("objective", "min"),
        obj_max=("objective", "max"),

        time_median=("time", "median"),
        time_mean=("time", "mean")
    ).reset_index()

    summary.to_csv(output_path, index=False)
    print(f"Saved summary CSV to {output_path}")


def plot_parameter_scaling(df, param, output_dir):
    """
    Generate Plotly and Matplotlib plots for a specific parameter (e.g. 'orders').
    """
    # Filter for this parameter
    subset = df[df["param"] == param].copy()
    if subset.empty:
        return

    # Sort by value for line plots
    subset.sort_values(by="value", inplace=True)

    # ---------------------------------------------------------
    # 1. Objective Value Plot (Makespan)
    # ---------------------------------------------------------

    # Aggregate for plotting (median)
    agg = subset.groupby(["module", "value"]).agg(
        obj_median=("objective", "median"),
        obj_min=("objective", "min"),
        obj_max=("objective", "max"),
        time_median=("time", "median")
    ).reset_index()

    # --- Plotly (Interactive) ---
    fig = px.line(
        agg,
        x="value",
        y="obj_median",
        color="module",
        markers=True,
        title=f"Scaling: {param} vs Objective (Makespan)",
        labels={"obj_median": "Median Makespan", "value": f"{param}"}
    )

    # Add bands? Maybe later. For now, just median lines are cleaner for comparison.

    out_html = os.path.join(output_dir, f"plot_{param}_objective.html")
    fig.write_html(out_html)
    print(f"Saved {out_html}")

    # --- Matplotlib (Static) ---
    plt.figure(figsize=(10, 6))
    for mod in agg["module"].unique():
        data = agg[agg["module"] == mod]
        plt.plot(data["value"], data["obj_median"], marker='o', label=mod)
        # Optional: fill between min/max
        # plt.fill_between(data["value"], data["obj_min"], data["obj_max"], alpha=0.1)

    plt.xlabel(param.capitalize())
    plt.ylabel("Median Makespan")
    plt.title(f"Scaling: {param} vs Objective")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_png = os.path.join(output_dir, f"plot_{param}_objective.png")
    plt.savefig(out_png)
    plt.close()
    print(f"Saved {out_png}")

    # ---------------------------------------------------------
    # 2. Runtime Plot
    # ---------------------------------------------------------

    # --- Plotly ---
    fig_time = px.line(
        agg,
        x="value",
        y="time_median",
        color="module",
        markers=True,)
    if px is not None:
        fig_time = px.line(
            agg,
            x="value",
            y="time_median",
            color="module",
            markers=True,
            log_y=False,  # Linear scale as requested
            title=f"Scaling: {param} vs Runtime",
            labels={"time_median": "Median Runtime [s]", "value": f"{param}"}
        )
        out_html_time = os.path.join(output_dir, f"plot_{param}_time.html")
        fig_time.write_html(out_html_time)
        print(f"Saved {out_html_time}")

    # --- Matplotlib ---
    plt.figure(figsize=(10, 6))
    for mod in agg["module"].unique():
        data = agg[agg["module"] == mod]
        plt.plot(data["value"], data["time_median"], marker='s', linestyle='--', label=mod)

    plt.xlabel(param.capitalize())
    plt.ylabel("Median Runtime [s]")
    # plt.yscale("log") # Linear scale as requested
    plt.title(f"Scaling: {param} vs Runtime")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_png_time = os.path.join(output_dir, f"plot_{param}_time.png")
    plt.savefig(out_png_time)
    plt.close()


def compute_best_baselines(df_cp):
    """
    Compute Best Known Objective and its solve time for each (param, value, seed) from CP logs.
    Returns: DataFrame with index (param, value, seed) and cols 'best_cp_obj', 'best_cp_time'.
    """
    # Filter for valid solutions
    valid = df_cp[df_cp["objective"].notnull()].copy()
    if valid.empty:
        return pd.DataFrame()

    # Find the row with the minimum objective for each group
    # If multiple rows have same min objective, take the one with min time
    # This ensures we get the runtime associated with the best objective
    best_idx = valid.sort_values("time").groupby(["param", "value", "seed"])["objective"].idxmin()
    best = valid.loc[best_idx, ["param", "value", "seed", "objective", "time"]].copy()

    best.rename(columns={"objective": "best_cp_obj", "time": "best_cp_time"}, inplace=True)
    return best


def print_beating_cp_summary(merged_df):
    """
    Print a summary of runs where Heuristic objective is strictly better than Best CP.
    """
    # Calculate Gap if not present
    if "gap_pct" not in merged_df.columns:
        merged_df["gap_pct"] = 100.0 * (merged_df["objective"] - merged_df["best_cp_obj"]) / merged_df["best_cp_obj"]

    # Filter for Gap < -0.01
    better = merged_df[merged_df["gap_pct"] < -0.01].copy()

    print("\n" + "="*80)
    print("  SUMMARY: HEURISTIC vs CP BASELINE")
    print("="*80)

    # Calculate overall stats
    total_comparisons = len(merged_df)
    total_better = len(better)
    pct_better = (total_better / total_comparisons * 100.0) if total_comparisons > 0 else 0.0

    print(f"Total Runs Compared: {total_comparisons}")
    print(f"Runs strictly better than CP: {total_better} ({pct_better:.1f}%)")

    # ---------------------------------------------------------
    # Breakdown by Parameter
    # ---------------------------------------------------------
    if not better.empty:
        print("\n--- Breakdown of Better Runs by Parameter ---")

        # Sort by gap (most negative first)
        better_sorted = better.sort_values("gap_pct")

        # Process each parameter found in the better set
        for param in better_sorted["param"].unique():
            # Get subset of ALL merged runs (to calculate total for this param)
            m_p = merged_df[merged_df["param"] == param]
            # Get subset of BETTER runs
            b_p = better_sorted[better_sorted["param"] == param]

            n_total = len(m_p)
            n_better = len(b_p)
            pct = 100.0 * n_better / n_total if n_total > 0 else 0.0

            print(f"\nParameter '{param}': {n_better}/{n_total} runs ({pct:.1f}%) surpassed CP Baseline:")

            for _, row in b_p.iterrows():
                val_fmt = f"{row['param']}={row['value']}"
                print(f"  - {val_fmt}, seed={row['seed']} -> Gap: {row['gap_pct']:.2f}% "
                      f"(Heur: {row['objective']:.1f} vs CP: {row['best_cp_obj']:.1f})")
    else:
        print("\nNo heuristic runs were strictly better than the CP baseline in this batch.")

    print("="*80 + "\n")


def plot_performance_profiles(df, df_cp, output_dir):
    # --- Performance Profile (Dolan-More) ---
    baselines_df = compute_best_baselines(df_cp)

    if baselines_df.empty:
        print("Warning: No valid CP baselines found in provided logs.")
        return

    # Check for missing parameters in baselines
    heur_params = set(df["param"].unique())
    cp_params = set(baselines_df["param"].unique())
    missing = heur_params - cp_params
    if missing:
        print(
            f"\n[WARNING] The following parameters are present in Heuristic results but MISSING in CP Baselines: {missing}")
        print("  -> These will NOT appear in the Scatter/Gap plots.")

    # Check for missing values within shared parameters (e.g. orders 250+ missing in CP)
    for param in heur_params:
        if param not in cp_params:
            continue

        heur_vals = set(df[df["param"] == param]["value"])
        cp_vals = set(baselines_df[baselines_df["param"] == param]["value"])

        missing_vals = heur_vals - cp_vals
        if missing_vals:
            # Sort for cleaner display
            sorted_missing = sorted(list(missing_vals))
            example_missing = sorted_missing[:5]
            print(
                f"\n[WARNING] For parameter '{param}', {len(missing_vals)} values in Heuristic results have NO CP BASELINE.")
            print(f"  -> Missing CP baselines for values: {example_missing} ...")
            # print(f"  -> {len(sorted_missing) * 5} heuristic runs will be excluded from Gap plots.")

    # --- Check for Seed Mismatch before merge ---
    # Identify keys (param, value) present in both but with disjoint seeds
    common_pv = pd.merge(
        df[["param", "value"]].drop_duplicates(),
        baselines_df[["param", "value"]].drop_duplicates(),
        on=["param", "value"]
    )

    mismatch_count = 0
    for _, row in common_pv.iterrows():
        p, v = row["param"], row["value"]
        h_seeds = set(df[(df["param"] == p) & (df["value"] == v)]["seed"])
        c_seeds = set(baselines_df[(baselines_df["param"] == p) & (baselines_df["value"] == v)]["seed"])

        if not h_seeds.intersection(c_seeds):
            mismatch_count += 1
            if mismatch_count <= 5:
                # Truncate sets for cleaner printing
                h_str = str(sorted(list(h_seeds))[:5]) + ("..." if len(h_seeds) > 5 else "")
                c_str = str(sorted(list(c_seeds))[:5]) + ("..." if len(c_seeds) > 5 else "")
                print(f"[WARNING] Seed Mismatch for {p}={v}: Heur seeds {h_str} vs CP seeds {c_str}")

    if mismatch_count > 0:
        print(
            f"\n[WARNING] Found {mismatch_count} (param, value) pairs with NO matching seeds between Heuristic and CP.")
        print("  -> These runs will be excluded from the Gap Comparison plots (Performance Profile/Scatter).")
        print("  -> Ensure you run CP and Heuristic on the same Seed to compare apples-to-apples.")

    # Merge heuristic results with baselines on (param, value, seed)
    # We want to keep all heuristic runs, even if baseline is missing (though we can't plot gap then)
    merged = pd.merge(df, baselines_df, on=["param", "value", "seed"], how="inner")

    if merged.empty:
        print("Warning: No intersection between heuristic results and CP baselines (after Seed match).")
        return

    # Compute Ratio = Heur / BestCP
    merged["ratio"] = merged["objective"] / merged["best_cp_obj"]
    merged["gap_pct"] = 100.0 * (merged["objective"] - merged["best_cp_obj"]) / merged["best_cp_obj"]

    # --- Print Summary of "Beating CP" ---
    print_beating_cp_summary(merged)

    # Clip any float errors where ratio < 1.0 but basically equal?
    # Or let them show up as super-efficient.

    # ----------------------------------------------------------------
    # Plot 1: Performance Profile (Quality)
    # ----------------------------------------------------------------
    # X: Ratio Tau, Y: Fraction of instances solved within Tau * Best
    # We step through tau from 1.0 to say 1.5 or 2.0

    if px is not None:
        # Plotly ECDF is perfect for this
        fig = px.ecdf(
            merged,
            x="ratio",
            color="module",
            ecdfnorm='probability',
            title="Performance Profile: Quality (Objective Ratio vs Best CP)",
            labels={
                "ratio": "Objective Ratio (Heuristic / Best CP)",
                "probability": "Fraction of Instances Solved"
            }
        )
        # Zoom in on the interesting part (1.0 to 1.5 usually)
        fig.update_xaxes(range=[0.95, 1.5])

        out_html = os.path.join(output_dir, "perf_profile_quality.html")
        fig.write_html(out_html)

    # Matplotlib version
    plt.figure(figsize=(10, 6))
    for mod in merged["module"].unique():
        data = merged[merged["module"] == mod]["ratio"].sort_values()
        y_vals = pd.Series(range(1, len(data) + 1)) / len(data)
        # Step plot for profile
        plt.step(data, y_vals, where='post', label=mod)

    plt.xlim(0.95, 1.5)
    plt.xlabel(r'Performance Ratio $\tau$ (Lower is Better)')
    plt.ylabel(r'Probability($r \leq \tau$)')
    plt.title('Performance Profile: Quality vs Best Known CP')
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_png_prof = os.path.join(output_dir, "perf_profile_quality.png")
    plt.savefig(out_png_prof, dpi=200)
    plt.close()
    print("Saved Performance Profile plots.")

    # ----------------------------------------------------------------
    # Plot 2: Scatter Gap vs Time
    # ----------------------------------------------------------------
    merged["gap_pct"] = 100.0 * (merged["objective"] - merged["best_cp_obj"]) / merged["best_cp_obj"]

    # --- CP Reference Lines & Legend ---
    # Instead of plotting CP points at their huge runtime, we just draw a line at y=0
    # and mention the median CP runtime in the legend or title.
    median_cp_time = merged["best_cp_time"].median()

    if px is not None:
        fig_scatter = px.scatter(
            merged,
            x="time",
            y="gap_pct",
            color="module",
            symbol="param",
            hover_data=["value", "seed"],
            log_x=True,
            title=f"Cost-Time Tradeoff (WS+CP Avg Time: {median_cp_time:.1f}s)",
            labels={
                "time": "Solve Time [s]",
                "gap_pct": "Gap to Best CP [%]",
                "param": "Parameter"
            }
        )
        # Add a horizontal line at y=0
        fig_scatter.add_shape(type="line",
                              x0=merged["time"].min(), y0=0,
                              x1=merged["time"].max(), y1=0,
                              line=dict(color="black", width=2, dash="dash"),
                              )

        out_html_scatter = os.path.join(output_dir, "scatter_gap_vs_time.html")
        fig_scatter.write_html(out_html_scatter)

    # Matplotlib Scatter
    plt.figure(figsize=(10, 6))
    modules = merged["module"].unique()

    # Use a colormap
    colors = plt.cm.tab10.colors

    for i, mod in enumerate(modules):
        subset = merged[merged["module"] == mod]
        plt.scatter(
            subset["time"],
            subset["gap_pct"],
            alpha=0.6,
            label=mod,
            edgecolors='w',
            color=colors[i % len(colors)]
        )

    # Dashed line at 0
    plt.axhline(0, color='black', linestyle='--', linewidth=1, label=f"CP+WS Baseline (Avg {median_cp_time:.0f}s)")

    plt.xscale("log")
    plt.xlabel("Solve Time [s]")
    plt.ylabel("Gap to Best CP [%]")
    plt.title("Cost-Time Tradeoff")
    plt.legend(title="Method")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    out_png_scatter = os.path.join(output_dir, "scatter_gap_vs_time.png")
    plt.savefig(out_png_scatter, dpi=200)
    plt.close()
    print("Saved Gap vs Time Scatter plots.")


def main():
    parser = argparse.ArgumentParser(description="Aggregate heuristic benchmark results")
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        help="Directories containing JSON result files",
        default=["./results/heuristic_local/full_final"]
    )
    parser.add_argument(
        "--cp-baseline-dirs",
        nargs="+",
        help="Directories containing CP baseline JSONs (e.g. logs/v5_and_heurv1_scaling_single_1)",
        default=["./logs/v6_and_smg_1_newdatagen"]
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to save plots and summary CSV",
        default="./results/heuristic_plots/final"
    )

    args = parser.parse_args()

    # Auto-detect optimal output directory if default 'results/plots' was not overridden
    # Logic: if input is 'logs/foo', output should be 'results/foo' (creating 'results/' if needed)
    # Only applies if user didn't explicitly set --output-dir (or provided the default)
    is_default_out = (args.output_dir == "results/plots")
    if is_default_out and args.input_dirs:
        # Take the basename of the first input directory
        inp = args.input_dirs[0].rstrip("/\\")
        base = os.path.basename(inp)
        # Construct new output path: results/<base>
        # (Assuming we want to mirror structure but change root to 'results')
        args.output_dir = os.path.join("results", base)
        print(f"Auto-set output directory to: {args.output_dir}")

    # 1. Load Data
    df = load_results(args.input_dirs)
    if df.empty:
        return

    # 2. Load CP Baseline if provided
    df_cp = pd.DataFrame()
    if args.cp_baseline_dirs:
        print(f"Loading CP baselines from {args.cp_baseline_dirs}...")
        df_cp = load_results(args.cp_baseline_dirs)

    os.makedirs(args.output_dir, exist_ok=True)

    # 3. Generate Summary CSV
    csv_path = os.path.join(args.output_dir, "summary_metrics.csv")
    generate_summary_csv(df, csv_path)

    # 5. Generate Plots per Parameter
    if not df_cp.empty:
        plot_performance_profiles(df, df_cp, args.output_dir)

    # Indentify all unique parameters present
    all_params = df["param"].unique()

    for param in all_params:
        if not param:
            continue  # skip None
        plot_parameter_scaling(df, param, args.output_dir)

    print("\nAggregation Complete.")


if __name__ == "__main__":
    main()
