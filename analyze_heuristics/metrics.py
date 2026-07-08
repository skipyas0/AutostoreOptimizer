"""metrics.py — Compute gap, rank, win-rate, and CP-gap metrics.

All functions are pure: DataFrames in -> DataFrames out, no I/O.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from pathlib import Path

def export_raw_csv(df: pd.DataFrame, csv_dir: Path, stem: str, overwrite: bool = True) -> None:
    """Save a DataFrame as a CSV in csv_dir with the given file stem.

    Args:
        df: DataFrame to export.
        csv_dir: Target directory (will be created if missing).
        stem: File name without extension.
        overwrite: Whether to overwrite existing files.
    """
    csv_dir = Path(csv_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)
    path = csv_dir / f"{stem}.csv"
    if not path.exists() or overwrite:
        df.to_csv(path, index=False)
        print(f"  Exported raw CSV: {path}")


def _add_metrics(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Add gap_{value_col} and rank_{value_col} columns.

    gap  = (value - best_on_instance) / best_on_instance * 100 [%]
    rank = dense rank within (module, instance_id); 1 = lowest = best.
    """
    df = df.copy()
    best = df.groupby(["module", "instance_id"])[value_col].transform("min")
    df[f"gap_{value_col}"] = (df[value_col] - best) / best * 100.0
    df[f"rank_{value_col}"] = (
        df.groupby(["module", "instance_id"], group_keys=False)[value_col]
        .transform(lambda x: rankdata(x.values, method="dense"))
    )
    return df


def compute_parameter_scaling(
    df_heuristics: pd.DataFrame,
    df_cp: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute parameter scaling data for heuristics and CP baseline.

    Returns (merged, agg) where merged contains all matched rows and their ratios,
    and agg contains the aggregated (median) objective and solve_time per
    (module, parameter, value).
    """
    if df_cp.empty or df_heuristics.empty:
        return pd.DataFrame(), pd.DataFrame()

    merged = pd.merge(
        df_heuristics,
        df_cp[["parameter", "value", "seed", "cp_objective"]],
        on=["parameter", "value", "seed"],
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()

    merged["ratio"] = merged["objective_value"] / merged["cp_objective"]
    merged["cp_gap"] = (merged["objective_value"] - merged["cp_objective"]) / merged["cp_objective"] * 100.0

    # Aggregate: median objective and solve_time per (module, parameter, value)
    agg = (
        merged.groupby(["module", "parameter", "value"])
        .agg(
            obj_median=("objective_value", "median"),
            obj_min=("objective_value", "min"),
            obj_max=("objective_value", "max"),
            time_median=("solve_time", "median"),
            time_min=("solve_time", "min"),
            time_max=("solve_time", "max"),
            cp_gap_mean=("cp_gap", "mean"),
            ratio_mean=("ratio", "mean"),
        )
        .reset_index()
    )

    return merged, agg


def compute_config_metrics(
    df_configs: pd.DataFrame,
    module: str,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Compute per-config gap, rank, and win-rate for one module.

    Returns nested dict: result[value_col][key]
      value_col in ("objective", "total_moves")
      key in ("raw", "gap_agg", "rank_agg", "winrate")
    """
    df = df_configs[
        (df_configs["module"] == module) & (df_configs["feasible"] == True)
    ].copy()

    config_attrs = (
        df[["config_id", "order_attr", "order_desc", "bin_attr", "bin_desc"]]
        .drop_duplicates("config_id")
    )

    results: dict[str, dict[str, pd.DataFrame]] = {}

    for vcol in ("objective", "total_moves"):
        sub = df.dropna(subset=[vcol]).copy()
        sub = _add_metrics(sub, vcol)

        gap_agg = (
            sub.groupby("config_id")[f"gap_{vcol}"]
            .agg(mean_gap="mean", median_gap="median", std_gap="std")
            .reset_index()
            .merge(config_attrs, on="config_id", how="left")
        )

        rank_agg = (
            sub.groupby("config_id")[f"rank_{vcol}"]
            .agg(mean_rank="mean", median_rank="median")
            .reset_index()
        )

        n_inst = (
            sub.groupby("config_id")["instance_id"]
            .nunique()
            .reset_index(name="n_instances")
        )
        win = (
            sub[sub[f"rank_{vcol}"] == 1]
            .groupby("config_id")["instance_id"]
            .nunique()
            .reset_index(name="win_count")
        )
        top3 = (
            sub[sub[f"rank_{vcol}"] <= 3]
            .groupby("config_id")["instance_id"]
            .nunique()
            .reset_index(name="top3_count")
        )
        w1pct = (
            sub[sub[f"gap_{vcol}"] <= 1.0]
            .groupby("config_id")["instance_id"]
            .nunique()
            .reset_index(name="within1_count")
        )
        winrate = n_inst
        for extra in (win, top3, w1pct):
            winrate = winrate.merge(extra, on="config_id", how="left")
        winrate = winrate.fillna(0)
        winrate["win_rate"] = winrate["win_count"] / winrate["n_instances"]
        winrate["top3_rate"] = winrate["top3_count"] / winrate["n_instances"]
        winrate["within1_rate"] = winrate["within1_count"] / winrate["n_instances"]
        winrate = winrate.merge(config_attrs, on="config_id", how="left")

        results[vcol] = {
            "raw": sub,
            "gap_agg": gap_agg,
            "rank_agg": rank_agg,
            "winrate": winrate,
        }

    return results


def compute_cp_gap(
    df_heuristics: pd.DataFrame,
    df_cp: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute CP gap for each heuristic on matched instances.

    cp_gap = (heuristic_obj - cp_obj) / cp_obj * 100 [%]

    Returns (raw, agg, by_param):
      raw:      per (module, instance) with cp_gap column
      agg:      per module — mean_cp_gap, median_cp_gap, std_cp_gap
      by_param: per (module, parameter, value) — mean_cp_gap, std_cp_gap
    """
    if df_cp.empty:
        print("[WARNING] df_cp is empty — no cross-heuristic CP gap computed.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    merged = pd.merge(
        df_heuristics,
        df_cp[["parameter", "value", "seed", "cp_objective"]],
        on=["parameter", "value", "seed"],
        how="inner",
    )

    if merged.empty:
        print("[WARNING] No matched instances between heuristics and CP reference.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    merged["cp_gap"] = (
        (merged["objective_value"] - merged["cp_objective"])
        / merged["cp_objective"] * 100.0
    )
    print(f"  CP gap: matched {len(merged)} / {len(df_heuristics)} instances.")

    agg = (
        merged.groupby("module")["cp_gap"]
        .agg(mean_cp_gap="mean", median_cp_gap="median", std_cp_gap="std")
        .reset_index()
    )
    by_param = (
        merged.groupby(["module", "parameter", "value"])["cp_gap"]
        .agg(mean_cp_gap="mean", std_cp_gap="std")
        .reset_index()
    )
    return merged, agg, by_param
