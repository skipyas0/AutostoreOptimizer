#!/usr/bin/env python3
"""run_analysis.py — CLI orchestrator for the heuristic analysis pipeline.

Usage:
    python run_analysis.py [--section {full_mode,twophase,cross,all}]
                           [--results-dir PATH]
                           [--base-dir PATH]
"""
from __future__ import annotations
import visualize as viz
from metrics import compute_config_metrics, compute_cp_gap, compute_parameter_scaling
from data_loader import load_all

import argparse
import sys
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ORDER_STATION = _HERE.parent
sys.path.insert(0, str(_ORDER_STATION.parent))


DEFAULT_BASE_DIR = _ORDER_STATION
DEFAULT_RESULTS_DIR = _ORDER_STATION / "results" / "heuristic_comparison"


def _make_output_dir(results_dir: Path) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = results_dir / ts
    for sub in ("figures", "interactive", "tables", "csv"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    return out


def _run_per_config(df_configs, module: str, out_dir: Path) -> None:
    print(f"\n{'='*60}\nPer-config analysis: {module}")
    metrics = compute_config_metrics(df_configs, module)
    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    for vcol in ("objective", "total_moves"):
        m = metrics[vcol]
        print(f"  Figures for value_col={vcol} ...")
        viz.plot_gap_bar(m["gap_agg"], module, out_dir, vcol, csv_dir)
        viz.plot_gap_std_scatter(m["gap_agg"], module, out_dir, vcol, csv_dir)
        viz.plot_winrate_bar(m["winrate"], module, out_dir, vcol, csv_dir)
        viz.plot_rank_cdf(m["raw"], module, out_dir, vcol, csv_dir)

    viz.plot_moves_gap_scatter(
        metrics["objective"]["gap_agg"],
        metrics["total_moves"]["gap_agg"],
        module, out_dir, csv_dir,
    )
    if module == "heuristic_ama_sgc":
        viz.plot_attr_heatmap(metrics["objective"]["raw"], module, out_dir, "objective", csv_dir)

    viz.write_config_table(
        metrics["objective"]["gap_agg"],
        metrics["objective"]["rank_agg"],
        metrics["objective"]["winrate"],
        metrics["total_moves"]["gap_agg"],
        module, out_dir, "objective", csv_dir,
    )


def _run_cross(df_heuristics, df_cp, out_dir: Path) -> None:
    print(f"\n{'='*60}\nCross-heuristic analysis ...")
    cp_gap_raw, cp_gap_agg, cp_gap_by_param = compute_cp_gap(df_heuristics, df_cp)
    if cp_gap_raw.empty:
        print("[WARNING] No CP reference data matched — skipping cross-heuristic figures.")
        return
    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    # Global cross-heuristic plots
    viz.plot_cross_cp_gap_box(cp_gap_raw, out_dir, csv_dir)
    viz.plot_cross_cp_gap_by_param(cp_gap_by_param, out_dir, csv_dir)
    viz.plot_cross_pareto(cp_gap_raw, out_dir, csv_dir)
    viz.write_cross_heuristic_table(cp_gap_agg, df_heuristics, df_cp, out_dir, csv_dir)

    # New: parameter scaling, performance profiles, gap vs time
    merged, _agg = compute_parameter_scaling(df_heuristics, df_cp)
    if merged.empty:
        print("[WARNING] No matched instances for parameter scaling — skipping.")
    else:
        viz.plot_parameter_scaling(df_heuristics, out_dir, csv_dir)
        viz.plot_performance_profile(merged, out_dir, csv_dir)
        viz.plot_gap_vs_time(merged, out_dir, csv_dir)

    groups_to_plot = {
        "Baseline": viz.HEURISTIC_FAMILIES["Baseline"],
        "CFSS": viz.HEURISTIC_FAMILIES["CFSS"],
        "AMA": viz.HEURISTIC_FAMILIES["AMA"],
        "RDI": viz.HEURISTIC_FAMILIES["RDI"],
        "GBS": viz.HEURISTIC_FAMILIES["GBS"],
        "AMA_Baseline": viz.HEURISTIC_FAMILIES["AMA"] + viz.HEURISTIC_FAMILIES["Baseline"],
        "CFSS_Baseline": viz.HEURISTIC_FAMILIES["CFSS"] + viz.HEURISTIC_FAMILIES["Baseline"],
        "RDI_Baseline": viz.HEURISTIC_FAMILIES["RDI"] + viz.HEURISTIC_FAMILIES["Baseline"],
        "GBS_Baseline": viz.HEURISTIC_FAMILIES["GBS"] + viz.HEURISTIC_FAMILIES["Baseline"],
        "AMA_Baseline_GBS": viz.HEURISTIC_FAMILIES["AMA"] + viz.HEURISTIC_FAMILIES["Baseline"] + viz.HEURISTIC_FAMILIES["GBS"],
        "AMA_Baseline_GBS_CFSS": viz.HEURISTIC_FAMILIES["AMA"] + viz.HEURISTIC_FAMILIES["Baseline"] + viz.HEURISTIC_FAMILIES["GBS"] + viz.HEURISTIC_FAMILIES["CFSS"]
    }

    # Per-group cross-heuristic plots
    for family_name, modules in groups_to_plot.items():
        print(f"  Generating cross-heuristic figures for group: {family_name}")
        family_gap_raw = cp_gap_raw[cp_gap_raw["module"].isin(modules)]
        family_gap_by_param = cp_gap_by_param[cp_gap_by_param["module"].isin(modules)]
        family_heuristics = df_heuristics[df_heuristics["module"].isin(modules)]
        
        if not family_gap_raw.empty:
            viz.plot_cross_cp_gap_box(family_gap_raw, out_dir, csv_dir, family_name)
            viz.plot_cross_pareto(family_gap_raw, out_dir, csv_dir, family_name)
        
        if not family_gap_by_param.empty:
            viz.plot_cross_cp_gap_by_param(family_gap_by_param, out_dir, csv_dir, family_name)
            
        if not family_heuristics.empty:
            viz.plot_parameter_scaling(family_heuristics, out_dir, csv_dir, family_name)
            
        if not merged.empty:
            family_merged = merged[merged["module"].isin(modules)]
            if not family_merged.empty:
                viz.plot_performance_profile(family_merged, out_dir, csv_dir, family_name)
                viz.plot_gap_vs_time(family_merged, out_dir, csv_dir, family_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Heuristic analysis pipeline — figures and LaTeX tables."
    )
    parser.add_argument("--section", default="all",
                        choices=["full_mode", "twophase", "cross", "all"])
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--base-dir",    type=Path, default=DEFAULT_BASE_DIR)
    args = parser.parse_args()

    out_dir = _make_output_dir(args.results_dir)
    print(f"Output directory: {out_dir}")

    df_configs, df_heuristics, df_cp = load_all(args.base_dir)

    if args.section in ("full_mode", "all"):
        _run_per_config(df_configs, "heuristic_ama_sgc", out_dir)
    if args.section in ("twophase", "all"):
        _run_per_config(df_configs, "heuristic_ama_sgc_2phase", out_dir)
    if args.section in ("cross", "all"):
        _run_cross(df_heuristics, df_cp, out_dir)

    n_fig = len(list((out_dir / "figures").glob("*.*")))
    n_html = len(list((out_dir / "interactive").glob("*.html")))
    n_tex = len(list((out_dir / "tables").glob("*.tex")))
    print(f"\n{'='*60}\nAnalysis complete: {out_dir}")
    print(f"  {n_fig} figure files (PDF+PNG) | {n_html} HTML | {n_tex} .tex tables")


if __name__ == "__main__":
    main()
