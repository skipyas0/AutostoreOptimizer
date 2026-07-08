"""visualize.py — Matplotlib + plotly figures and LaTeX tables.

All public functions write to out_dir/figures/ (PDF+PNG) and
out_dir/interactive/ (HTML). They accept metric DataFrames and
an output directory; they return None.
"""
from __future__ import annotations
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import metrics

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

ORDER_ATTR_COLOURS: dict[str, str] = {
    "sum_rt":          "#1f77b4",
    "max_rt":          "#ff7f0e",
    "order_size":      "#2ca02c",
    "sum_cycle":       "#d62728",
    "sku_rarity":      "#9467bd",
    "sku_contention":  "#8c564b",
    "sharing_degree":  "#e377c2",
    "min_copies":      "#7f7f7f",
}
DEFAULT_COLOUR = "#bcbd22"

MODULE_LABELS: dict[str, str] = {
    "autostore_heuristic":      "Greedy",
    "heuristic_cfss_sgc":       "CFSS-SGC",
    "heuristic_ama_sgc":        "AMA-SGC\nFull",
    "heuristic_ama_sgc_2phase": "AMA-SGC\n2-Phase",
    "heuristic_rdi_sgc":        "RDI-SGC\nSum RT",
    "heuristic_rdi_sgc_sharing_degree": "RDI-SGC\nShare-Deg",
    "heuristic_rdi_sgc_best_score": "RDI-SGC\nBest-Score",
    "heuristic_gbs":            "GBS\nReadiness",
    "heuristic_gbs_critical_path": "GBS\nCrit Path",
    "heuristic_gbs_max_sharing": "GBS\nMax Sharing",
}

# MODULE_COLOURS: dict[str, str] = {
#     "autostore_heuristic":      "#7f7f7f",
#     "heuristic_cfss_sgc":       "#1f77b4",
#     "heuristic_ama_sgc":        "#ff7f0e",
#     "heuristic_ama_sgc_2phase": "#2ca02c",
#     "heuristic_rdi_sgc":        "#bcbd22",
#     "heuristic_rdi_sgc_sharing_degree": "#9467bd",
#     "heuristic_rdi_sgc_best_score": "#8c564b",
#     "heuristic_gbs":            "#d62728",
#     "heuristic_gbs_critical_path": "#e377c2",
#     "heuristic_gbs_max_sharing": "#17becf",
# }

MODULE_COLOURS: dict[str, str] = {
    # Baseline / Control
    "autostore_heuristic":              "#808080",  # Neutral Gray

    # CFSS Family
    "heuristic_cfss_sgc":               "#d1d40a",  # Strong Blue

    # AMA Family (Oranges)
    "heuristic_ama_sgc":                "#e6550d",  # Deep Orange
    "heuristic_ama_sgc_2phase":         "#fd8d3c",  # Lighter Orange

    # RDI Family (Greens)
    "heuristic_rdi_sgc":                "#006d2c",  # Deep Green
    "heuristic_rdi_sgc_sharing_degree": "#31a354",  # Medium Green
    "heuristic_rdi_sgc_best_score":     "#74c476",  # Light Green

    # GBS Family (Purples)
    "heuristic_gbs":                    "#54278f",  # Deep Purple
    "heuristic_gbs_critical_path":      "#5e51a8",  # Medium Purple
    "heuristic_gbs_max_sharing":        "#7c7fca",  # Light Purple
}

MODULE_ORDER: list[str] = [
    "autostore_heuristic",
    "heuristic_ama_sgc_2phase",
    "heuristic_ama_sgc",
    "heuristic_gbs_max_sharing",
    "heuristic_gbs",
    "heuristic_gbs_critical_path",
    "heuristic_cfss_sgc",
    "heuristic_rdi_sgc",
    "heuristic_rdi_sgc_sharing_degree",
    "heuristic_rdi_sgc_best_score",
]

HEURISTIC_FAMILIES = {
    "Baseline": ["autostore_heuristic"],
    "CFSS": ["heuristic_cfss_sgc"],
    "AMA": ["heuristic_ama_sgc", "heuristic_ama_sgc_2phase"],
    "RDI": ["heuristic_rdi_sgc", "heuristic_rdi_sgc_sharing_degree", "heuristic_rdi_sgc_best_score"],
    "GBS": ["heuristic_gbs", "heuristic_gbs_critical_path", "heuristic_gbs_max_sharing"],
}

def _module_prefix(module: str) -> str:

    return "full_mode" if module == "heuristic_ama_sgc" else "twophase"


def _ensure_dirs(out_dir: Path, subdir: str = None) -> tuple[Path, Path, Path]:
    fig_dir = out_dir / "figures"
    int_dir = out_dir / "interactive"
    csv_dir = out_dir / "csv"
    
    if subdir:
        fig_dir = fig_dir / subdir
        int_dir = int_dir / subdir
        csv_dir = csv_dir / subdir
        
    fig_dir.mkdir(parents=True, exist_ok=True)
    int_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir, int_dir, csv_dir


def _save_mpl(fig: plt.Figure, fig_dir: Path, stem: str) -> None:
    fig.savefig(fig_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def _save_plotly(fig: go.Figure, int_dir: Path, stem: str) -> None:
    fig.update_layout(font=dict(size=24))
    fig.write_html(str(int_dir / f"{stem}.html"))


def _bar_colors(order_attrs: pd.Series) -> list[str]:
    return [ORDER_ATTR_COLOURS.get(oa, DEFAULT_COLOUR) for oa in order_attrs]


def _pareto_mask(df: pd.DataFrame, x_col: str, y_col: str) -> pd.Series:
    """Boolean mask of Pareto-optimal rows minimising both x_col and y_col."""
    is_pareto = pd.Series(True, index=df.index)
    x = df[x_col].values
    y = df[y_col].values
    for i in range(len(df)):
        if not is_pareto.iloc[i]:
            continue
        dominated = (x <= x[i]) & (y <= y[i]) & ((x < x[i]) | (y < y[i]))
        if dominated.any():
            is_pareto.iloc[i] = False
    return is_pareto


# ── Per-config figures ────────────────────────────────────────────────────────

def plot_gap_bar(
    gap_agg: pd.DataFrame,
    module: str,
    out_dir: Path,
    value_col: str = "objective",
    csv_dir: Path | None = None,
    show_titles: bool = False,
) -> None:
    """Sorted horizontal bar + ±1 std error bars for all configs.

    gap_agg columns: config_id, mean_gap, std_gap, order_attr,
                     order_desc, bin_attr, bin_desc
    """
    fig_dir, int_dir, csv_dir_actual = _ensure_dirs(out_dir)
    stem = f"{_module_prefix(module)}_gap_bar_{value_col}"
    if csv_dir is not None:
        csv_dir_actual = csv_dir
    metrics.export_raw_csv(gap_agg, csv_dir_actual, stem)
    df = gap_agg.sort_values("mean_gap").reset_index(drop=True)
    if module in ["heuristic_ama_sgc", "heuristic_ama_sgc_2phase"] and len(df) > 16:
        # Best are at the beginning (index 0)
        df_top = df.iloc[:11].copy()
        df_bottom = df.iloc[-5:].copy()
        omitted = len(df) - 16
        dummy = pd.DataFrame([{
            "config_id": f"... {omitted} configs omitted ...",
            "mean_gap": 0, "std_gap": 0,
            "order_attr": "omitted", "order_desc": False,
            "bin_attr": "omitted", "bin_desc": False
        }])
        df = pd.concat([df_top, dummy, df_bottom], ignore_index=True)
    n = len(df)
    height = max(6, n * 0.18)
    colours = _bar_colors(df["order_attr"])
    label = "Full-mode" if module == "heuristic_ama_sgc" else "2-Phase"

    # ── matplotlib ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, height))
    y_pos = np.arange(n)
    ax.barh(y_pos, df["mean_gap"], xerr=df["std_gap"].fillna(0),
            color=colours, alpha=0.85, capsize=2, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["config_id"], fontsize=max(8, min(16, 400 // n)))
    ax.set_xlabel(f"Mean Gap from Best on Instance (%) — {value_col}")
    if show_titles:

        ax.set_title(f"AMA-SGC {label} — Config Gap ({value_col}), sorted by mean")
    ax.axvline(0, color="black", linewidth=0.8)
    seen = {a: c for a, c in ORDER_ATTR_COLOURS.items() if a in df["order_attr"].values}
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=a) for a, c in seen.items()]
    ax.legend(handles=handles, title="order_attr" if show_titles else None, fontsize=14, loc="lower right")
    fig.tight_layout()
    _save_mpl(fig, fig_dir, stem)

    # ── plotly ──────────────────────────────────────────────────────────────
    hover = (
        "config: %{customdata[0]}<br>order_attr: %{customdata[1]}<br>"
        "bin_attr: %{customdata[3]}<br>"
        "mean_gap: %{x:.3f}%  std: %{customdata[5]:.3f}%<extra></extra>"
    )
    fig_px = go.Figure()
    for attr, grp in df.groupby("order_attr", sort=False):
        fig_px.add_trace(go.Bar(
            y=grp["config_id"], x=grp["mean_gap"], orientation="h",
            name=attr,
            marker_color=ORDER_ATTR_COLOURS.get(attr, DEFAULT_COLOUR),
            error_x=dict(type="data", array=grp["std_gap"].fillna(0).tolist(),
                         visible=True),
            customdata=grp[["config_id", "order_attr", "order_desc",
                            "bin_attr", "bin_desc", "std_gap"]].values,
            hovertemplate=hover,
        ))
    fig_px.update_layout(
        barmode="overlay",
        title=f"AMA-SGC {label} — Gap Bar ({value_col})" if show_titles else None,
        xaxis_title=f"Mean Gap (%) — {value_col}" if show_titles else None,
        yaxis=dict(autorange="reversed"),
        height=max(400, n * 18),
    )
    _save_plotly(fig_px, int_dir, stem)
    print(f"  Saved {stem}")


def plot_gap_std_scatter(
    gap_agg: pd.DataFrame,
    module: str,
    out_dir: Path,
    value_col: str = "objective",
    csv_dir: Path | None = None,
    show_titles: bool = False,
) -> None:
    """Mean gap vs std-gap scatter — primary decision plot.

    Pareto-front configs (minimising both axes) are annotated.
    Bottom-left = low mean gap AND low variance = best reliable config.
    """
    fig_dir, int_dir, csv_dir_actual = _ensure_dirs(out_dir)
    stem = f"{_module_prefix(module)}_gap_std_scatter_{value_col}"
    if csv_dir is not None:
        csv_dir_actual = csv_dir
    metrics.export_raw_csv(gap_agg, csv_dir_actual, stem)
    df = gap_agg.dropna(subset=["mean_gap", "std_gap"]).copy()
    pareto = _pareto_mask(df, "mean_gap", "std_gap")
    label = "Full-mode" if module == "heuristic_ama_sgc" else "2-Phase"

    # ── matplotlib ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    for attr in df["order_attr"].unique():
        sub = df[df["order_attr"] == attr]
        ax.scatter(sub["mean_gap"], sub["std_gap"],
                   color=ORDER_ATTR_COLOURS.get(attr, DEFAULT_COLOUR),
                   alpha=0.65, s=25, label=attr)
    for _, row in df[pareto].iterrows():
        ax.annotate(row["config_id"], (row["mean_gap"], row["std_gap"]),
                    fontsize=12, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(f"Mean Gap (%) — {value_col}")
    ax.set_ylabel(f"Std of Gap (%) — {value_col}")
    if show_titles:

        ax.set_title(
        f"AMA-SGC {label} — Config Selection: Mean Gap vs Std ({value_col})\n"
        "Bottom-left = best (low mean AND low variance). Stars = Pareto front."
    )
    ax.legend(title="order_attr" if show_titles else None, fontsize=14)
    fig.tight_layout()
    _save_mpl(fig, fig_dir, stem)

    # ── plotly ──────────────────────────────────────────────────────────────
    df = df.copy()
    df["pareto"] = pareto.values
    hover = (
        "config: %{customdata[0]}<br>order_attr: %{customdata[1]}<br>"
        "bin_attr: %{customdata[3]}<br>"
        "mean_gap: %{x:.3f}%<br>std_gap: %{y:.3f}%<br>"
        "Pareto: %{customdata[6]}<extra></extra>"
    )
    fig_px = go.Figure()
    for attr, grp in df.groupby("order_attr", sort=False):
        fig_px.add_trace(go.Scatter(
            x=grp["mean_gap"], y=grp["std_gap"], mode="markers",
            name=attr,
            marker=dict(
                color=ORDER_ATTR_COLOURS.get(attr, DEFAULT_COLOUR),
                size=8, opacity=0.7,
                symbol=grp["pareto"].map({True: "star", False: "circle"}).tolist(),
            ),
            customdata=grp[["config_id", "order_attr", "order_desc",
                            "bin_attr", "bin_desc", "mean_gap", "pareto"]].values,
            hovertemplate=hover,
        ))
    fig_px.update_layout(
        title=f"AMA-SGC {label} — Mean Gap vs Std ({value_col})" if show_titles else None,
        xaxis_title=f"Mean Gap (%) — {value_col}" if show_titles else None,
        yaxis_title=f"Std of Gap (%) — {value_col}" if show_titles else None,
    )
    _save_plotly(fig_px, int_dir, stem)
    print(f"  Saved {stem}")


def plot_winrate_bar(
    winrate: pd.DataFrame,
    module: str,
    out_dir: Path,
    value_col: str = "objective",
    csv_dir: Path | None = None,
    show_titles: bool = False,
) -> None:
    """Horizontal grouped bar: win_rate, top3_rate, within1_rate per config.

    Sorted by win_rate descending (ascending in horizontal bar = top = best).
    winrate columns: config_id, win_rate, top3_rate, within1_rate,
                     order_attr, order_desc, bin_attr, bin_desc
    """
    fig_dir, int_dir, csv_dir_actual = _ensure_dirs(out_dir)
    stem = f"{_module_prefix(module)}_winrate_bar_{value_col}"
    if csv_dir is not None:
        csv_dir_actual = csv_dir
    metrics.export_raw_csv(winrate, csv_dir_actual, stem)
    df = winrate.sort_values("win_rate", ascending=True).reset_index(drop=True)
    if module in ["heuristic_ama_sgc", "heuristic_ama_sgc_2phase"] and len(df) > 16:
        # Best are at the end (index -1) because ascending=True
        df_bottom = df.iloc[:5].copy()  # Worst 5
        df_top = df.iloc[-11:].copy()   # Best 11
        omitted = len(df) - 16
        dummy = pd.DataFrame([{
            "config_id": f"... {omitted} configs omitted ...",
            "win_rate": 0, "top3_rate": 0, "within1_rate": 0,
            "order_attr": "omitted", "order_desc": False,
            "bin_attr": "omitted", "bin_desc": False
        }])
        df = pd.concat([df_bottom, dummy, df_top], ignore_index=True)
    n = len(df)
    height = max(6, n * 0.18)
    y_pos = np.arange(n)
    bar_h = 0.25
    label = "Full-mode" if module == "heuristic_ama_sgc" else "2-Phase"

    # ── matplotlib ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, height))
    ax.barh(y_pos + bar_h, df["within1_rate"] * 100, height=bar_h,
            label="Within 1%",  color="#2ca02c", alpha=0.8)
    ax.barh(y_pos,          df["top3_rate"] * 100, height=bar_h,
            label="Top-3",     color="#ff7f0e", alpha=0.8)
    ax.barh(y_pos - bar_h,  df["win_rate"] * 100, height=bar_h,
            label="Win rate",  color="#1f77b4", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["config_id"], fontsize=max(8, min(16, 400 // n)))
    ax.set_xlabel("Rate (%)")
    ax.set_xlim(0, 105)
    if show_titles:

        ax.set_title(f"AMA-SGC {label} — Win / Top-3 / Within-1% Rates ({value_col})")
    ax.legend(fontsize=16)
    fig.tight_layout()
    _save_mpl(fig, fig_dir, stem)

    # ── plotly ──────────────────────────────────────────────────────────────
    custom = df[["config_id", "order_attr", "order_desc", "bin_attr", "bin_desc"]].values
    hover = ("config: %{customdata[0]}<br>order_attr: %{customdata[1]}<br>"
             "bin_attr: %{customdata[3]}<br>value: %{x:.1f}%<extra></extra>")
    fig_px = go.Figure()
    for rate_col, lbl, colour in [
        ("win_rate",     "Win rate", "#1f77b4"),
        ("top3_rate",    "Top-3",    "#ff7f0e"),
        ("within1_rate", "≤1%",      "#2ca02c"),
    ]:
        fig_px.add_trace(go.Bar(
            y=df["config_id"], x=df[rate_col] * 100,
            orientation="h", name=lbl, marker_color=colour,
            customdata=custom, hovertemplate=hover,
        ))
    fig_px.update_layout(
        barmode="group",
        title=f"AMA-SGC {label} — Win Rates ({value_col})" if show_titles else None,
        xaxis_title="Rate (%)" if show_titles else None, yaxis=dict(autorange="reversed"),
        height=max(400, n * 18),
    )
    _save_plotly(fig_px, int_dir, stem)
    print(f"  Saved {stem}")


def plot_rank_cdf(
    rank_raw: pd.DataFrame,
    module: str,
    out_dir: Path,
    value_col: str = "objective",
    csv_dir: Path | None = None,
    show_titles: bool = False,
) -> None:
    """P(rank ≤ k) vs k for all configs.

    Full-mode matplotlib: one mean line per order_attr family.
    2-phase matplotlib: individual lines for all configs.
    Plotly: always individual lines with hover.

    rank_raw columns: config_id, rank_{value_col}, instance_id, order_attr
    """
    fig_dir, int_dir, csv_dir_actual = _ensure_dirs(out_dir)
    rank_col = f"rank_{value_col}"
    stem = f"{_module_prefix(module)}_rank_cdf_{value_col}"
    if csv_dir is not None:
        csv_dir_actual = csv_dir
    metrics.export_raw_csv(rank_raw, csv_dir_actual, stem)
    is_full = module == "heuristic_ama_sgc"
    label = "Full-mode" if is_full else "2-Phase"

    all_configs = rank_raw["config_id"].unique()
    max_rank = int(rank_raw[rank_col].max())
    k_vals = np.arange(1, max_rank + 1)

    def cdf_for_config(cfg: str) -> np.ndarray:
        ranks = rank_raw[rank_raw["config_id"] == cfg][rank_col].values
        return np.array([(ranks <= k).mean() for k in k_vals])

    # ── matplotlib ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    if is_full:
        for attr in rank_raw["order_attr"].unique():
            cfgs = rank_raw[rank_raw["order_attr"] == attr]["config_id"].unique()
            mean_cdf = np.stack([cdf_for_config(c) for c in cfgs]).mean(axis=0)
            ax.step(k_vals, mean_cdf, where="post",
                    color=ORDER_ATTR_COLOURS.get(attr, DEFAULT_COLOUR),
                    label=attr, linewidth=1.5)
        legend_title = "order_attr family (mean)"
    else:
        for cfg in all_configs:
            attr = rank_raw[rank_raw["config_id"] == cfg]["order_attr"].iloc[0]
            ax.step(k_vals, cdf_for_config(cfg), where="post",
                    color=ORDER_ATTR_COLOURS.get(attr, DEFAULT_COLOUR),
                    alpha=0.6, linewidth=1.0, label=cfg)
        legend_title = "config"

    ax.set_xlabel(f"Rank k — {value_col}")
    ax.set_ylabel(r"$P(\mathrm{rank} \leq k)$")
    ax.set_xlim(1, min(max_rank, 20))
    if show_titles:

        ax.set_title(f"AMA-SGC {label} — Rank CDF ({value_col})"
                 + (" — mean per order_attr" if is_full else ""))
    ax.legend(title=legend_title, fontsize=12, ncol=2)
    fig.tight_layout()
    _save_mpl(fig, fig_dir, stem)

    # ── plotly (all configs individually) ───────────────────────────────────
    fig_px = go.Figure()
    for cfg in all_configs:
        attr = rank_raw[rank_raw["config_id"] == cfg]["order_attr"].iloc[0]
        cdf = cdf_for_config(cfg)
        fig_px.add_trace(go.Scatter(
            x=k_vals.tolist(), y=cdf.tolist(), mode="lines", name=cfg,
            line=dict(color=ORDER_ATTR_COLOURS.get(attr, DEFAULT_COLOUR), width=1),
            hovertemplate=f"config: {cfg}<br>rank ≤ %{{x}}: %{{y:.2f}}<extra></extra>",
        ))
    fig_px.update_layout(
        title=f"AMA-SGC {label} — Rank CDF ({value_col})" if show_titles else None,
        xaxis_title=f"Rank k — {value_col}" if show_titles else None,
        yaxis_title="P(rank ≤ k)" if show_titles else None,
    )
    _save_plotly(fig_px, int_dir, stem)
    print(f"  Saved {stem}")


def plot_moves_gap_scatter(
    gap_obj_agg: pd.DataFrame,
    gap_moves_agg: pd.DataFrame,
    module: str,
    out_dir: Path,
    csv_dir: Path | None = None,
    show_titles: bool = False,
) -> None:
    """Objective gap (x) vs moves gap (y) per config.

    Reveals configs good on makespan but expensive on moves, or vice versa.
    gap_obj_agg:   config_id, mean_gap, order_attr, order_desc, bin_attr, bin_desc
    gap_moves_agg: config_id, mean_gap
    """
    fig_dir, int_dir, csv_dir_actual = _ensure_dirs(out_dir)
    stem = f"{_module_prefix(module)}_moves_gap_scatter"
    if csv_dir is not None:
        csv_dir_actual = csv_dir
    metrics.export_raw_csv(gap_obj_agg, csv_dir_actual, stem)
    label = "Full-mode" if module == "heuristic_ama_sgc" else "2-Phase"

    df = gap_obj_agg[["config_id", "mean_gap", "order_attr",
                      "order_desc", "bin_attr", "bin_desc"]].merge(
        gap_moves_agg[["config_id", "mean_gap"]].rename(
            columns={"mean_gap": "mean_moves_gap"}),
        on="config_id", how="inner",
    ).dropna(subset=["mean_gap", "mean_moves_gap"])
    pareto = _pareto_mask(df, "mean_gap", "mean_moves_gap")

    # ── matplotlib ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    for attr in df["order_attr"].unique():
        sub = df[df["order_attr"] == attr]
        ax.scatter(sub["mean_gap"], sub["mean_moves_gap"],
                   color=ORDER_ATTR_COLOURS.get(attr, DEFAULT_COLOUR),
                   alpha=0.65, s=25, label=attr)
    for _, row in df[pareto].iterrows():
        ax.annotate(row["config_id"], (row["mean_gap"], row["mean_moves_gap"]),
                    fontsize=12, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Mean Objective Gap (%)")
    ax.set_ylabel("Mean Total-Moves Gap (%)")
    if show_titles:

        ax.set_title(f"AMA-SGC {label} — Objective vs Moves Gap\n"
                 "Bottom-left = low gap on both metrics. Annotated = Pareto front.")
    ax.legend(title="order_attr" if show_titles else None, fontsize=14)
    fig.tight_layout()
    _save_mpl(fig, fig_dir, stem)

    # ── plotly ──────────────────────────────────────────────────────────────
    df = df.copy()
    df["pareto"] = pareto.values
    hover = ("config: %{customdata[0]}<br>order_attr: %{customdata[1]}<br>"
             "bin_attr: %{customdata[3]}<br>"
             "obj_gap: %{x:.3f}%<br>moves_gap: %{y:.3f}%<br>"
             "Pareto: %{customdata[5]}<extra></extra>")
    fig_px = go.Figure()
    for attr, grp in df.groupby("order_attr", sort=False):
        fig_px.add_trace(go.Scatter(
            x=grp["mean_gap"], y=grp["mean_moves_gap"], mode="markers", name=attr,
            marker=dict(color=ORDER_ATTR_COLOURS.get(attr, DEFAULT_COLOUR),
                        size=8, opacity=0.7,
                        symbol=grp["pareto"].map({True: "star", False: "circle"}).tolist()),
            customdata=grp[["config_id", "order_attr", "order_desc",
                            "bin_attr", "bin_desc", "pareto"]].values,
            hovertemplate=hover,
        ))
    fig_px.update_layout(
        title=f"AMA-SGC {label} — Objective vs Moves Gap" if show_titles else None,
        xaxis_title="Mean Objective Gap (%)" if show_titles else None,
        yaxis_title="Mean Total-Moves Gap (%)" if show_titles else None,
    )
    _save_plotly(fig_px, int_dir, stem)
    print(f"  Saved {stem}")


def plot_attr_heatmap(
    gap_raw: pd.DataFrame,
    module: str,
    out_dir: Path,
    value_col: str = "objective",
    csv_dir: Path | None = None,
    show_titles: bool = False,
) -> None:
    """order_attr × bin_attr heatmap of mean gap, split by sort directions.

    Only produced for full-mode (heuristic_ama_sgc). Skipped otherwise.
    gap_raw columns: order_attr, order_desc, bin_attr, bin_desc, gap_{value_col}
    """
    if module != "heuristic_ama_sgc":
        return
    fig_dir, int_dir, csv_dir_actual = _ensure_dirs(out_dir)
    stem = f"full_mode_attr_heatmap_{value_col}"
    if csv_dir is not None:
        csv_dir_actual = csv_dir
    metrics.export_raw_csv(gap_raw, csv_dir_actual, stem)
    gap_col = f"gap_{value_col}"

    df_plot = gap_raw.copy()
    df_plot["order_full"] = df_plot["order_attr"] + df_plot["order_desc"].map({True: "↓", False: "↑"})
    df_plot["bin_full"] = df_plot["bin_attr"] + df_plot["bin_desc"].map({True: "↓", False: "↑"})

    pivot = (
        df_plot.groupby(["order_full", "bin_full"])[gap_col]
        .mean()
        .unstack("bin_full")
    )
    order_attrs = sorted(pivot.index.tolist())
    bin_attrs = sorted(pivot.columns.tolist())
    matrix = pivot.reindex(index=order_attrs, columns=bin_attrs).values

    # ── matplotlib ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, len(bin_attrs) * 1.3),
                                    max(4, len(order_attrs) * 0.9)))
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(bin_attrs)))
    ax.set_xticklabels(bin_attrs, rotation=35, ha="right", fontsize=18)
    ax.set_yticks(range(len(order_attrs)))
    ax.set_yticklabels(order_attrs, fontsize=18)
    if show_titles:

        ax.set_title(f"Full-mode AMA-SGC — Mean Gap by order_attr × bin_attr ({value_col})\n"
                 "Split by sort direction (↓=desc, ↑=asc); lower = better")
    plt.colorbar(im, ax=ax, label="Mean Gap (%)")
    vmax = np.nanmax(matrix)
    for i in range(len(order_attrs)):
        for j in range(len(bin_attrs)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=16,
                        color="white" if v > vmax * 0.65 else "black")
    fig.tight_layout()
    _save_mpl(fig, fig_dir, stem)

    # ── plotly ──────────────────────────────────────────────────────────────
    fig_px = go.Figure(data=go.Heatmap(
        z=matrix.tolist(), x=bin_attrs, y=order_attrs,
        colorscale="RdYlGn", reversescale=True,
        text=[[f"{v:.2f}%" if not np.isnan(v) else "" for v in row] for row in matrix],
        texttemplate="%{text}",
        hovertemplate="order_attr: %{y}<br>bin_attr: %{x}<br>mean gap: %{z:.3f}%<extra></extra>",
        colorbar=dict(title="Mean Gap (%)"),
    ))
    fig_px.update_layout(
        title=f"Full-mode AMA-SGC — Mean Gap Heatmap ({value_col})" if show_titles else None,
        xaxis_title="bin_attr" if show_titles else None, yaxis_title="order_attr" if show_titles else None,
    )
    _save_plotly(fig_px, int_dir, stem)
    print(f"  Saved {stem}")


# ── Cross-heuristic figures ───────────────────────────────────────────────────

def plot_cross_cp_gap_box(cp_gap_raw: pd.DataFrame, out_dir: Path, csv_dir: Path = None, family_name: str = None, show_titles: bool = False) -> None:
    """Box plot of CP gap distribution per heuristic.

    cp_gap_raw columns: module, cp_gap
    """
    fig_dir, int_dir, csv_dir_actual = _ensure_dirs(out_dir, subdir=family_name)
    stem = f"{family_name}_cross_cp_gap_box" if family_name else "cross_cp_gap_box"
    if csv_dir is not None:
        csv_dir_actual = csv_dir / family_name if family_name else csv_dir
        csv_dir_actual.mkdir(parents=True, exist_ok=True)
    metrics.export_raw_csv(cp_gap_raw, csv_dir_actual, stem)
    modules = cp_gap_raw["module"].unique().tolist()
    modules = sorted(modules, key=lambda m: MODULE_ORDER.index(m) if m in MODULE_ORDER else 999)
    labels = [MODULE_LABELS.get(m, m) for m in modules]
    colours = [MODULE_COLOURS.get(m, "#333") for m in modules]

    # ── matplotlib ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [cp_gap_raw[cp_gap_raw["module"] == m]["cp_gap"].dropna().values
            for m in modules]
    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, c in zip(bp["boxes"], colours):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2,
               label="CP+WS reference (0%)")
    ax.set_ylabel("Gap to CP+WS Best Solution (%)")
    
    title_suffix = f" ({family_name.replace('_', ' + ')})" if family_name else ""
    if show_titles:

        ax.set_title(f"Cross-Heuristic — Gap to CP+WS Best Solution{title_suffix}")
    ax.legend(fontsize=16)
    fig.tight_layout()
    _save_mpl(fig, fig_dir, stem)

    # ── plotly ──────────────────────────────────────────────────────────────
    fig_px = go.Figure()
    for m, lbl, c in zip(modules, labels, colours):
        fig_px.add_trace(go.Box(
            y=cp_gap_raw[cp_gap_raw["module"] == m]["cp_gap"].dropna().tolist(),
            name=lbl, marker_color=c, boxmean="sd",
        ))
    fig_px.add_hline(y=0, line_dash="dash", line_color="black",
                     annotation_text="CP+WS reference (0%)",
                     annotation_position="top right")
    fig_px.update_layout(title=f"Cross-Heuristic — Gap to CP+WS Best Solution{title_suffix}" if show_titles else None,
                         yaxis_title="Gap to CP+WS Best Solution (%)")
    _save_plotly(fig_px, int_dir, stem)
    print(f"  Saved {stem}" + (f" in {family_name}" if family_name else ""))


def plot_cross_cp_gap_by_param(
    cp_gap_by_param: pd.DataFrame, out_dir: Path, csv_dir: Path = None, family_name: str = None, show_titles: bool = False
) -> None:
    """Faceted bar: mean CP gap per heuristic for each parameter value.

    cp_gap_by_param columns: module, parameter, value, mean_cp_gap, std_cp_gap
    """
    fig_dir, int_dir, csv_dir_actual = _ensure_dirs(out_dir, subdir=family_name)
    stem = f"{family_name}_cross_cp_gap_by_param" if family_name else "cross_cp_gap_by_param"
    title_suffix = f" ({family_name.replace('_', ' + ')})" if family_name else ""
    
    if csv_dir is not None:
        csv_dir_actual = csv_dir / family_name if family_name else csv_dir
        csv_dir_actual.mkdir(parents=True, exist_ok=True)
    metrics.export_raw_csv(cp_gap_by_param, csv_dir_actual, stem)
    params = sorted(cp_gap_by_param["parameter"].unique())
    modules = cp_gap_by_param["module"].unique().tolist()
    modules = sorted(modules, key=lambda m: MODULE_ORDER.index(m) if m in MODULE_ORDER else 999)
    n_mods = len(modules)

    # ── matplotlib ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(params), figsize=(5 * len(params), 5), sharey=True)
    if len(params) == 1:
        axes = [axes]
    for ax, param in zip(axes, params):
        sub = cp_gap_by_param[cp_gap_by_param["parameter"] == param].sort_values("value")
        x_vals = sorted(sub["value"].unique())
        bar_w = 0.8 / n_mods
        for i, mod in enumerate(modules):
            md = sub[sub["module"] == mod].set_index("value")
            x_pos = [xi + (i - n_mods / 2 + 0.5) * bar_w for xi in range(len(x_vals))]
            y_vals = [float(md.loc[v, "mean_cp_gap"]) if v in md.index else 0.0
                      for v in x_vals]
            yerr = [float(md.loc[v, "std_cp_gap"]) if v in md.index else 0.0
                    for v in x_vals]
            ax.bar(x_pos, y_vals, width=bar_w, yerr=yerr, capsize=3,
                   label=MODULE_LABELS.get(mod, mod),
                   color=MODULE_COLOURS.get(mod, "#333"), alpha=0.8)
        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels([str(v) for v in x_vals], fontsize=16)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        if show_titles:

            ax.set_title(f"param = {param}")
        ax.set_xlabel(param)
    axes[0].set_ylabel("Mean CP Gap (%)")
    axes[0].legend(fontsize=14)
    if show_titles:

        fig.suptitle(f"Cross-Heuristic — Mean CP Gap by Parameter{title_suffix}")
    fig.tight_layout()
    _save_mpl(fig, fig_dir, stem)

    # ── matplotlib (individual subplots) ───────────────────────────────────
    for param in params:
        fig_single, ax_single = plt.subplots(figsize=(6, 5))
        sub = cp_gap_by_param[cp_gap_by_param["parameter"] == param].sort_values("value")
        x_vals = sorted(sub["value"].unique())
        bar_w = 0.8 / n_mods
        for i, mod in enumerate(modules):
            md = sub[sub["module"] == mod].set_index("value")
            x_pos = [xi + (i - n_mods / 2 + 0.5) * bar_w for xi in range(len(x_vals))]
            y_vals = [float(md.loc[v, "mean_cp_gap"]) if v in md.index else 0.0
                      for v in x_vals]
            yerr = [float(md.loc[v, "std_cp_gap"]) if v in md.index else 0.0
                    for v in x_vals]
            ax_single.bar(x_pos, y_vals, width=bar_w, yerr=yerr, capsize=3,
                   label=MODULE_LABELS.get(mod, mod),
                   color=MODULE_COLOURS.get(mod, "#333"), alpha=0.8)
        ax_single.set_xticks(range(len(x_vals)))
        ax_single.set_xticklabels([str(v) for v in x_vals], fontsize=16)
        ax_single.axhline(0, color="black", linestyle="--", linewidth=0.8)
        if show_titles:

            ax_single.set_title(f"Mean CP Gap by Parameter: {param}{title_suffix}")
        ax_single.set_xlabel(param)
        ax_single.set_ylabel("Mean CP Gap (%)")
        ax_single.legend(fontsize=14)
        fig_single.tight_layout()
        _save_mpl(fig_single, fig_dir, f"{stem}_{param}")

    # ── plotly ──────────────────────────────────────────────────────────────
    fig_px = px.bar(
        cp_gap_by_param, x="value", y="mean_cp_gap", error_y="std_cp_gap",
        color="module", facet_col="parameter", barmode="group",
        color_discrete_map=MODULE_COLOURS,
        category_orders={"module": modules},
        labels={"mean_cp_gap": "Mean CP Gap (%)", "value": "Parameter value"},
        title=f"Cross-Heuristic — Mean CP Gap by Parameter{title_suffix}" if show_titles else None,
    )
    fig_px.for_each_trace(lambda t: t.update(name=MODULE_LABELS.get(t.name, t.name)))
    fig_px.add_hline(y=0, line_dash="dash", line_color="black")
    _save_plotly(fig_px, int_dir, stem)
    print(f"  Saved {stem}" + (f" in {family_name}" if family_name else ""))


def plot_cross_pareto(cp_gap_raw: pd.DataFrame, out_dir: Path, csv_dir: Path = None, family_name: str = None, show_titles: bool = False) -> None:
    """Solve-time vs CP gap scatter across all heuristics and instances.

    x = heuristic solve_time [s]
    y = CP gap [%]
    Dashed horizontal line at y = 0 (CP reference).
    Plotly has linear/log x-axis dropdown.

    cp_gap_raw columns: module, parameter, value, seed, solve_time, cp_gap
    """
    fig_dir, int_dir, csv_dir_actual = _ensure_dirs(out_dir, subdir=family_name)
    stem = f"{family_name}_cross_pareto" if family_name else "cross_pareto"
    title_suffix = f" ({family_name.replace('_', ' + ')})" if family_name else ""
    
    if csv_dir is not None:
        csv_dir_actual = csv_dir / family_name if family_name else csv_dir
        csv_dir_actual.mkdir(parents=True, exist_ok=True)
    metrics.export_raw_csv(cp_gap_raw, csv_dir_actual, stem)
    params = sorted(cp_gap_raw["parameter"].unique())
    param_markers_mpl = dict(zip(params, ["o", "s", "^", "D", "v", "P"]))
    param_symbols_ply = dict(zip(params, ["circle", "square", "triangle-up", "diamond"]))

    modules = cp_gap_raw["module"].unique().tolist()
    modules = sorted(modules, key=lambda m: MODULE_ORDER.index(m) if m in MODULE_ORDER else 999)

    # ── matplotlib ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted_modules: set[str] = set()
    for mod in modules:
        sub = cp_gap_raw[cp_gap_raw["module"] == mod]
        for param in sub["parameter"].unique():
            s2 = sub[sub["parameter"] == param]
            ax.scatter(
                s2["solve_time"], s2["cp_gap"],
                color=MODULE_COLOURS.get(mod, "#333"),
                marker=param_markers_mpl.get(param, "o"),
                alpha=0.6, s=30,
                label=MODULE_LABELS.get(mod, mod) if mod not in plotted_modules else "_",
            )
            plotted_modules.add(mod)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.5,
               label="CP+WS Best Solution (0%)")
    ax.set_xlabel("Heuristic Solve Time [s]")
    ax.set_xscale("log")
    ax.set_ylabel("Gap to CP+WS Best Solution (%)")
    if show_titles:

        ax.set_title(f"Quality–Time Trade-off: Heuristics vs CP+WS Best Solution{title_suffix}")
    ax.legend(title="Method" if show_titles else None, fontsize=16)
    fig.tight_layout()
    _save_mpl(fig, fig_dir, stem)

    # ── plotly (with linear/log x-scale dropdown) ────────────────────────
    hover = ("method: %{customdata[0]}<br>param: %{customdata[1]}=%{customdata[2]}<br>"
             "seed: %{customdata[3]}<br>solve_time: %{x:.2f}s<br>"
             "CP gap: %{y:.2f}%<extra></extra>")
    fig_px = go.Figure()
    shown: set[str] = set()
    for mod in modules:
        sub = cp_gap_raw[cp_gap_raw["module"] == mod]
        for param in sorted(sub["parameter"].unique()):
            s2 = sub[sub["parameter"] == param]
            fig_px.add_trace(go.Scatter(
                x=s2["solve_time"], y=s2["cp_gap"], mode="markers",
                name=MODULE_LABELS.get(mod, mod), legendgroup=mod,
                showlegend=(mod not in shown),
                marker=dict(color=MODULE_COLOURS.get(mod, "#333"), size=7,
                            opacity=0.65,
                            symbol=param_symbols_ply.get(param, "circle")),
                customdata=s2[["module", "parameter", "value", "seed"]].values,
                hovertemplate=hover,
            ))
            shown.add(mod)
    fig_px.add_hline(y=0, line_dash="dash", line_color="black", line_width=2,
                     annotation_text="CP+WS Best Solution (0%)",
                     annotation_position="top right")
    fig_px.update_layout(
        title=f"Quality–Time Trade-off: Heuristics vs CP+WS Best Solution{title_suffix}" if show_titles else None,
        xaxis_title="Heuristic Solve Time [s]" if show_titles else None,
        xaxis_type="log",
        yaxis_title="Gap to CP+WS Best Solution (%)" if show_titles else None,
        legend_title="Method" if show_titles else None,
        updatemenus=[dict(
            type="dropdown", direction="down", x=1.0, y=1.15,
            active=0,
            buttons=[
                dict(label="x: Log",    method="relayout",
                     args=[{"xaxis.type": "log"}]),
                dict(label="x: Linear", method="relayout",
                     args=[{"xaxis.type": "linear"}]),
            ],
        )],
    )
    _save_plotly(fig_px, int_dir, stem)
    print(f"  Saved {stem}" + (f" in {family_name}" if family_name else ""))


# ── LaTeX tables ──────────────────────────────────────────────────────────────

def _booktabs_table(df: pd.DataFrame, col_headers: list[str],
                    col_formats: list[str], caption: str, label: str,
                    short_caption: str = None, top_k: int = None, bottom_k: int = None,
                    omitted_text: str = None, tabcolsep: str = None) -> str:
    """Render DataFrame as a booktabs LaTeX table string."""
    col_spec = "l" + "r" * (len(col_headers) - 1)
    
    caption_str = rf"\caption[{short_caption}]{{{caption}}}" if short_caption else rf"\caption{{{caption}}}"
    
    lines = [r"\begin{table}[htbp]", r"\centering"]
    if tabcolsep:
        lines.append(rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}")
    lines.extend([
        r"\small",
        caption_str, rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_spec}}}", r"\toprule",
        " & ".join(f"\\textbf{{{h}}}" if not h.startswith("\\textbf") and not h.startswith("\\multicolumn") else h for h in col_headers) + r" \\",
        r"\midrule",
    ])
    
    if top_k is not None and bottom_k is not None and len(df) > (top_k + bottom_k):
        df_top = df.iloc[:top_k]
        df_bottom = df.iloc[-bottom_k:]
        omitted = len(df) - top_k - bottom_k
        omitted_line = rf"\multicolumn{{{len(col_headers)}}}{{c}}{{\emph{{\ldots\,{omitted}\,configurations omitted{omitted_text}}}}} \\"
        
        for _, row in df_top.iterrows():
            cells = []
            for col, fmt in zip(df.columns, col_formats):
                val = row[col]
                try:
                    cells.append(fmt % val if isinstance(val, (int, float)) else str(val))
                except (TypeError, ValueError):
                    cells.append(str(val))
            lines.append(" & ".join(cells) + r" \\")
            
        lines.append(r"\midrule")
        lines.append(omitted_line)
        lines.append(r"\midrule")
        
        for _, row in df_bottom.iterrows():
            cells = []
            for col, fmt in zip(df.columns, col_formats):
                val = row[col]
                try:
                    cells.append(fmt % val if isinstance(val, (int, float)) else str(val))
                except (TypeError, ValueError):
                    cells.append(str(val))
            lines.append(" & ".join(cells) + r" \\")
    else:
        for _, row in df.iterrows():
            cells = []
            for col, fmt in zip(df.columns, col_formats):
                val = row[col]
                try:
                    cells.append(fmt % val if isinstance(val, (int, float)) else str(val))
                except (TypeError, ValueError):
                    cells.append(str(val))
            lines.append(" & ".join(cells) + r" \\")
            
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)

def _longtable(df: pd.DataFrame, col_headers: list[str],
               col_formats: list[str], caption: str, label: str) -> str:
    """Render DataFrame as a longtable string."""
    col_spec = "l" + "r" * (len(col_headers) - 1)
    header_row = " & ".join(f"\\textbf{{{h}}}" if not h.startswith("\\textbf") and not h.startswith("\\multicolumn") else h for h in col_headers) + r" \\"
    
    # We replace "Sorting configurations" with "Config" for the continued header
    header_row_continued = header_row.replace("Sorting configurations", "Config")
    
    lines = [
        r"{\setlength{\tabcolsep}{4pt}%",
        rf"\begin{{longtable}}{{{col_spec}}}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}} \\",
        r"\toprule",
        header_row,
        r"\midrule",
        r"\endfirsthead",
        r"\caption[]{(\emph{continued})} \\",
        r"\toprule",
        header_row_continued,
        r"\midrule",
        r"\endhead",
        rf"\midrule \multicolumn{{{len(col_headers)}}}{{r}}{{\emph{{continued on next page}}}} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    
    for _, row in df.iterrows():
        cells = []
        for col, fmt in zip(df.columns, col_formats):
            val = row[col]
            try:
                cells.append(fmt % val if isinstance(val, (int, float)) else str(val))
            except (TypeError, ValueError):
                cells.append(str(val))
        lines.append(" & ".join(cells) + r" \\")
        
    lines += [r"\end{longtable}", r"}%"]
    return "\n".join(lines)



def write_config_table(
    gap_agg: pd.DataFrame,
    rank_agg: pd.DataFrame,
    winrate: pd.DataFrame,
    gap_moves_agg: pd.DataFrame,
    module: str,
    out_dir: Path,
    value_col: str = "objective",
    csv_dir: Path | None = None,
    show_titles: bool = False,
) -> None:
    """Write all per-config metrics as a booktabs .tex table, sorted by mean gap.

    Saved to out_dir/tables/{prefix}_configs.tex and out_dir/tables/{prefix}_configs_full.tex
    """
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    prefix = _module_prefix(module)
    csv_dir_actual = Path(out_dir) / "csv" if csv_dir is None else csv_dir

    df = (
        gap_agg[["config_id", "mean_gap", "std_gap"]]
        .merge(rank_agg[["config_id", "mean_rank", "median_rank"]], on="config_id", how="left")
        .merge(winrate[["config_id", "win_rate", "top3_rate", "within1_rate"]],
               on="config_id", how="left")
        .merge(gap_moves_agg[["config_id", "mean_gap"]].rename(
               columns={"mean_gap": "mean_moves_gap"}),
               on="config_id", how="left")
        .sort_values("mean_gap")
    )
    metrics.export_raw_csv(df, csv_dir_actual, f"{prefix}_configs")

    col_headers = [
        "\\textbf{Sorting configurations}", "$\\bar{\\Delta}$", "$\\sigma_{\\Delta}$",
        "$\\bar{r}$", "$\\tilde{r}$", "Win",
        "Top-3", "$\\bar{\\Delta}_m$"
    ]
    
    col_formats = ["%s", "%.2f", "%.2f", "%.2f", "%.1f", "%.3f", "%.3f", "%.2f"]
    cols = ["config_id", "mean_gap", "std_gap", "mean_rank", "median_rank",
            "win_rate", "top3_rate", "mean_moves_gap"]
            
    best_mode = ["", "min", "min", "min", "min", "max", "max", "min"]
    
    df_fmt = pd.DataFrame()
    for col, fmt, mode in zip(cols, col_formats, best_mode):
        if mode == "min":
            best_val = df[col].min()
            df_fmt[col] = df[col].apply(lambda x: r"\textbf{" + (fmt % x) + "}" if pd.notnull(x) and abs(x - best_val) < 1e-6 else (fmt % x if pd.notnull(x) else ""))
        elif mode == "max":
            best_val = df[col].max()
            df_fmt[col] = df[col].apply(lambda x: r"\textbf{" + (fmt % x) + "}" if pd.notnull(x) and abs(x - best_val) < 1e-6 else (fmt % x if pd.notnull(x) else ""))
        else:
            df_fmt[col] = df[col].apply(lambda x: str(x).replace("_", "\\_"))

    lbl = "AMA-SGC Full-mode" if module == "heuristic_ama_sgc" else "AMA-SGC Two-Phase"
    short_caption = f"{lbl} per-config metrics"
    num_configs = len(df)
    
    clean_lbl = lbl.replace("AMA-SGC Full-mode", "AMA-SGC full-grid").replace("AMA-SGC Two-Phase", "AMA-SGC two-phase")
    
    caption_main = (
        f"Top and bottom {clean_lbl} "
        f"configurations (of {num_configs}), sorted by mean objective gap. Best value in each column is \\textbf{{bold}}. "
        r"$\bar{\Delta}$: mean objective gap from per-instance best (\%), $\sigma_{\Delta}$: its standard deviation, "
        r"$\bar{r}$/$\tilde{r}$: mean/median rank among all configurations, Win: fraction of instances where the configuration achieves the best objective, "
        r"Top-3: fraction in the three best, $\bar{\Delta}_m$: mean total-moves gap (\%)."
    )
    
    omitted_text = f" (full ranking in \\cref{{tab:app-{prefix}_configs}})"
    tex = _booktabs_table(df_fmt, col_headers, ["%s"] * len(cols), caption_main,
                          f"tab:{prefix}_configs", short_caption=short_caption,
                          top_k=10, bottom_k=5, omitted_text=omitted_text, tabcolsep="4pt")
    path = tables_dir / f"{prefix}_configs.tex"
    path.write_text(tex, encoding="utf-8")
    print(f"  Saved {path}")
    
    caption_full = (
        f"All {num_configs} {clean_lbl} "
        f"configurations, sorted by mean objective gap. Best value in each column is \\textbf{{bold}}. "
        r"$\bar{\Delta}$: mean objective gap from per-instance best (\%), $\sigma_{\Delta}$: its standard deviation, "
        r"$\bar{r}$/$\tilde{r}$: mean/median rank among all configurations, Win: fraction of instances where the configuration achieves the best objective, "
        r"Top-3: fraction in the three best, $\bar{\Delta}_m$: mean total-moves gap (\%)."
    )
    
    tex_full = _longtable(df_fmt, col_headers, ["%s"] * len(cols), caption_full,
                          f"tab:app-{prefix}_configs")
    path_full = tables_dir / f"{prefix}_configs_full.tex"
    path_full.write_text(tex_full, encoding="utf-8")
    print(f"  Saved {path_full}")


def write_cross_heuristic_table(
    cp_gap_agg: pd.DataFrame,
    df_heuristics: pd.DataFrame,
    df_cp: pd.DataFrame,
    out_dir: Path,
    csv_dir: Path | None = None,
    show_titles: bool = False,
) -> None:
    """Write cross-heuristic summary as booktabs .tex table.

    Saved to out_dir/tables/cross_heuristic_summary.tex.
    """
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_dir_actual = Path(out_dir) / "csv" if csv_dir is None else csv_dir

    time_agg = (
        df_heuristics.groupby("module")["solve_time"]
        .median()
        .reset_index(name="median_solve_time")
    )
    df = (
        cp_gap_agg
        .merge(time_agg, on="module", how="left")
        .assign(method=lambda d: d["module"].map(MODULE_LABELS).fillna(d["module"]))
        .sort_values("mean_cp_gap")
    )
    cols = ["method", "mean_cp_gap", "median_cp_gap", "std_cp_gap", "median_solve_time"]
    metrics.export_raw_csv(df, csv_dir_actual, "cross_heuristic_summary")

    col_headers = ["Method", "Mean CP Gap (\\%)", "Median CP Gap (\\%)",
                   "Std CP Gap", "Median Time (s)"]
    col_formats = ["%s", "%.2f", "%.2f", "%.2f", "%.1f"]
    caption = (
        r"Cross-heuristic comparison. CP gap $= (\text{heuristic obj} - "
        r"\text{CP+WS Best obj}) / \text{CP+WS Best obj} \times 100\%$. "
        "CP+WS Best objective is the final incumbent from CP Optimizer with "
        "greedy warm-start (30-minute time limit)."
    )
    short_caption = "Cross-heuristic CP gap summary"
    tex = _booktabs_table(df[cols], col_headers, col_formats, caption,
                          "tab:cross_heuristic_summary", short_caption=short_caption)
    path = tables_dir / "cross_heuristic_summary.tex"
    path.write_text(tex, encoding="utf-8")
    print(f"  Saved {path}")


# ── Cross-heuristic scaling plots ───────────────────────────────────────────────

def plot_parameter_scaling(
    df_heuristics: pd.DataFrame,
    out_dir: Path,
    csv_dir: Path = None,
    family_name: str = None,
    show_titles: bool = False,
) -> None:
    """Scale plots for each parameter (e.g., orders, stations).

    Generates both matplotlib (PDF/PNG) and plotly (HTML) versions.
    Creates one pair of plots per unique parameter in the data.

    df_heuristics columns: module, parameter, value, objective_value, solve_time
    """
    fig_dir, int_dir, csv_dir_actual = _ensure_dirs(out_dir, subdir=family_name)

    # Group by parameter
    for param in sorted(df_heuristics["parameter"].unique()):
        subset = df_heuristics[df_heuristics["parameter"] == param].copy()
        if subset.empty:
            continue

        # Aggregate: median objective and solve_time per (module, value)
        agg = (
            subset.groupby(["module", "value"])
            .agg(
                obj_median=("objective_value", "median"),
                obj_min=("objective_value", "min"),
                obj_max=("objective_value", "max"),
                time_median=("solve_time", "median"),
            )
            .reset_index()
        )

        stem = f"{family_name}_scaling_{param}" if family_name else f"scaling_{param}"
        title_suffix = f" ({family_name.replace('_', ' + ')})" if family_name else ""
        if csv_dir is not None:
            csv_dir_actual = csv_dir / family_name if family_name else csv_dir
            csv_dir_actual.mkdir(parents=True, exist_ok=True)
        metrics.export_raw_csv(agg, csv_dir_actual, stem)

        # ── Matplotlib ──
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        modules = agg["module"].unique().tolist()
        modules = sorted(modules, key=lambda m: MODULE_ORDER.index(m) if m in MODULE_ORDER else 999)

        for mod in modules:
            data = agg[agg["module"] == mod]
            color = MODULE_COLOURS.get(mod, "#333")
            ax1.plot(data["value"], data["obj_median"], marker="o", label=MODULE_LABELS.get(mod, mod), color=color)
            ax2.plot(data["value"], data["time_median"], marker="s", linestyle="--", label=MODULE_LABELS.get(mod, mod), color=color)

        ax1.set_xlabel(param.capitalize())
        ax1.set_ylabel("Median Objective Value")
        if show_titles:

            ax1.set_title(f"Scaling: {param} vs Objective (Makespan){title_suffix}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel(param.capitalize())
        ax2.set_ylabel("Median Runtime [s]")
        if show_titles:

            ax2.set_title(f"Scaling: {param} vs Runtime{title_suffix}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        _save_mpl(fig, fig_dir, stem)

        # ── Plotly ──
        fig_obj = go.Figure()
        fig_time = go.Figure()
        for mod in modules:
            data = agg[agg["module"] == mod]
            color = MODULE_COLOURS.get(mod, "#333")
            fig_obj.add_trace(go.Scatter(
                x=data["value"], y=data["obj_median"], mode="lines+markers",
                name=MODULE_LABELS.get(mod, mod),
                line=dict(color=color)
            ))
            fig_time.add_trace(go.Scatter(
                x=data["value"], y=data["time_median"], mode="lines+markers",
                name=MODULE_LABELS.get(mod, mod),
                line=dict(color=color)
            ))

        fig_obj.update_layout(title=f"Scaling: {param} vs Objective{title_suffix}" if show_titles else None, xaxis_title=param.capitalize(), yaxis_title="Median Objective Value")
        fig_time.update_layout(title=f"Scaling: {param} vs Runtime{title_suffix}" if show_titles else None, xaxis_title=param.capitalize(), yaxis_title="Median Runtime [s]")

        _save_plotly(fig_obj, int_dir, f"{stem}_objective")
        _save_plotly(fig_time, int_dir, f"{stem}_time")

    print(f"  Saved scaling plots for parameters" + (f" in {family_name}" if family_name else ""))


def plot_performance_profile(
    df_merged: pd.DataFrame,
    out_dir: Path,
    csv_dir: Path = None,
    family_name: str = None,
    show_titles: bool = False,
) -> None:
    """Performance profile (Dolan-More) plot comparing heuristics to CP baseline.

    df_merged columns: module, ratio (heuristic_obj / cp_best_obj)
    """
    fig_dir, int_dir, csv_dir_actual = _ensure_dirs(out_dir, subdir=family_name)
    stem = f"{family_name}_perf_profile_quality" if family_name else "perf_profile_quality"
    title_suffix = f" ({family_name.replace('_', ' + ')})" if family_name else ""
    if csv_dir is not None:
        csv_dir_actual = csv_dir / family_name if family_name else csv_dir
        csv_dir_actual.mkdir(parents=True, exist_ok=True)
    metrics.export_raw_csv(df_merged, csv_dir_actual, stem)

    # ── Matplotlib ──
    fig, ax = plt.subplots(figsize=(10, 6))
    for mod in df_merged["module"].unique():
        data = df_merged[df_merged["module"] == mod]["ratio"].sort_values()
        y_vals = pd.Series(range(1, len(data) + 1)) / len(data)
        ax.step(data, y_vals, where="post", label=MODULE_LABELS.get(mod, mod))

    ax.set_xlim(0.95, 1.5)
    ax.set_xlabel(r"Performance Ratio $\tau$ (Lower is Better)")
    ax.set_ylabel(r"Probability($r \leq \tau$)")
    if show_titles:

        ax.set_title(f"Performance Profile: Quality vs Best Known CP{title_suffix}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_mpl(fig, fig_dir, stem)

    # ── Plotly ──
    fig_px = go.Figure()
    for mod in df_merged["module"].unique():
        data = df_merged[df_merged["module"] == mod]["ratio"].sort_values()
        y_vals = pd.Series(range(1, len(data) + 1)) / len(data)
        fig_px.add_trace(go.Scatter(
            x=data, y=y_vals, mode="lines",
            name=MODULE_LABELS.get(mod, mod),
            line_shape="vh"
        ))

    fig_px.update_xaxes(range=[0.95, 1.5])
    fig_px.update_layout(
        title=f"Performance Profile: Quality vs Best Known CP{title_suffix}" if show_titles else None,
        xaxis_title="Performance Ratio $\tau$ (Lower is Better)" if show_titles else None,
        yaxis_title="Probability($r \leq \tau$)" if show_titles else None,
    )
    _save_plotly(fig_px, int_dir, stem)
    print(f"  Saved {stem}" + (f" in {family_name}" if family_name else ""))


def plot_gap_vs_time(
    df_merged: pd.DataFrame,
    out_dir: Path,
    csv_dir: Path = None,
    family_name: str = None,
    show_titles: bool = False,
) -> None:
    """Scatter plot of gap to CP vs solve time.

    df_merged columns: module, parameter, value, seed, solve_time, cp_gap
    """
    fig_dir, int_dir, csv_dir_actual = _ensure_dirs(out_dir, subdir=family_name)
    stem = f"{family_name}_scatter_gap_vs_time" if family_name else "scatter_gap_vs_time"
    title_suffix = f" ({family_name.replace('_', ' + ')})" if family_name else ""
    if csv_dir is not None:
        csv_dir_actual = csv_dir / family_name if family_name else csv_dir
        csv_dir_actual.mkdir(parents=True, exist_ok=True)
    metrics.export_raw_csv(df_merged, csv_dir_actual, stem)

    # ── Matplotlib ──
    fig, ax = plt.subplots(figsize=(10, 6))
    modules = df_merged["module"].unique()
    colors = plt.cm.tab10.colors

    for i, mod in enumerate(modules):
        subset = df_merged[df_merged["module"] == mod]
        ax.scatter(
            subset["solve_time"], subset["cp_gap"],
            alpha=0.6, label=MODULE_LABELS.get(mod, mod),
            edgecolors="w", color=colors[i % len(colors)]
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=1, label="CP Baseline")
    ax.set_xscale("log")
    ax.set_xlabel("Solve Time [s]")
    ax.set_ylabel("Gap to Best CP [%]")
    if show_titles:

        ax.set_title(f"Cost-Time Tradeoff{title_suffix}")
    ax.legend(title="Method")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    fig.tight_layout()
    _save_mpl(fig, fig_dir, stem)

    # ── Plotly ──
    fig_px = go.Figure()
    for mod in df_merged["module"].unique():
        subset = df_merged[df_merged["module"] == mod]
        fig_px.add_trace(go.Scatter(
            x=subset["solve_time"], y=subset["cp_gap"],
            mode="markers", name=MODULE_LABELS.get(mod, mod),
            hovertemplate="%{customdata[0]}=%{customdata[1]}<br>seed: %{customdata[2]}<br>solve_time: %{x:.2f}s<br>gap: %{y:.2f}%<extra></extra>",
            customdata=subset[["parameter", "value", "seed"]].values
        ))

    fig_px.add_hline(y=0, line_dash="dash", line_color="black", annotation_text="CP Baseline (0%)")
    fig_px.update_xaxes(type="log")
    fig_px.update_layout(
        title=f"Cost-Time Tradeoff{title_suffix}" if show_titles else None,
        xaxis_title="Solve Time [s] (log scale)" if show_titles else None,
        yaxis_title="Gap to Best CP [%]" if show_titles else None,
    )
    _save_plotly(fig_px, int_dir, stem)
    print(f"  Saved {stem}" + (f" in {family_name}" if family_name else ""))


