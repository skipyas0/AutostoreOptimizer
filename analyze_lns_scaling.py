"""Tables and figures for benchmark_lns_scaling.py.

Usage: python analyze_lns_scaling.py [results/lns_scaling]
Writes to <dir>/analysis: table_scaling.{csv,md}, table_breakeven.{csv,md},
fig_overhead_vs_size, fig_overhead_composition, fig_breakeven_fraction (.pdf/.png).
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
MODEL_COLORS = {"full": BLUE, "subproblem": ORANGE}
MODEL_LABELS = {"full": "Full model", "subproblem": "Subproblem"}
COMPONENTS = {  # column -> (label, color); categorical slots in fixed order
    "solve_cpp_presolve": ("CP extraction", "#2a78d6"),
    "build": ("Constraint gen. / model build", "#eb6834"),
    "constr_update_time": ("Constraint update", "#1baf7a"),
    "solve_api_overhead": ("API overhead", "#eda100"),
    "merge": ("Merge", "#6250d6"),
}
SIZE_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
             "#1c5cab", "#184f95", "#104281", "#0d366b"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "legend.frameon": False, "lines.linewidth": 2,
})


def _size_label(stations, orders):
    return f"{stations}/{orders}"


def _md_table(df, floatfmt="{:.2f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                 for v in r]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def load(root):
    it = pd.read_csv(os.path.join(root, "lns_iterations.csv"))
    runs = pd.read_csv(os.path.join(root, "lns_runs.csv"))
    it["build"] = it["constr_generate_time"].clip(lower=0)
    it["merge"] = it["merge_time"].clip(lower=0)
    it["overhead"] = it[list(COMPONENTS)].clip(lower=0).sum(axis=1)
    it["size"] = [_size_label(s, o) for s, o in zip(it.stations, it.orders)]
    nbh_path = os.path.join(root, "nbh_sweep.csv")
    nbh = pd.read_csv(nbh_path) if os.path.exists(nbh_path) else None
    if nbh is not None:
        nbh["overhead"] = nbh[["t_context", "t_build", "t_api", "t_extraction",
                               "t_merge"]].sum(axis=1)
    return it, runs, nbh


def scaling_table(it, runs):
    g = it.groupby(["stations", "orders", "lns_model"])
    agg = g.agg(num_variables=("num_variables", "first"),
                overhead_med=("overhead", "median"),
                overhead_q1=("overhead", lambda x: x.quantile(0.25)),
                overhead_q3=("overhead", lambda x: x.quantile(0.75)),
                extraction_med=("solve_cpp_presolve", "median"),
                search_med=("solve_cpp_search", "median"),
                unfrozen_orders_med=("n_unfrozen_orders", "median"),
                iters=("iteration", "count")).reset_index()
    agg = agg.merge(runs[["stations", "orders", "lns_model", "setup_time"]],
                    on=["stations", "orders", "lns_model"])
    wide = agg.pivot(index=["stations", "orders"], columns="lns_model")
    t = pd.DataFrame({
        "size (stations/orders)": [_size_label(s, o) for s, o in wide.index],
        "variables (full)": wide[("num_variables", "full")].astype(int).values,
        "active orders (med.)": wide[("unfrozen_orders_med", "subproblem")].values,
        "full overhead [s]": wide[("overhead_med", "full")].values,
        "sub overhead [s]": wide[("overhead_med", "subproblem")].values,
        "sub/full": (wide[("overhead_med", "subproblem")]
                     / wide[("overhead_med", "full")]).values,
        "full setup [s]": wide[("setup_time", "full")].values,
        "sub setup [s]": wide[("setup_time", "subproblem")].values,
    })
    return agg, t


def breakeven(nbh, agg):
    """Order fraction at which subproblem overhead reaches the full model's median."""
    full = agg[agg.lns_model == "full"].set_index(["stations", "orders"])
    rows = []
    for (s, o), grp in nbh.groupby(["stations", "orders"]):
        if (s, o) not in full.index:
            continue
        ref = full.loc[(s, o), "overhead_med"]
        med = grp.groupby("fraction")["overhead"].median()
        ratio = med / ref
        frac = np.nan
        above = np.where(ratio.values >= 1.0)[0]
        if len(above) and above[0] == 0:
            frac = 0.0  # already more expensive at the smallest neighbourhood
        elif len(above):
            i = above[0]
            x0, x1 = ratio.index[i - 1], ratio.index[i]
            y0, y1 = ratio.values[i - 1], ratio.values[i]
            frac = x0 + (1.0 - y0) * (x1 - x0) / (y1 - y0)
        rows.append({
            "size (stations/orders)": _size_label(s, o),
            "variables (full)": int(full.loc[(s, o), "num_variables"]),
            "full overhead [s]": ref,
            "sub overhead @10% [s]": med.get(0.1, np.nan),
            "sub overhead @100% [s]": med.get(1.0, np.nan),
            "@100% / full": med.get(1.0, np.nan) / ref,
            "break-even order fraction": frac,
        })
    return pd.DataFrame(rows)


def fig_overhead_vs_size(agg, out):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for m in ("full", "subproblem"):
        d = agg[agg.lns_model == m].sort_values("num_variables")
        ax.fill_between(d.num_variables, d.overhead_q1, d.overhead_q3,
                        color=MODEL_COLORS[m], alpha=0.15, linewidth=0)
        ax.plot(d.num_variables, d.overhead_med, color=MODEL_COLORS[m], marker="o",
                markersize=5, label=MODEL_LABELS[m])
        last = d.iloc[-1]
        ax.annotate(MODEL_LABELS[m], (last.num_variables, last.overhead_med),
                    xytext=(6, 0), textcoords="offset points", va="center", color=INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Interval variables in the full model")
    ax.set_ylabel("Per-iteration overhead [s]\n(median, IQR band)")
    ax.set_title("LNS overhead outside the CP search", loc="left", color=INK)
    ax.legend(loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig_overhead_vs_size.{ext}"), dpi=200)
    plt.close(fig)


def fig_composition(it, out):
    med = it.groupby(["stations", "orders", "lns_model"])[list(COMPONENTS)].median()
    med = med.clip(lower=0).reset_index().sort_values(["stations", "orders"])
    sizes = med[["stations", "orders"]].drop_duplicates().values.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharey=True)
    for ax, m in zip(axes, ("full", "subproblem")):
        d = med[med.lns_model == m].set_index(["stations", "orders"])
        x = np.arange(len(sizes))
        bottom = np.zeros(len(sizes))
        for col, (label, color) in COMPONENTS.items():
            vals = np.array([d.loc[tuple(s), col] if tuple(s) in d.index else 0
                             for s in sizes])
            ax.bar(x, vals, bottom=bottom, color=color, width=0.7, label=label,
                   edgecolor="white", linewidth=1)
            bottom += vals
        ax.set_xticks(x, [_size_label(*s) for s in sizes], rotation=45)
        ax.set_title(MODEL_LABELS[m], loc="left", color=INK)
        ax.set_xlabel("Instance size (stations/orders)")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Median time per iteration [s]")
    axes[1].legend(loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig_overhead_composition.{ext}"), dpi=200)
    plt.close(fig)


def fig_breakeven(nbh, agg, out):
    full = agg[agg.lns_model == "full"].set_index(["stations", "orders"])
    groups = sorted(nbh.groupby(["stations", "orders"]), key=lambda g: g[0][0] * 1e4 + g[0][1])
    colors = SIZE_RAMP[-len(groups):] if len(groups) <= len(SIZE_RAMP) else SIZE_RAMP
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for ((s, o), grp), c in zip(groups, colors):
        # below CP Optimizer's 0.01 s timer resolution the ratio is noise
        if (s, o) not in full.index or full.loc[(s, o), "overhead_med"] < 0.01:
            continue
        med = grp.groupby("fraction")["overhead"].median() / full.loc[(s, o), "overhead_med"]
        ax.plot(med.index * 100, med.values, color=c, marker="o", markersize=4,
                label=_size_label(s, o))
    ax.axhline(1.0, color=INK, linewidth=1, linestyle="--")
    ax.annotate("equal to full-model overhead", (3, 1.0), xytext=(0, 4),
                textcoords="offset points", ha="left", va="bottom", color=MUTED)
    ax.set_yscale("log")
    ax.set_xlabel("Active orders in the neighbourhood [%]")
    ax.set_ylabel("Subproblem overhead / full-model overhead")
    ax.set_title("When does building the subproblem cost as much as\n"
                 "re-extracting the full model?", loc="left", color=INK)
    ax.legend(title="stations/orders", ncol=2, fontsize=8, title_fontsize=8,
              loc="lower right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig_breakeven_fraction.{ext}"), dpi=200)
    plt.close(fig)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "results/lns_scaling"
    out = os.path.join(root, "analysis")
    os.makedirs(out, exist_ok=True)
    it, runs, nbh = load(root)
    agg, table = scaling_table(it, runs)
    table.to_csv(os.path.join(out, "table_scaling.csv"), index=False)
    with open(os.path.join(out, "table_scaling.md"), "w") as f:
        f.write(_md_table(table))
    print(_md_table(table))
    fig_overhead_vs_size(agg, out)
    fig_composition(it, out)
    if nbh is not None:
        be = breakeven(nbh, agg)
        be.to_csv(os.path.join(out, "table_breakeven.csv"), index=False)
        with open(os.path.join(out, "table_breakeven.md"), "w") as f:
            f.write(_md_table(be))
        print(_md_table(be))
        fig_breakeven(nbh, agg, out)

    # Does full-model extraction depend on the neighbourhood size? (per instance)
    full = it[it.lns_model == "full"]
    corr = full.groupby("size").apply(
        lambda d: d["solve_cpp_presolve"].corr(d["num_optimized"]), include_groups=False)
    print("corr(full extraction, #optimized vars) per size:\n", corr.round(2).to_string())


if __name__ == "__main__":
    main()
