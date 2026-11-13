"""Figures on the long run table. Every function takes the methods to show (in
legend order) and a `look` dict method -> {label, color, ls, family}, so the same
function draws the full and the reduced variant."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from . import metrics, style

BOX = dict(showfliers=False, medianprops=dict(color=style.INK, lw=1.2),
           boxprops=dict(color=style.MUTED, lw=0.8),
           whiskerprops=dict(color=style.MUTED, lw=0.8),
           capprops=dict(color=style.MUTED, lw=0.8))


def _spread(ys, min_gap):
    """Label positions at least min_gap apart, each crowded group centred on its marks."""
    order = np.argsort(ys)
    groups = [[i] for i in order]  # index lists, bottom to top
    merged = True
    while merged:
        merged = False
        pos = [np.mean([ys[i] for i in g]) for g in groups]
        for k in range(len(groups) - 1):
            span_lo = pos[k] + (len(groups[k]) - 1) * min_gap / 2
            span_hi = pos[k + 1] - (len(groups[k + 1]) - 1) * min_gap / 2
            if span_hi - span_lo < min_gap:
                groups[k] += groups.pop(k + 1)
                merged = True
                break
    out = np.empty(len(ys))
    for g in groups:
        centre = np.mean([ys[i] for i in g])
        for j, i in enumerate(g):
            out[i] = centre + (j - (len(g) - 1) / 2) * min_gap
    return out


def _time_axis(ax, t_min, time_limit, right=None):
    from matplotlib.ticker import FixedLocator, NullFormatter
    ticks = [t for t in (0.1, 1, 10, 60, 300, 900, 3600) if t_min <= t <= time_limit]
    if time_limit not in ticks:
        ticks.append(time_limit)
    ax.set_xscale("log")
    ax.set_xlim(t_min, right or time_limit)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())


def convergence(runs, methods, look, sizes, time_limit, out, name, bands=False,
                title=None, t_min=1.0, highlight=(), extend=1.7):
    """Mean gap to BKS over time (log time axis), one panel per size class.

    Anytime methods: mean over runs, drawn where every run has a solution
    (± std band if bands=True), with a dot at the time limit; methods in `highlight`
    get their final value written next to that dot. Heuristics: a red star at their
    mean runtime and a dashed reference line from there on, labelled at the right edge.
    The time axis runs `extend` times past the limit so the end points stay readable.
    """
    from matplotlib.lines import Line2D
    heur_rt = runs.loc[runs.method.isin(methods), "runtime"] if "runtime" in runs else None
    if heur_rt is not None and heur_rt.notna().any():  # keep fast heuristics on the plot
        t_min = min(t_min, 0.6 * heur_rt.min())
    grid = np.geomspace(t_min, time_limit, 400)
    right = time_limit * extend
    longest = max((len(look[m]["label"]) for m in methods
                   if look[m]["family"] == "heuristic"), default=0)
    panel_w = 3.9 + 0.055 * longest  # room for the heuristic labels right of each panel
    fig, axes = plt.subplots(1, len(sizes), figsize=(panel_w * len(sizes), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    ymax = 0
    labels_per_ax, ends_per_ax = [], []
    for ax, size in zip(axes, sizes):
        r_size = runs[runs["size"] == size]
        heur_labels, ends = [], []
        for m in methods:
            r = r_size[r_size.method == m]
            if not len(r):
                continue
            lk = look[m]
            if lk["family"] == "heuristic":
                g, rt = r.gap_final.mean(), r.runtime.mean()
                ax.hlines(g, rt, right, color=lk["color"], lw=1.1,
                          ls=(0, (4, 2)), zorder=2)
                ax.plot(rt, g, marker="*", ms=11, color=style.HEURISTIC_MARK,
                        mec="white", mew=0.8, ls="none", zorder=7)
                heur_labels.append((g, lk["label"]))
                ymax = max(ymax, g)
                continue
            c = metrics.curves(r, grid)
            valid = np.isfinite(c).all(axis=0)
            mean = np.full(grid.shape, np.nan)
            mean[valid] = c[:, valid].mean(axis=0)
            ax.plot(grid, mean, color=lk["color"], ls=lk.get("ls", "-"),
                    lw=lk.get("lw", 1.8), label=lk["label"], zorder=4)
            ax.plot(time_limit, mean[-1], "o", ms=5, color=lk["color"], mec="white",
                    mew=1.2, zorder=5)
            if m in highlight:
                ends.append((mean[-1], lk))
            if bands:
                sd = np.full(grid.shape, np.nan)
                if len(c) > 1:
                    sd[valid] = c[:, valid].std(axis=0, ddof=1)
                ax.fill_between(grid, mean - sd, mean + sd, color=lk["color"],
                                alpha=0.12, lw=0)
            ymax = max(ymax, np.nanmax(mean[grid >= 30]) if valid[grid >= 30].any() else 0)
        labels_per_ax.append(heur_labels)
        ends_per_ax.append(ends)
        n_inst = r_size.instance.nunique()
        ax.set_title(f"{size} ({n_inst} instances)")
        _time_axis(ax, t_min, time_limit, right)
        ax.axvline(time_limit, color=style.GRID_STRONG, lw=1, zorder=1)
        ax.set_xlabel("Time [s] (log scale)")
        ax.axhline(0, color=style.MUTED, lw=0.9, zorder=3)
    axes[0].set_ylabel("Gap to best known solution [%]")
    axes[0].set_ylim(bottom=-0.09 * ymax if ymax else -0.1, top=1.1 * ymax if ymax else 1)
    lo, hi = axes[0].get_ylim()
    for ax, heur_labels, ends in zip(axes, labels_per_ax, ends_per_ax):
        ax.annotate(f"{time_limit:g} s limit", (time_limit, hi), xytext=(3, -2),
                    textcoords="offset points", ha="left", va="top", fontsize=6.5,
                    color=style.MUTED)
        for v, lk in ends:
            ax.annotate(f"{v:.2f} %", (time_limit, v), xytext=(6, 4),
                        textcoords="offset points", ha="left", va="bottom", fontsize=7,
                        color=style.INK, fontweight="bold")
        if heur_labels:
            ys = _spread([y for y, _ in heur_labels], min_gap=0.055 * (hi - lo))
            for y, (_, lab) in zip(ys, heur_labels):
                ax.annotate(lab, (right, y), xytext=(4, 0), textcoords="offset points",
                            va="center", fontsize=7, color=style.MUTED,
                            annotation_clip=False)
    axes[0].annotate("0 = best known solution (any method)",
                     (t_min, 0), xytext=(3, -3), textcoords="offset points",
                     ha="left", va="top", fontsize=6.5, color=style.MUTED)
    any_labels = any(labels_per_ax)
    handles, labels = axes[0].get_legend_handles_labels()
    if any_labels:
        handles.append(Line2D([], [], color=style.HEURISTIC_COLOR, lw=1.1, ls=(0, (4, 2)),
                              marker="*", ms=10, mfc=style.HEURISTIC_MARK, mec="white"))
        labels.append("Heuristic (star: mean runtime)")
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)),
               bbox_to_anchor=(0.5, -0.12 if len(labels) <= 4 else -0.2))
    if title:
        fig.suptitle(title, y=1.02, fontsize=10)
    fig.subplots_adjust(wspace=0.1 + 0.06 * longest / panel_w * 3.6 if any_labels else 0.1)
    style.save(fig, out, name)


def final_distribution(runs, methods, look, sizes, out, name, title=None):
    """Final gap to BKS per method (box + runs as dots), one panel per size class."""
    rng = np.random.default_rng(0)
    h = 0.32 * len(methods) + 0.9
    fig, axes = plt.subplots(1, len(sizes), figsize=(3.4 * len(sizes), h), sharey=True)
    axes = np.atleast_1d(axes)
    y = np.arange(len(methods))
    for ax, size in zip(axes, sizes):
        r = runs[runs["size"] == size]
        for i, m in enumerate(methods):
            vals = r[r.method == m].gap_final.dropna().values
            if not len(vals):
                continue
            ax.boxplot(vals, positions=[i], widths=0.55, orientation="horizontal", **BOX)
            ax.scatter(vals, i + rng.uniform(-0.17, 0.17, len(vals)), s=12,
                       color=look[m]["color"], edgecolor="white", lw=0.5, zorder=3)
        ax.set_title(size)
        ax.set_xlabel("Final gap to best known [%]")
        ax.grid(axis="y", visible=False)
        ax.axvline(0, color=style.MUTED, lw=0.8)
    axes[0].set_yticks(y, [look[m]["label"] for m in methods])
    axes[0].invert_yaxis()
    fig.subplots_adjust(top=1 - 0.6 / h)
    if title:
        fig.suptitle(title, y=1 - 0.1 / h, fontsize=10)
    style.save(fig, out, name)


def mean_ranks(runs, methods, look, sizes, out, name, title=None):
    """Mean rank per method over all instances (bar) and per size class (markers)."""
    rk = metrics.ranks(runs[runs.method.isin(methods)])
    size_of = runs.drop_duplicates("instance").set_index("instance")["size"]
    fig_h = 0.3 * len(methods) + 1.0
    fig, ax = plt.subplots(figsize=(6, fig_h))
    y = np.arange(len(methods))
    overall = rk.mean().reindex(methods)
    ax.barh(y, overall.values, 0.6, color=[look[m]["color"] for m in methods])
    markers = ["o", "s", "^", "D", "v", "P"]
    for k, size in enumerate(sizes):
        rs = rk[rk.index.map(size_of) == size].mean().reindex(methods)
        ax.scatter(rs.values, y, marker=markers[k % len(markers)], s=22,
                   facecolor="white", edgecolor=style.INK, lw=0.9, zorder=3,
                   label=f"mean rank, {size}")
    # overall value in a column right of the plot (keeps clear of the markers)
    for yi, v in zip(y, overall.values):
        ax.annotate(f"{v:.2f}", (1.0, yi), xycoords=("axes fraction", "data"),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=7,
                    color=style.MUTED)
    ax.annotate("mean", (1.0, 1.0), xycoords="axes fraction", xytext=(6, 2),
                textcoords="offset points", va="bottom", fontsize=7, color=style.MUTED)
    ax.set_yticks(y, [look[m]["label"] for m in methods])
    ax.invert_yaxis()
    ax.set_xlim(0.5, len(methods) + 0.5)
    ax.set_xlabel(f"Mean rank over instances (1 = best of {len(methods)}); "
                  "bar = all instances")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.6 / (0.77 * fig_h)),
              ncol=len(sizes), handletextpad=0.2, columnspacing=1.2)
    if title:
        ax.set_title(title)
    style.save(fig, out, name)


def instance_heatmap(runs, methods, look, instances, out, name, title=None):
    """Mean final gap to BKS per method (rows) and instance (columns)."""
    inst = instances.sort_values(["orders", "gen_seed"])
    gap = metrics.instance_means(runs, "gap_final").reindex(index=inst.instance,
                                                             columns=methods).T
    cmap = LinearSegmentedColormap.from_list("blue", style.BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(0.5 * gap.shape[1] + 2.6, 0.32 * len(methods) + 1.2))
    vmax = np.nanpercentile(gap.values, 95)
    im = ax.imshow(gap.values, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    for i in range(gap.shape[0]):
        for j in range(gap.shape[1]):
            v = gap.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6,
                        color="white" if v > 0.55 * vmax else style.INK)
    ax.set_yticks(range(len(methods)), [look[m]["label"] for m in methods])
    ax.set_xticks(range(len(inst)), [f"{r.size[0]}{r.gen_seed}" for r in inst.itertuples()],
                  fontsize=7)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    bounds = np.flatnonzero(inst["size"].values[1:] != inst["size"].values[:-1])
    for b in bounds:
        ax.axvline(b + 0.5, color="white", lw=3)
    ax.set_xlabel("Instance (size initial + generator seed)")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Mean final gap to best known [%]")
    cb.outline.set_visible(False)
    if title:
        ax.set_title(title)
    style.save(fig, out, name)


def heuristic_quality_time(runs, methods, look, sizes, out, name, title=None):
    """Heuristics: mean runtime vs mean gap to BKS, one panel per size class."""
    fig, axes = plt.subplots(1, len(sizes), figsize=(3.6 * len(sizes), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, size in zip(axes, sizes):
        r = runs[(runs["size"] == size) & runs.method.isin(methods)]
        agg = r.groupby("method").agg(rt=("runtime", "mean"), g=("gap_final", "mean"),
                                      gsd=("gap_final", "std"))
        for m, a in agg.iterrows():
            ax.errorbar(a.rt, a.g, yerr=a.gsd, fmt="o", ms=5, color=style.INK,
                        ecolor=style.MUTED, elinewidth=0.8, capsize=2, mec="white")
        ys = _spread(agg.g.values, min_gap=0.06 * max(agg.g.max(), 1))
        for (m, a), yl in zip(agg.iterrows(), ys):
            ax.annotate(look[m]["label"], (a.rt, a.g), xytext=(a.rt * 1.35, yl),
                        textcoords="data", va="center", fontsize=7, color=style.MUTED)
        lo, hi = agg.rt.min(), agg.rt.max()
        ax.set_xscale("log")
        ax.set_xlim(lo / 2, hi * 10 ** 1.5)  # room for the labels right of the points
        from matplotlib.ticker import FixedLocator, NullFormatter
        ticks = [t for t in (0.1, 1, 10, 100, 1000) if lo / 2 <= t <= hi * 10 ** 1.5]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(size)
        ax.set_xlabel("Mean runtime [s] (log scale)")
    axes[0].set_ylabel("Gap to best known [%] (mean ± std)")
    if title:
        fig.suptitle(title, y=1.03, fontsize=10)
    style.save(fig, out, name)


def lns_time_composition(iters, methods, look, sizes, out, name, title=None):
    """Mean time per LNS iteration split into its parts, per config and size."""
    it = iters.copy()
    parts = [p for p in style.TIME_PARTS if p[0] in it.columns]
    it["other"] = (it.total_iter_time - it[[c for c, _, _ in parts]].sum(axis=1)).clip(lower=0)
    fig, axes = plt.subplots(1, len(sizes), figsize=(3.6 * len(sizes), 0.3 * len(methods) + 1.3),
                             sharey=True)
    axes = np.atleast_1d(axes)
    y = np.arange(len(methods))
    for ax, size in zip(axes, sizes):
        g = it[it["size"] == size].groupby("method")
        left = np.zeros(len(methods))
        for c, label, col in parts + [("other", "Other / logging", style.OTHER_COLOR)]:
            vals = g[c].mean().reindex(methods).fillna(0).values
            ax.barh(y, vals, 0.6, left=left, color=col, label=label, edgecolor="white", lw=1)
            left += vals
        ax.set_title(size)
        ax.set_xlabel("Mean time per iteration [s]")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [look[m]["label"] for m in methods])
    axes[0].invert_yaxis()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.22))
    if title:
        fig.suptitle(title, y=1.04, fontsize=10)
    style.save(fig, out, name)


def strategy_heatmap(table, out, name, value, column="Method", title=None, fmt="{:.0f}"):
    """Strategy (rows) x `column` heatmap of a numeric column (rows sorted by mean)."""
    cols = list(pd.unique(table[column]))
    piv = table.pivot_table(index="Strategy", columns=column, values=value,
                            aggfunc="sum")[cols]
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
    cmap = LinearSegmentedColormap.from_list("blue", style.BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(0.75 * piv.shape[1] + 3.2, 0.27 * len(piv) + 1.0))
    vmax = np.nanmax(piv.values)
    im = ax.imshow(piv.values, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6.5,
                        color="white" if v > 0.55 * vmax else style.INK)
    ax.set_yticks(range(len(piv)), [s.replace("_", " ") for s in piv.index], fontsize=7)
    ax.set_xticks(range(piv.shape[1]), [c.replace(" · ", "\n") for c in piv.columns],
                  fontsize=7.5)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(value)
    cb.outline.set_visible(False)
    if title:
        ax.set_title(title)
    style.save(fig, out, name)


# --------------------------------------------------------------------------- #
# Solution structure (solutions.py)
# --------------------------------------------------------------------------- #
SIZE_MARKERS = ["o", "s", "^", "D", "v", "P"]


def solution_structure(sol, methods, look, sizes, columns, labels, out, name, title=None,
                       ncols=4):
    """One panel per metric; per method one marker per size class (mean of the
    instance means), joined by a thin line in the method color.
    labels: column -> (label, unit)."""
    methods = [m for m in methods if (sol.method == m).any()]
    inst = sol.groupby(["method", "size", "instance"])[columns].mean().reset_index()
    means = inst.groupby(["method", "size"])[columns].mean()
    nrows = int(np.ceil(len(columns) / ncols))
    fig_h = nrows * (0.24 * len(methods) + 0.9) + 0.4
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols + 1.6, fig_h),
                             sharey=True, squeeze=False)
    y = np.arange(len(methods))
    for ax, col in zip(axes.flat, columns):
        for yi, m in zip(y, methods):
            vals = [means.loc[(m, s), col] if (m, s) in means.index else np.nan for s in sizes]
            ax.plot([np.nanmin(vals), np.nanmax(vals)], [yi, yi], color=look[m]["color"],
                    lw=1.2, alpha=0.6, zorder=2)
            for k, (s, v) in enumerate(zip(sizes, vals)):
                ax.scatter(v, yi, marker=SIZE_MARKERS[k % 6], s=22, color=look[m]["color"],
                           edgecolor="white", lw=0.6, zorder=3)
        label, unit = labels[col]
        ax.set_title(label, fontsize=8.5)
        ax.set_xlabel(f"[{unit}]" if unit else "", fontsize=7)
        ax.grid(axis="y", visible=False)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(columns):]:
        ax.set_visible(False)
    for row in axes:
        row[0].set_yticks(y, [look[m]["label"] for m in methods], fontsize=7)
    axes[0, 0].invert_yaxis()
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker=SIZE_MARKERS[k % 6], ls="none", color=style.MUTED,
                      mec="white", ms=6, label=s) for k, s in enumerate(sizes)]
    fig.legend(handles=handles, loc="lower center", ncol=len(sizes),
               bbox_to_anchor=(0.55, -0.03 - 0.25 / fig_h),
               title="mean over instances of the size class", title_fontsize=7)
    if title:
        fig.suptitle(title, y=1.0 + 0.1 / fig_h, fontsize=10)
    fig.tight_layout()
    style.save(fig, out, name)


def work_vs_utilisation(sol, methods, look, sizes, ref, out, name, title=None):
    """Makespan = robot work / (MoveCap x robot utilisation), so in log terms
    change of makespan = change of work - change of utilisation (all vs. `ref`).
    One panel per size: x = robot work, y = robot utilisation, both relative to the
    reference method on the same instance; diagonals are equal makespan change."""
    methods = [m for m in methods if (sol.method == m).any()]
    inst = sol.groupby(["method", "instance", "size"])[["robot_work", "robot_util_pct"]] \
        .mean().reset_index()
    base = inst[inst.method == ref].set_index("instance")
    inst = inst[inst.instance.isin(base.index)].copy()
    inst["dx"] = 100 * np.log(inst.robot_work / inst.instance.map(base.robot_work))
    inst["dy"] = 100 * np.log(inst.robot_util_pct / inst.instance.map(base.robot_util_pct))
    lim_x = (min(inst.dx.min(), 0) - 1, max(inst.dx.max(), 0) + 1)
    lim_y = (min(inst.dy.min(), 0) - 1, max(inst.dy.max(), 0) + 1)
    fig, axes = plt.subplots(1, len(sizes), figsize=(3.9 * len(sizes), 3.8),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    span = max(lim_x[1] - lim_x[0], lim_y[1] - lim_y[0])
    step = 5 if span > 12 else 2
    for ax, size in zip(axes, sizes):
        xs = np.array([lim_x[0] - span, lim_x[1] + span])
        for c in np.arange(-60, 61, step):  # iso-makespan lines: dy = dx - c
            ax.plot(xs, xs - c, color=style.GRID_STRONG if c else style.MUTED,
                    lw=0.6 if c else 0.9, zorder=1)
            yl = lim_x[1] - c
            if c and lim_y[0] < yl < lim_y[1]:
                ax.annotate(f"{c:+d} %", (lim_x[1], yl), xytext=(-2, 1),
                            textcoords="offset points", ha="right", va="bottom",
                            fontsize=6, color=style.MUTED, rotation=38)
        g = inst[inst["size"] == size]
        heur_pts = []
        for m in methods:
            gm, lk = g[g.method == m], look[m]
            if not len(gm):
                continue
            ax.scatter(gm.dx, gm.dy, s=10, color=lk["color"], alpha=0.35, lw=0, zorder=2)
            mx, my = gm.dx.mean(), gm.dy.mean()
            if lk["family"] == "heuristic":
                ax.scatter(mx, my, marker="*", s=90, color=style.HEURISTIC_MARK,
                           edgecolor="white", lw=0.6, zorder=4)
                heur_pts.append((mx, my, m))
            else:
                ax.scatter(mx, my, s=46, color=lk["color"], edgecolor="white", lw=1,
                           zorder=4, label=lk["label"] if ax is axes[0] else None)
        if heur_pts:  # labels spread vertically, on the roomier side of each star
            ys = _spread(np.array([p[1] for p in heur_pts]), 0.055 * (lim_y[1] - lim_y[0]))
            mid = (lim_x[0] + lim_x[1]) / 2
            for (mx, my, m), ly in zip(heur_pts, ys):
                left = mx > mid
                dx = -0.08 if left else 0.08
                ax.annotate(m, (mx, my), xytext=(mx + dx * (lim_x[1] - lim_x[0]), ly),
                            ha="right" if left else "left", va="center", fontsize=6.5,
                            color=style.MUTED, arrowprops=dict(arrowstyle="-", lw=0.5,
                                                               color=style.GRID_STRONG))
        ax.set_xlim(*lim_x)
        ax.set_ylim(*lim_y)
        ax.set_title(f"{size} ({g.instance.nunique()} instances)")
        ax.set_xlabel(f"Robot work vs. {ref} [%, log]")
    axes[0].set_ylabel(f"Robot utilisation vs. {ref} [%, log]")
    axes[0].annotate("diagonals: makespan change vs. reference\n(lower right = shorter)",
                     (0.02, 0.98), xycoords="axes fraction", va="top", fontsize=6.5,
                     color=style.MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles.append(Line2D([], [], marker="*", ls="none", ms=10, color=style.HEURISTIC_MARK,
                          mec="white"))
    labels.append("Heuristic (labelled)")
    handles.append(Line2D([], [], marker="o", ls="none", ms=4, color=style.MUTED, alpha=0.4))
    labels.append("one instance (mean over runs)")
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)),
               bbox_to_anchor=(0.5, -0.1 if len(labels) <= 4 else -0.2))
    if title:
        fig.suptitle(title, y=1.02, fontsize=10)
    style.save(fig, out, name)


def lns_fetches_over_time(iters, lns, look, sizes, start_fetches, refs, time_limit, out,
                          name, title=None, t_min=1.0):
    """Bin fetches of the LNS best solution over time, relative to the start solution
    (mean over runs and instances), with the final fetch counts of `refs` as dashed
    reference lines. refs: DataFrame method, size, value (relative change in %)."""
    grid = np.geomspace(t_min, time_limit, 300)
    fig, axes = plt.subplots(1, len(sizes), figsize=(4.6 * len(sizes), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, size in zip(axes, sizes):
        it = iters[iters["size"] == size]
        for m in lns:
            curves = []
            for (inst, run), g in it[it.method == m].groupby(["instance", "run_id"]):
                g = g.sort_values("t_end")
                f0 = start_fetches.get(inst, np.nan)
                t = np.concatenate([[0.0], g.t_end.values])
                v = np.concatenate([[f0], g.best_bin_fetches.values])
                idx = np.searchsorted(t, grid, side="right") - 1
                curves.append(100 * (v[idx] / f0 - 1))
            if curves:
                ax.plot(grid, np.nanmean(curves, axis=0), color=look[m]["color"], lw=1.6,
                        label=look[m]["label"] if ax is axes[0] else None, zorder=3)
        r = refs[refs["size"] == size]
        labels = []
        for row in r.itertuples():
            ax.axhline(row.value, color=style.HEURISTIC_COLOR if row.family == "heuristic"
                       else style.CP_COLOR, lw=0.9, ls=(0, (4, 2)), zorder=2)
            labels.append((row.value, row.method))
        ax.axhline(0, color=style.MUTED, lw=0.9)
        _time_axis(ax, t_min, time_limit)
        ax.set_title(f"{size} ({it.instance.nunique()} instances)")
        ax.set_xlabel("Time [s] (log scale)")
        ax._ref_labels = labels
    axes[0].set_ylabel("Bin fetches vs. start solution [%]")
    lo, hi = axes[0].get_ylim()
    for ax in axes:
        if ax._ref_labels:
            ys = _spread([y for y, _ in ax._ref_labels], 0.06 * (hi - lo))
            for y, (_, lab) in zip(ys, ax._ref_labels):
                ax.annotate(lab, (time_limit, y), xytext=(4, 0), textcoords="offset points",
                            va="center", fontsize=6.5, color=style.MUTED,
                            annotation_clip=False)
    axes[0].annotate("0 = LNS start solution", (t_min, 0), xytext=(3, 3),
                     textcoords="offset points", fontsize=6.5, color=style.MUTED)
    fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.2))
    fig.subplots_adjust(wspace=0.35)
    if title:
        fig.suptitle(title, y=1.02, fontsize=10)
    style.save(fig, out, name)
