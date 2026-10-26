"""Aggregate experiment1: LNS+CP vs. plain CP vs. heuristic on three instance sizes.

Usage:  python analyze_experiment1.py [experiment_dir] [--out OUT_DIR]

Reads, per instance directory (GEN{seed}-{stations}-{lanes}-{skus}-{orders}-{movecap}):
  heuristic_summary.txt                  heuristic makespan
  cplog_{k}.txt                          CP Optimizer log (incumbents with timestamps)
  experiments/*/experiment_dataframe.pkl LNS per-iteration records (5 runs)
  experiments/*/run_1.log                total variable count + strategy names

Writes tables (CSV, Markdown, LaTeX) and figures (PDF, PNG) into OUT_DIR.

Conventions
  improvement over heuristic  = (H - X) / H * 100        (positive = better than heuristic)
  gap of LNS to CP            = (LNS - CP) / CP * 100     (negative = LNS better than CP)
  Each LNS run is compared with the mean of the 5 CP runs on the same instance;
  mean +- std is then taken over all 25 LNS runs of a size class.
  Time is wall-clock seconds within the 900 s budget. The CP clock includes model
  extraction; the LNS clock starts after the initial model is built (see README).
"""

import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SIZE_BY_STATIONS = {5: "Small", 10: "Medium", 15: "Large"}
SIZES = ["Small", "Medium", "Large"]
TIME_POINTS = [60, 300, 900]
TIME_LIMIT = 900

# Validated categorical palette (dataviz reference, light mode), fixed order.
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_MAGENTA, C_GREEN = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
C_GRAY = "#8a8985"
METHOD_COLOR = {"LNS": C_BLUE, "CP": C_ORANGE}
SIZE_COLOR = {"Small": C_BLUE, "Medium": C_ORANGE, "Large": C_AQUA}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2,
    "savefig.bbox": "tight", "savefig.dpi": 200,
})


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def parse_cp_log(path):
    """Return (incumbents [(time, obj)], extraction_time, final_obj)."""
    incumbents, extraction, final = [], np.nan, np.nan
    with open(path) as f:
        for line in f:
            m = re.match(r"\s*\*\s+(\d+)\s+\S+\s+([\d.]+)s", line)
            if m:
                incumbents.append((float(m.group(2)), float(m.group(1))))
                continue
            m = re.search(r"Initial process time.*\(([\d.]+)s extraction", line)
            if m:
                extraction = float(m.group(1))
            m = re.search(r"Best objective\s*:\s*(\d+)", line)
            if m:
                final = float(m.group(1))
    if incumbents and not np.isnan(final):
        assert min(o for _, o in incumbents) == final, path
    return incumbents, extraction, final


def lns_trajectory(run_df, heuristic):
    """Incumbent trajectory [(t, makespan)] of one LNS run, starting at the heuristic."""
    traj = [(0.0, heuristic)]
    best = heuristic
    t_start = 0.0
    for _, r in run_df.sort_values("iteration").iterrows():
        offset = t_start + (r.total_iter_time - r.solve_time)  # time before solve starts
        for s in r.discovered_solutions or []:
            ms = s["kpis"]["makespan"]
            if ms < best:
                best = ms
                traj.append((offset + s["time"], ms))
        if r.best < best:  # safety net: improvement without a logged discovery
            best = r.best
            traj.append((t_start + r.total_iter_time, best))
        t_start += r.total_iter_time
    assert best == run_df.best.min() or best == heuristic, "trajectory mismatch"
    return traj


def value_at(traj, t):
    """Best objective known at time t (NaN if no solution yet)."""
    v = np.nan
    for tt, o in traj:
        if tt <= t:
            v = o if np.isnan(v) else min(v, o)
        else:
            break
    return v


def load(exp_dir):
    instances, lns_runs, cp_runs, iters = [], [], [], []
    for d in sorted(glob.glob(os.path.join(exp_dir, "GEN*"))):
        name = os.path.basename(d)
        m = re.match(r"GEN(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)", name)
        seed, stations, _, _, orders, _ = map(int, m.groups())
        size = SIZE_BY_STATIONS[stations]
        with open(os.path.join(d, "heuristic_summary.txt")) as f:
            heur = float(re.search(r"Makespan:\s*(\d+)", f.read()).group(1))
        instances.append(dict(instance=name, size=size, gen_seed=seed,
                              stations=stations, orders=orders, heuristic=heur))

        for log in sorted(glob.glob(os.path.join(d, "cplog_*.txt"))):
            k = int(re.search(r"cplog_(\d+)", log).group(1))
            inc, extr, final = parse_cp_log(log)
            cp_runs.append(dict(instance=name, size=size, run=k, traj=inc,
                                extraction=extr, final=final,
                                first_sol_time=inc[0][0] if inc else np.nan))

        exp = sorted(glob.glob(os.path.join(d, "experiments", "*")))[-1]
        df = pd.read_pickle(os.path.join(exp, "experiment_dataframe.pkl"))
        with open(os.path.join(exp, "run_1.log")) as f:
            log = f.read()
        total_vars = int(re.search(r"Optimized \d+/(\d+)", log).group(1))
        strat_names = dict((int(i), n) for n, i in
                           re.findall(r"Strategy: (\w+)\s*\((\d+)\)", log))
        df = df.copy()
        df["instance"], df["size"] = name, size
        df["strategy_name"] = df.strategy.map(strat_names)
        df["unfrozen_pct"] = 100 * df.num_optimized / total_vars
        for rid, rdf in df.groupby("run_id"):
            rdf = rdf.sort_values("iteration")
            prev_best = np.concatenate([[heur], rdf.best.values[:-1]])
            prev_best = np.minimum.accumulate(prev_best)
            df.loc[rdf.index, "prev_best"] = prev_best
            traj = lns_trajectory(rdf, heur)
            lns_runs.append(dict(instance=name, size=size, run=int(rid), traj=traj,
                                 final=traj[-1][1], n_iters=len(rdf)))
        iters.append(df)

    iters = pd.concat(iters, ignore_index=True)
    iters["improving"] = iters.best < iters.prev_best
    iters["delta"] = (iters.prev_best - iters.best).clip(lower=0)
    iters["delta_pct"] = 100 * iters.delta / iters.prev_best
    return (pd.DataFrame(instances), pd.DataFrame(lns_runs),
            pd.DataFrame(cp_runs), iters)


# --------------------------------------------------------------------------- #
# Table helpers
# --------------------------------------------------------------------------- #
def ms(series, fmt="{:.2f}"):
    s = pd.Series(series).dropna()
    if len(s) == 0:
        return "n/a"
    if len(s) == 1:
        return fmt.format(s.iloc[0])
    return f"{fmt.format(s.mean())} ± {fmt.format(s.std(ddof=1))}"


def write_table(df, out, name, caption):
    df.to_csv(os.path.join(out, f"{name}.csv"), index=False)
    with open(os.path.join(out, f"{name}.md"), "w") as f:
        md = ["| " + " | ".join(map(str, df.columns)) + " |",
              "|" + "---|" * df.shape[1]]
        md += ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
        f.write(f"**{caption}**\n\n" + "\n".join(md) + "\n")
    cols = "l" * 1 + "r" * (df.shape[1] - 1)
    esc = lambda x: str(x).replace("%", r"\%").replace("±", r"$\pm$").replace("_", r"\_")
    lines = [r"\begin{table}[ht]", r"\centering", rf"\caption{{{esc(caption)}}}",
             rf"\label{{tab:{name}}}", rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
             " & ".join(esc(c) for c in df.columns) + r" \\", r"\midrule"]
    lines += [" & ".join(esc(v) for v in row) + r" \\" for row in df.itertuples(index=False)]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(out, f"{name}.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def run_level(inst, lns, cp):
    """One row per LNS run with values at each time point and comparisons."""
    H = inst.set_index("instance").heuristic
    rows = []
    for _, r in lns.iterrows():
        cps = cp[cp.instance == r.instance]
        row = dict(instance=r.instance, size=r["size"], run=r.run, heuristic=H[r.instance])
        for t in TIME_POINTS:
            lv = value_at(r.traj, t)
            cv = np.array([value_at(c, t) for c in cps.traj])
            cp_mean = np.nanmean(cv) if np.isfinite(cv).all() else np.nan
            row[f"lns_{t}"] = lv
            row[f"cp_mean_{t}"] = cp_mean
            row[f"lns_impr_{t}"] = 100 * (H[r.instance] - lv) / H[r.instance]
            row[f"gap_{t}"] = 100 * (lv - cp_mean) / cp_mean
        rows.append(row)
    runs = pd.DataFrame(rows)

    cp_rows = []
    for _, c in cp.iterrows():
        row = dict(instance=c.instance, size=c["size"], run=c.run)
        for t in TIME_POINTS:
            v = value_at(c.traj, t)
            row[f"cp_{t}"] = v
            row[f"cp_impr_{t}"] = 100 * (H[c.instance] - v) / H[c.instance]
        cp_rows.append(row)
    return runs, pd.DataFrame(cp_rows)


def table_per_instance(inst, lns, cp, runs, out):
    rows = []
    for _, i in inst.sort_values(["stations", "gen_seed"]).iterrows():
        l = lns[lns.instance == i.instance].final
        c = cp[cp.instance == i.instance].final
        rows.append({
            "Size": i["size"], "Instance seed": i.gen_seed, "Heuristic": f"{i.heuristic:.0f}",
            "CP": ms(c, "{:.1f}"), "CP best": f"{c.min():.0f}",
            "LNS": ms(l, "{:.1f}"), "LNS best": f"{l.min():.0f}",
            "LNS impr. vs heur. [%]": ms(runs[runs.instance == i.instance].lns_impr_900),
            "Gap LNS to CP [%]": ms(runs[runs.instance == i.instance].gap_900),
        })
    write_table(pd.DataFrame(rows), out, "table1_per_instance",
                "Final makespan after 900 s per instance (mean ± std over 5 solver seeds)")


def table_summary(runs, cp_runs, lns, cp, out):
    rows = []
    for size in SIZES:
        r = runs[runs["size"] == size]
        c = cp_runs[cp_runs["size"] == size]
        for t in TIME_POINTS:
            cp_has = c[f"cp_{t}"].notna()
            rows.append({
                "Size": size, "Time [s]": t,
                "LNS impr. vs heur. [%]": ms(r[f"lns_impr_{t}"]),
                "CP impr. vs heur. [%]": ms(c[f"cp_impr_{t}"]) if cp_has.all()
                else f"n/a ({(~cp_has).sum()}/{len(c)} CP runs w/o solution)",
                "Gap LNS to CP [%]": ms(r[f"gap_{t}"]),
                "LNS better / equal / worse": "{}/{}/{}".format(
                    (r[f"gap_{t}"] < 0).sum(), (r[f"gap_{t}"] == 0).sum(),
                    (r[f"gap_{t}"] > 0).sum()) if r[f"gap_{t}"].notna().all() else "n/a",
            })
    write_table(pd.DataFrame(rows), out, "table2_summary_by_size",
                "Improvement over the heuristic and gap of LNS to CP at 60/300/900 s "
                "(mean ± std over 25 runs per size; gap < 0 means LNS is better)")


def table_strategies(iters, out):
    rows = []
    for size in SIZES:
        it = iters[iters["size"] == size]
        tot_impr = it.delta.sum()
        for name, g in it.groupby("strategy_name"):
            imp = g[g.improving]
            rows.append({
                "Size": size, "Strategy": name, "Uses": len(g),
                "Improving iters": len(imp),
                "Success rate [%]": f"{100 * len(imp) / len(g):.1f}",
                "Solutions / improving iter": ms(imp.num_discovered_solutions, "{:.1f}"),
                "Improvement / improving iter [%]": ms(imp.delta_pct, "{:.3f}"),
                "Share of total improvement [%]":
                    f"{100 * imp.delta.sum() / tot_impr:.1f}" if tot_impr > 0 else "n/a",
                "Unfrozen vars [%]": ms(g.unfrozen_pct, "{:.1f}"),
            })
    write_table(pd.DataFrame(rows), out, "table3_strategies",
                "Neighbourhood strategy performance per size class, pooled over 25 LNS runs")

    rows = []
    for size in SIZES:
        it = iters[iters["size"] == size]
        per_run = it.groupby(["instance", "run_id"]).agg(
            n=("iteration", "size"), imp=("improving", "sum"))
        imp = it[it.improving]
        rows.append({
            "Size": size,
            "Iterations / run": ms(per_run.n, "{:.1f}"),
            "Improving iters / run": ms(per_run.imp, "{:.1f}"),
            "Improving iters [%]": f"{100 * it.improving.mean():.1f}",
            "Solutions / improving iter": ms(imp.num_discovered_solutions, "{:.2f}"),
            "Improvement / improving iter [%]": ms(imp.delta_pct, "{:.3f}"),
            "Improvement / improving iter [abs]": ms(imp.delta, "{:.1f}"),
        })
    write_table(pd.DataFrame(rows), out, "table4_iteration_stats",
                "LNS iteration statistics per size class (mean ± std)")


TIME_PARTS = [  # (column, label, color)
    ("lns_strategy_time", "Neighbourhood selection", C_MAGENTA),
    ("constr_generate_time", "Freeze constraint generation", C_YELLOW),
    ("constr_update_time", "Model update", C_GREEN),
    ("solve_api_overhead", "Solver API overhead", C_GRAY),
    ("solve_cpp_presolve", "CP extraction / presolve", C_ORANGE),
    ("solve_cpp_search", "CP search", C_BLUE),
]


def table_time(iters, cp, out):
    it = iters.copy()
    it["other"] = it.total_iter_time - it[[c for c, _, _ in TIME_PARTS]].sum(axis=1)
    rows = []
    for size in SIZES:
        g = it[it["size"] == size]
        row = {"Size": size, "Iteration total [s]": ms(g.total_iter_time)}
        for c, label, _ in TIME_PARTS:
            row[f"{label} [s]"] = ms(g[c], "{:.3f}")
        row["Search share [%]"] = f"{100 * g.solve_cpp_search.sum() / g.total_iter_time.sum():.1f}"
        c = cp[cp["size"] == size]
        row["Full-model CP extraction [s]"] = ms(c.extraction, "{:.1f}")
        row["Full-model CP first incumbent [s]"] = ms(c.first_sol_time, "{:.1f}")
        rows.append(row)
    df = pd.DataFrame(rows)
    write_table(df, out, "table5_time_composition",
                "Composition of one LNS iteration (mean ± std over all iterations) "
                "and full-model CP start-up cost")
    return it


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_convergence(inst, lns, cp, out):
    H = inst.set_index("instance").heuristic
    grid = np.linspace(0, TIME_LIMIT, 901)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, size in zip(axes, SIZES):
        for method, runs in (("LNS", lns), ("CP", cp)):
            r = runs[runs["size"] == size]
            curves = np.array([[100 * (H[x.instance] - value_at(x.traj, t)) / H[x.instance]
                                for t in grid] for _, x in r.iterrows()])
            valid = np.isfinite(curves).all(axis=0)
            mean = np.where(valid, np.nanmean(np.where(np.isfinite(curves), curves, 0), 0), np.nan)
            std = np.where(valid, np.nanstd(np.where(np.isfinite(curves), curves, 0), 0, ddof=1), np.nan)
            col = METHOD_COLOR[method]
            ax.plot(grid, mean, color=col, label=method)
            ax.fill_between(grid, mean - std, mean + std, color=col, alpha=0.15, lw=0)
        for t in TIME_POINTS[:-1]:
            ax.axvline(t, color="#c3c2b7", lw=1, ls="--", zorder=0)
        ax.axhline(0, color="#52514e", lw=0.8)
        ax.set_title(f"{size} ({inst[inst['size'] == size].stations.iloc[0]} stations, "
                     f"{inst[inst['size'] == size].orders.iloc[0]} orders)")
        ax.set_xlabel("Time [s]")
        ax.set_xlim(0, TIME_LIMIT)
    axes[0].set_ylabel("Improvement over heuristic [%]")
    axes[0].legend(loc="lower right")
    fig.suptitle("Anytime improvement over the heuristic (mean ± std, 25 runs per method)",
                 y=1.03, fontsize=10)
    save(fig, out, "fig1_convergence")


def fig_final_distributions(runs, cp_runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    rng = np.random.default_rng(0)
    # (a) improvement over heuristic, LNS vs CP
    ax = axes[0]
    for i, size in enumerate(SIZES):
        for j, (method, vals) in enumerate(
                (("LNS", runs[runs["size"] == size].lns_impr_900),
                 ("CP", cp_runs[cp_runs["size"] == size].cp_impr_900))):
            x = i + (j - 0.5) * 0.36
            ax.boxplot(vals, positions=[x], widths=0.28, showfliers=False,
                       medianprops=dict(color="#0b0b0b"), boxprops=dict(color=C_GRAY),
                       whiskerprops=dict(color=C_GRAY), capprops=dict(color=C_GRAY))
            ax.scatter(x + rng.uniform(-0.08, 0.08, len(vals)), vals, s=14,
                       color=METHOD_COLOR[method], edgecolor="white", lw=0.5, zorder=3,
                       label=method if i == 0 else None)
    ax.set_xticks(range(3), SIZES)
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.set_ylabel("Improvement over heuristic [%]")
    ax.set_title("(a) Final improvement over heuristic (900 s)")
    ax.legend(loc="upper right")
    # (b) gap LNS to CP
    ax = axes[1]
    for i, size in enumerate(SIZES):
        vals = runs[runs["size"] == size].gap_900
        ax.boxplot(vals, positions=[i], widths=0.4, showfliers=False,
                   medianprops=dict(color="#0b0b0b"), boxprops=dict(color=C_GRAY),
                   whiskerprops=dict(color=C_GRAY), capprops=dict(color=C_GRAY))
        ax.scatter(i + rng.uniform(-0.1, 0.1, len(vals)), vals, s=14, color=C_BLUE,
                   edgecolor="white", lw=0.5, zorder=3)
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.set_xticks(range(3), SIZES)
    ax.set_ylabel("Gap LNS to CP [%]  (< 0: LNS better)")
    ax.set_title("(b) Final gap of LNS to mean CP (900 s)")
    save(fig, out, "fig2_final_distributions")


def fig_gap_over_time(runs, out):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    w = 0.25
    for i, size in enumerate(SIZES):
        r = runs[runs["size"] == size]
        for j, t in enumerate(TIME_POINTS):
            v = r[f"gap_{t}"]
            x = j + (i - 1) * w
            if v.isna().any():
                ax.text(x, 0.3, "no CP\nsolution", ha="center", va="bottom",
                        fontsize=7, color="#52514e")
                continue
            if abs(v.mean()) < 0.05:
                ax.text(x, 0.15, f"{v.mean():.2f}", ha="center", va="bottom", fontsize=7,
                        color="#52514e")
            ax.bar(x, v.mean(), w * 0.9, color=SIZE_COLOR[size],
                   label=size if j == len(TIME_POINTS) - 1 else None)
            ax.errorbar(x, v.mean(), v.std(ddof=1), color="#0b0b0b", lw=1, capsize=2)
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.set_xticks(range(len(TIME_POINTS)), [f"{t} s" for t in TIME_POINTS])
    ax.set_ylabel("Gap LNS to CP [%]  (< 0: LNS better)")
    ax.set_title("Gap of LNS to CP at different time points (mean ± std)")
    ax.legend(loc="lower left")
    save(fig, out, "fig3_gap_over_time")


def fig_strategies(iters, out):
    names = sorted(iters.strategy_name.unique())
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), gridspec_kw=dict(wspace=0.08))
    metrics = [
        ("Success rate [%]", lambda g: 100 * g.improving.mean(), None),
        ("Solutions per improving iteration", lambda g: g[g.improving].num_discovered_solutions.mean(),
         lambda g: g[g.improving].num_discovered_solutions.std(ddof=1)),
        ("Improvement per improving iteration [%]", lambda g: g[g.improving].delta_pct.mean(),
         lambda g: g[g.improving].delta_pct.std(ddof=1)),
    ]
    h = 0.4
    y = np.arange(len(names))
    sizes = [s for s in SIZES if iters[iters["size"] == s].improving.any()]
    for ax, (title, f, fe) in zip(axes, metrics):
        for i, size in enumerate(sizes):
            it = iters[iters["size"] == size]
            vals = [f(it[it.strategy_name == n]) for n in names]
            errs = [fe(it[it.strategy_name == n]) for n in names] if fe else None
            vals = np.nan_to_num(np.array(vals, float))
            ax.barh(y + (i - 0.5) * h, vals, h * 0.9, color=SIZE_COLOR[size], label=size,
                    xerr=np.nan_to_num(np.array(errs, float)) if errs else None,
                    error_kw=dict(lw=0.8, capsize=1.5, ecolor="#52514e"))
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [n.replace("_", " ") for n in names])
    for ax in axes:
        ax.invert_yaxis()
    for ax in axes[1:]:
        ax.set_yticks(y, [""] * len(y))
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.6, -0.08), ncol=len(sizes))
    fig.suptitle("Neighbourhood strategy performance (pooled over 25 runs per size; "
                 "Large omitted: no improving iteration)", y=1.0, fontsize=10)
    save(fig, out, "fig4_strategies")


def fig_time_composition(it, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.6), sharey=True)
    y = np.arange(len(SIZES))
    for ax, relative in zip(axes, (False, True)):
        left = np.zeros(len(SIZES))
        for c, label, col in TIME_PARTS + [("other", "Other / logging", "#d8d7d2")]:
            vals = np.array([it[it["size"] == s][c].mean() for s in SIZES])
            if relative:
                vals = 100 * vals / np.array([it[it["size"] == s].total_iter_time.mean()
                                              for s in SIZES])
            ax.barh(y, vals, 0.6, left=left, color=col, label=label,
                    edgecolor="white", lw=1)
            left += vals
        ax.set_xlabel("Share of iteration time [%]" if relative else "Mean time per iteration [s]")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, SIZES)
    axes[0].invert_yaxis()
    axes[0].set_title("(a) Absolute")
    axes[1].set_title("(b) Relative")
    axes[1].set_xlim(0, 100)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.2))
    fig.suptitle("Time composition of one LNS iteration (iteration limit 10 s)",
                 y=1.04, fontsize=10)
    save(fig, out, "fig5_time_composition")


def fig_improvement_timeline(iters, out):
    """Where in the run improvements happen: improving iterations over time."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
    for ax, size in zip(axes, SIZES[:2]):
        it = iters[iters["size"] == size].sort_values(["instance", "run_id", "iteration"]).copy()
        it["t_end"] = it.groupby(["instance", "run_id"]).total_iter_time.cumsum()
        bins = np.arange(0, TIME_LIMIT + 60, 60)
        it["bin"] = pd.cut(it.t_end, bins)
        rate = it.groupby("bin", observed=False).improving.mean() * 100
        ax.bar(bins[:-1] + 30, rate.values, 54, color=SIZE_COLOR[size])
        ax.set_title(f"{size}")
        ax.set_xlabel("Time [s]")
    axes[0].set_ylabel("Improving iterations [%]")
    fig.suptitle("Share of improving LNS iterations per 60 s window", y=1.03, fontsize=10)
    save(fig, out, "fig6_improvement_timeline")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dir", nargs="?", default="results/experiment1")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.exp_dir, "analysis")
    os.makedirs(out, exist_ok=True)

    inst, lns, cp, iters = load(a.exp_dir)
    runs, cp_runs = run_level(inst, lns, cp)
    runs.to_csv(os.path.join(out, "data_lns_runs.csv"), index=False)
    cp_runs.to_csv(os.path.join(out, "data_cp_runs.csv"), index=False)
    iters.drop(columns=["unfrozen_orders", "unfrozen_skus", "discovered_solutions"]).to_csv(
        os.path.join(out, "data_lns_iterations.csv"), index=False)

    table_per_instance(inst, lns, cp, runs, out)
    table_summary(runs, cp_runs, lns, cp, out)
    table_strategies(iters, out)
    it = table_time(iters, cp, out)

    fig_convergence(inst, lns, cp, out)
    fig_final_distributions(runs, cp_runs, out)
    fig_gap_over_time(runs, out)
    fig_strategies(iters, out)
    fig_time_composition(it, out)
    fig_improvement_timeline(iters, out)
    print(f"Wrote tables and figures to {out}")


if __name__ == "__main__":
    main()
