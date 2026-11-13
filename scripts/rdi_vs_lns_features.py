"""Which instance properties go with RDI-LB share beating LNS? (companion to rdi_vs_lns_large.py)

Usage: python scripts/rdi_vs_lns_features.py [root] [--lns LNS-6] [--out DIR]

Target per instance:  adv = (mean LNS final - RDI-LB share) / BKS * 100   (> 0: RDI better)

Explanatory variables:
  * Instance.get_features() (normalised features) and a few raw statistics
  * workload bounds:  picker LB = sum of pick times / stations,
                      robot LB  = sum over fetches of (rt + rt_ret) / MoveCap,
    evaluated with one fetch per order line (no sharing) and with one fetch per used SKU
  * solution structure: bin fetches of the LNS start (LB-SGC), of RDI-LB share and of the
    final LNS solutions (dataframe KPIs), and picks per bin

With 5 instances per size, correlations are descriptive only.
"""

import argparse
import glob
import json
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluation.data import finished_experiments  # noqa: E402


def bin_events(path):
    with open(path) as f:
        m = re.search(r"BinEvents:\s*(\d+)", f.read())
    return int(m.group(1)) if m else np.nan


def lns_final_kpis(d, cfg_match):
    """Final best-solution KPIs per run of the experiment whose config matches."""
    for e in finished_experiments(d):
        cfg = json.load(open(os.path.join(e, "experiment_config.json")))
        if all(cfg.get(k) == v for k, v in cfg_match.items()):
            df = pd.read_pickle(os.path.join(e, "experiment_dataframe.pkl"))
            last = df.sort_values("iteration").groupby("run_id").tail(1)
            return last[["best_makespan", "best_bin_fetches", "best_total_flow_time",
                         "best_idle_time", "best_max_picks"]]
    return None


def solution_stats(sol, inst):
    """Robot and picker utilisation of a HeuristicSolution."""
    events = [e for evs in sol.bin_events.values() for e in evs]
    robot = sum(inst.rt[e.sku] + inst.rt_ret[e.sku] for e in events)
    picks = sum(len(e.pick_intervals) for e in events)
    pick_time = sum(b - a for e in events for a, b in e.pick_intervals)
    per_station = [sum(b - a for e in evs for a, b in e.pick_intervals)
                   for evs in sol.bin_events.values()]
    return dict(fetches=len(events), picks_per_bin=picks / len(events),
                robot_util=robot / (inst.movecap * sol.makespan),
                picker_util=pick_time / (len(inst.S) * sol.makespan),
                station_pick_cv=np.std(per_station) / np.mean(per_station))


def instance_row(d, inst):
    lines = [(o, k) for o, ks in inst.orders_requirements.items() for k in ks]
    rt2 = {k: inst.rt[k] + inst.rt_ret[k] for k in inst.unique_skus}
    pick_total = sum(inst.p[k] for _, k in lines)
    robot_lines = sum(rt2[k] for _, k in lines)
    robot_skus = sum(rt2.values())
    st = inst.stats
    return {
        "lines": len(lines), "used_skus": st["used_skus"],
        "lines_per_sku": len(lines) / st["used_skus"],
        "picker_LB": pick_total / len(inst.S),
        "robot_LB_noshare": robot_lines / inst.movecap,
        "robot_LB_oncepersku": robot_skus / inst.movecap,
        "robot_to_picker_noshare": (robot_lines / inst.movecap) / (pick_total / len(inst.S)),
        "max_sharing_saving_pct": 100 * (1 - robot_skus / robot_lines),
        "order_size_mean": st["order_size_mean"], "order_size_cv":
            st["order_size_stdev"] / st["order_size_mean"],
        "rt_mean": st["rt_mean"], "p_mean": st["p_mean"],
        "sku_freq_max": st["sku_freq_max"], "pareto_20": st["pareto_ratios"][20],
        "jaccard_w": st["avg_jaccard_weighted"],
        **{k.replace("feat_", "f_"): v for k, v in inst.features.items()},
    }


def figure(df, out):
    import matplotlib.pyplot as plt
    from evaluation import style
    style.apply()
    sizes = ["Small", "Medium", "Large"]
    colors = dict(zip(sizes, style.CATEGORICAL[:3]))
    markers = dict(zip(sizes, ["o", "s", "^"]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw=dict(width_ratios=[1, 1.1]))
    ax = axes[0]
    for sz in sizes:
        g = df[df["size"] == sz]
        ax.scatter(g.robot_to_picker_noshare, g.adv_rdi_pct, s=40, color=colors[sz],
                   marker=markers[sz], edgecolor="white", lw=1, label=sz, zorder=3)
    ax.axhline(0, color=style.MUTED, lw=0.8)
    ax.set_xlabel("Robot / picker workload (no sharing)")
    ax.set_ylabel("RDI-LB share advantage over LNS-6 [% BKS]")
    ax.set_title("(a) More robot-bound: RDI-LB share catches up", loc="left")
    ax.legend(loc="lower right")
    ax = axes[1]
    g = df[df["size"] == "Large"].sort_values("adv_rdi_pct")
    y = np.arange(len(g))
    ax.barh(y + 0.2, g.work_saving_rdi_pct, 0.38, color=style.CATEGORICAL[2],
            label="less robot work (fewer fetches)")
    ax.barh(y - 0.2, -g.util_deficit_rdi_pct, 0.38, color=style.CATEGORICAL[1],
            label="lower robot utilisation")
    ax.scatter(g.adv_log_pct, y, marker="D", s=30, color=style.INK, zorder=3,
               label="net advantage")
    for yi, v in zip(y, g.adv_log_pct):
        ax.annotate(f"{v:+.2f} %", (1.0, yi), xycoords=("axes fraction", "data"),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=7,
                    color=style.INK, fontweight="bold")
    ax.axvline(0, color=style.MUTED, lw=0.8)
    ax.set_yticks(y, [n.split("-")[0] for n in g.instance])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Contribution to ln(LNS-6 / RDI-LB share) [%]")
    ax.set_title("(b) Large: why RDI-LB share wins or loses", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=7)
    style.save(fig, out, "fig_rdi_vs_lns_mechanism")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="precalculated_instances")
    ap.add_argument("--runs", default="results/lns_config_comparison/data/runs.csv")
    ap.add_argument("--lns", default="LNS-6")
    ap.add_argument("--out", default="results/lns_config_comparison/rdi_vs_lns")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    runs = pd.read_csv(a.runs)
    lns_match = {"only_first_sol": False, "combine_strat_with_tail": "none",
                 "cp_objective_phase_schedule": "mmmf"}  # LNS-6
    rows = []
    for d in sorted(glob.glob(os.path.join(a.root, "GEN*"))):
        name = os.path.basename(d)
        r = runs[runs.instance == name]
        bks = r.bks.iloc[0]
        rdi = r[r.method == "RDI-LB share"].final.iloc[0]
        lns = r[r.method == a.lns].final
        inst = pickle.load(open(os.path.join(d, "instance.pkl"), "rb"))
        sol = pickle.load(open(os.path.join(d, "heuristic_rdi_lb_share_solution.pkl"), "rb"))
        start = pickle.load(open(os.path.join(d, "heuristic_solution.pkl"), "rb"))
        kp = lns_final_kpis(d, lns_match)
        rs, ss = solution_stats(sol, inst), solution_stats(start, inst)
        row = {"instance": name, "size": r["size"].iloc[0], "bks": bks,
               "rdi_lb_share": rdi, "lns_mean": lns.mean(), "lns_sd": lns.std(ddof=1),
               "adv_rdi_pct": 100 * (lns.mean() - rdi) / bks,
               "start": r[r.method == "LB-SGC"].final.iloc[0],
               "rdi_lb": r[r.method == "RDI-LB"].final.iloc[0],
               "share_gain_pct": 100 * (r[r.method == "RDI-LB"].final.iloc[0] - rdi) / bks,
               "fetches_start": ss["fetches"], "fetches_rdi": rs["fetches"],
               "fetches_rdi_lb": bin_events(os.path.join(d, "heuristic_rdi_lb_summary.txt")),
               "fetches_lns_final": kp.best_bin_fetches.mean() if kp is not None else np.nan,
               "picks_per_bin_start": ss["picks_per_bin"], "picks_per_bin_rdi": rs["picks_per_bin"],
               "robot_util_rdi": rs["robot_util"], "picker_util_rdi": rs["picker_util"],
               "robot_util_start": ss["robot_util"], "picker_util_start": ss["picker_util"],
               "station_pick_cv_rdi": rs["station_pick_cv"],
               "station_pick_cv_start": ss["station_pick_cv"],
               **instance_row(d, inst)}
        row["fetch_saving_rdi_vs_lns_pct"] = 100 * (row["fetches_lns_final"] - row["fetches_rdi"]) \
            / row["fetches_lns_final"]
        row["rdi_over_picker_LB"] = rdi / row["picker_LB"]
        row["rdi_over_robot_LB_rdi_fetches"] = rdi / (row["robot_util_rdi"] * rdi) \
            if row["robot_util_rdi"] else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    # Robot-bound decomposition: makespan = robot work / (MoveCap * robot utilisation).
    # LNS robot work is approximated by the start's, scaled by the LNS final fetch count.
    w_start = df.robot_util_start * df.start          # robot work / MoveCap
    w_rdi = df.robot_util_rdi * df.rdi_lb_share
    w_lns = w_start * df.fetches_lns_final / df.fetches_start
    df["robot_util_lns_implied"] = w_lns / df.lns_mean
    df["work_saving_rdi_pct"] = 100 * np.log(w_lns / w_rdi)      # RDI needs less robot work
    df["util_deficit_rdi_pct"] = 100 * np.log(df.robot_util_lns_implied / df.robot_util_rdi)
    df["adv_log_pct"] = df.work_saving_rdi_pct - df.util_deficit_rdi_pct  # = 100 ln(lns/rdi)
    df.to_csv(os.path.join(a.out, "instance_diagnostics.csv"), index=False)
    figure(df, a.out)

    pd.set_option("display.width", 250, "display.max_columns", 40)
    show = ["instance", "adv_rdi_pct", "rdi_lb_share", "lns_mean", "start", "share_gain_pct",
            "fetches_start", "fetches_lns_final", "fetches_rdi", "fetch_saving_rdi_vs_lns_pct",
            "picks_per_bin_rdi", "robot_util_rdi", "picker_util_rdi",
            "robot_to_picker_noshare", "max_sharing_saving_pct", "lines_per_sku",
            "robot_util_lns_implied", "work_saving_rdi_pct", "util_deficit_rdi_pct"]
    for size, g in df.groupby("size", sort=False):
        print(f"\n=== {size}")
        print(g[show].round(3).to_string(index=False))

    # correlations of every numeric variable with the RDI advantage
    num = df.select_dtypes("number").drop(columns=["adv_rdi_pct"])
    num = num.loc[:, num.nunique() > 1]
    out = []
    for size, g in list(df.groupby("size", sort=False)) + [("All (within-size ranks)", df)]:
        for c in num.columns:
            x, y = g[c], g["adv_rdi_pct"]
            if size.startswith("All"):  # remove size effects: rank within size class
                x = g.groupby("size")[c].rank()
                y = g.groupby("size")["adv_rdi_pct"].rank()
            if x.nunique() < 2:
                continue
            rho, p = stats.spearmanr(x, y)
            out.append(dict(size=size, variable=c, spearman=rho, p=p, n=len(g)))
    corr = pd.DataFrame(out)
    corr.to_csv(os.path.join(a.out, "correlations.csv"), index=False)
    for size, g in corr.groupby("size", sort=False):
        top = g.reindex(g.spearman.abs().sort_values(ascending=False).index).head(12)
        print(f"\n--- strongest rank correlations with adv_rdi_pct: {size} "
              f"({len(g)} variables tested)")
        print(top.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
