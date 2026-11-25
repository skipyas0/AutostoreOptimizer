"""Validity check, step 4: does the static model rank schedules the way the simulator does?

    .venv/bin/python scripts/validity/analyze.py [--out results/validity]

Reads <out>/*/seed*/static.csv, plans/, calibration.json and <out>/sim.csv (simulate.py); writes Markdown
and CSV tables and figures to <out>/analysis/.

Static predictors of the simulated makespan, per schedule:
    C          static makespan
    W          static robot work (sum of rt + rt_ret over visits)
    fetches    number of bin visits
    visits_max most visits at one station
    C_port     port-aware bound: max over stations of (visits x tau + picking time), tau = port time per
               visit (lowering, lifting, changeover) measured in the instance's calibration run
    C_max      max(C, C_port)
Correlations are Spearman rank correlations across the schedules of one instance, then averaged over
instances (n = instances, 95% t-interval).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bridge  # noqa: E402

sys.path.insert(0, bridge.REPO)
from evaluation import tables  # noqa: E402
from evaluation.solutions import metrics, read_solution  # noqa: E402
from instance import Instance  # noqa: E402

PREDICTORS = ["C", "W", "fetches", "visits_max", "C_port", "C_max"]
FAMILY = {"sgc": "SGC", "lb_sgc": "SGC", "sgc_share": "SGC", "lb_sgc_share": "SGC", "ama_two_phase": "SGC",
          "lb_ama_two_phase_singles": "SGC", "cfss": "CFSS", "cfss_share": "CFSS", "rdi": "RDI", "rdi_lb_share": "RDI"}


def family(m: str) -> str:
    if m.startswith("lns"):
        return "LNS"
    if m.startswith("gbs"):
        return "GBS"
    return FAMILY.get(m, "other")


def write(df: pd.DataFrame, path: str, digits: int = 2) -> str:
    shown = df.copy()
    for c in shown.columns:
        if shown[c].dtype.kind == "f":
            shown[c] = shown[c].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
    df.to_csv(path + ".csv", index=False)
    text = tables.to_markdown(shown)
    open(path + ".md", "w").write(text)
    return text


def ci(x) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (x.mean() if len(x) else np.nan, np.nan, np.nan)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() - h, x.mean() + h


def lns_static(out: str) -> pd.DataFrame:
    rows = []
    for exp in glob.glob(os.path.join(out, "*", "lns", "precalculated_instances", "VAL-*", "experiments", "*")):
        root = os.path.dirname(os.path.dirname(exp))
        _, regime, seed = os.path.basename(root).split("-")
        inst = Instance.from_pickle(os.path.join(root, "instance.pkl"))
        cfg = json.load(open(os.path.join(root, "config.json")))
        for f in glob.glob(os.path.join(exp, "run*-solution.*")):
            run = os.path.basename(f).split("-")[0]
            rows.append(dict(regime=regime, seed=int(seed), method=f"lns_{cfg['heuristic']}_{run}",
                             **metrics(*read_solution(f), inst)))
    return pd.DataFrame(rows)


def plan_load(out: str) -> pd.DataFrame:
    """Per plan: visits and picking time of its busiest station, and the port-aware bound."""
    rows = []
    for d in glob.glob(os.path.join(out, "*", "seed*")):
        regime, seed = d.split(os.sep)[-2], int(d.split("seed")[-1])
        sim = json.load(open(os.path.join(d, "sim", f"instance-seed{seed}.json")))
        cal = json.load(open(os.path.join(d, "calibration.json")))
        cm = json.load(open(os.path.join(d, "cal", "metrics.json")))[0]
        tau = (cm["portBusy"] + cm["portChange"]) * cm["simTime"] * cm["stations"] / cm["ops"]["fetches"]
        for p in glob.glob(os.path.join(d, "plans", "*.json")):
            plan = json.load(open(p))
            visits = [len(sp["visits"]) for sp in plan["stations"]]
            pick = [sum(sim["pickTime"][v["sku"]] * len(v["orders"]) for v in sp["visits"]) for sp in plan["stations"]]
            rows.append(dict(regime=regime, seed=seed, method=os.path.basename(p)[:-5], tau=tau,
                             visits_max=max(visits), C_port=max(n * tau + t for n, t in zip(visits, pick)),
                             cal_makespan=cal["cal_makespan"]))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(bridge.REPO, "results", "validity"))
    args = ap.parse_args()
    dst = os.path.join(args.out, "analysis")
    os.makedirs(dst, exist_ok=True)

    st = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(args.out, "*", "seed*", "static.csv"))])
    lns = lns_static(args.out)
    st = pd.concat([st, lns], ignore_index=True) if len(lns) else st
    st = st.rename(columns={"makespan": "C", "robot_work": "W"})
    sim = pd.read_csv(os.path.join(args.out, "sim.csv"))
    load = plan_load(args.out)
    df = sim[sim["mode"] != "native"].merge(st, on=["regime", "seed", "method"]).merge(load, on=["regime", "seed", "method"])
    df["C_max"] = np.maximum(df.C, df.C_port)
    df["family"] = df.method.map(family)
    native = sim[sim["mode"] == "native"]
    not_done = df[~df.sim_done.astype(bool)]
    df = df[df.sim_done.astype(bool)]
    print(f"{len(df)} plan runs ({df.method.nunique()} methods, {df.groupby(['regime', 'seed']).ngroups} instances); "
          f"{len(not_done)} unfinished excluded\n")

    # 1. rank correlation of each static predictor with the simulated makespan
    rows = []
    for (regime, mode), g in df.groupby(["regime", "mode"]):
        row = {"regime": regime, "mode": mode, "instances": g.seed.nunique(), "schedules / inst.": g.groupby("seed").size().mean()}
        for p in PREDICTORS:
            rho = [stats.spearmanr(h[p], h.sim_makespan)[0] for _, h in g.groupby("seed")]
            m, lo, hi = ci(rho)
            row[f"ρ {p}"] = f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]"
        row["sim / static C"] = (g.sim_makespan / g.C).mean()
        row["sim / C_port"] = (g.sim_makespan / g.C_port).mean()
        rows.append(row)
    print("1. Spearman ρ (static predictor vs simulated makespan), mean over instances [95% CI]")
    print(write(pd.DataFrame(rows), os.path.join(dst, "t1_rank_correlation")), "\n")

    # 2. selection regret: choose the schedule a predictor ranks best; how far is it from the simulator's best?
    rows = []
    for (regime, mode), g in df.groupby(["regime", "mode"]):
        row = {"regime": regime, "mode": mode}
        for p in PREDICTORS:
            reg = []
            for _, h in g.groupby("seed"):
                pick = h.loc[h[p].idxmin()]
                reg.append(100 * (pick.sim_makespan / h.sim_makespan.min() - 1))
            row[f"{p} pick: regret %"] = np.mean(reg)
        rows.append(row)
    print("2. Selection regret: simulated makespan of the schedule each predictor ranks best, vs the best one [%]")
    print(write(pd.DataFrame(rows), os.path.join(dst, "t2_selection_regret")), "\n")

    # 3. per method (full_b3): static and simulated makespan relative to the instance's best
    g = df[df["mode"] == "full_b3"].copy()
    for c, best in (("C", "C_rel"), ("sim_makespan", "sim_rel"), ("C_port", "C_port_rel")):
        g[best] = 100 * (g[c] / g.groupby(["regime", "seed"])[c].transform("min") - 1)
    g["C_rank"] = g.groupby(["regime", "seed"]).C.rank()
    g["sim_rank"] = g.groupby(["regime", "seed"]).sim_makespan.rank()
    t3 = (g.groupby(["regime", "family", "method"])
          .agg(C=("C", "mean"), C_rel=("C_rel", "mean"), fetches=("fetches", "mean"), visits_max=("visits_max", "mean"),
               sim=("sim_makespan", "mean"), sim_rel=("sim_rel", "mean"), C_port_rel=("C_port_rel", "mean"),
               C_rank=("C_rank", "mean"), sim_rank=("sim_rank", "mean"),
               robot_idle=("sim_robot_idle_pct", "mean"), port_busy=("sim_port_busy_pct", "mean"))
          .reset_index().sort_values(["regime", "sim_rel"]))
    print("3. Per method (full plan, buffer 3): static C and simulated makespan, % above the instance's best")
    print(write(t3, os.path.join(dst, "t3_per_method"), digits=1), "\n")

    # 4. static plans vs the simulator's own heuristics
    rows = []
    for regime, g in df.groupby("regime"):
        nat = native[native.regime == regime].groupby("method").sim_makespan.mean()
        best_static = g[g["mode"] == "full_b3"].groupby("seed").sim_makespan.min().mean()
        for label, v in [("best static plan (full, buffer 3), per instance", best_static),
                         *[(f"{m} (full, buffer 3)", g[(g["mode"] == "full_b3") & (g.method == m)].sim_makespan.mean())
                           for m in ("gbs_regret_lc", "lb_sgc") + tuple(sorted(x for x in g.method.unique() if x.startswith("lns")))[:1]],
                         *[(f"simulator {m}", nat[m]) for m in nat.index]]:
            rows.append({"regime": regime, "execution": label, "sim makespan": v,
                         "vs simulator FCFS %": 100 * (v / nat["sim_fcfs_b3"] - 1)})
    print("4. Static plans vs the simulator's own heuristics (mean simulated makespan)")
    print(write(pd.DataFrame(rows), os.path.join(dst, "t4_vs_native"), digits=1), "\n")

    # 5. LNS: does its static gain carry over?
    if len(lns):
        rows = []
        for (regime, mode), g in df.groupby(["regime", "mode"]):
            for m in sorted(x for x in g.method.unique() if x.startswith("lns")):
                start = m.split("_", 1)[1].rsplit("_", 1)[0]
                a = g[g.method == m].set_index("seed")
                b = g[g.method == start].set_index("seed").loc[a.index]
                dc = 100 * (a.C / b.C - 1)
                ds = 100 * (a.sim_makespan / b.sim_makespan - 1)
                rows.append({"regime": regime, "mode": mode, "LNS": m, "n": len(a),
                             "static ΔC %": dc.mean(), "static Δfetches %": (100 * (a.fetches / b.fetches - 1)).mean(),
                             "sim Δmakespan %": ds.mean(), "sim better / worse": f"{(ds < 0).sum()} / {(ds > 0).sum()}",
                             "p (Wilcoxon, sim)": stats.wilcoxon(ds).pvalue if (ds != 0).any() and len(ds) > 1 else np.nan})
        print("5. LNS vs its warm start: static gain and simulated gain")
        print(write(pd.DataFrame(rows), os.path.join(dst, "t5_lns_transfer"), digits=3), "\n")

    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = df[df["mode"] == "full_b3"].copy()
    for c in ("C", "C_port", "fetches", "sim_makespan"):
        g[c + "_n"] = g[c] / g.groupby(["regime", "seed"])[c].transform("min")
    regimes = sorted(g.regime.unique())
    xs = [("C_n", "static makespan C (÷ instance best)"), ("fetches_n", "fetches (÷ instance best)"),
          ("C_port_n", "port-aware bound C_port (÷ instance best)")]
    fig, axes = plt.subplots(len(regimes), len(xs), figsize=(4.6 * len(xs), 4.0 * len(regimes)), squeeze=False)
    colors = {"SGC": "#7f7f7f", "CFSS": "#bcbd22", "RDI": "#d62728", "GBS": "#1f77b4", "LNS": "#2ca02c", "other": "k"}
    for i, regime in enumerate(regimes):
        h = g[g.regime == regime]
        for j, (x, label) in enumerate(xs):
            ax = axes[i][j]
            for fam, hh in h.groupby("family"):
                ax.scatter(hh[x], hh.sim_makespan_n, s=12, alpha=0.7, label=fam, color=colors.get(fam, "k"))
            ax.set_xlabel(label)
            ax.set_ylabel("simulated makespan (÷ instance best)")
            ax.set_title(f"{regime}-bound regime")
            ax.set_yscale("log")
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(dst, f"fig1_static_vs_sim.{ext}"), dpi=150)
    df.to_csv(os.path.join(dst, "runs.csv"), index=False)
    print(f"written to {dst}")


if __name__ == "__main__":
    main()
