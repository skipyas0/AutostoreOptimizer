"""Warm-start benchmark, step 3: does a better warm start give a better LNS result?

    .venv/bin/python scripts/warmstart_benchmark/analyze.py [<out>]    # default results/warmstart_benchmark

Reads <out>/<start>/precalculated_instances/*/experiments/*/ (one experiment per start and instance,
RUNS runs each) and <out>/warm_starts.csv, writes tables (Markdown + CSV) and figures to <out>/analysis/.

Per run, the final incumbent (``runK-solution.pkl``, the best solution by makespan) is measured with
``evaluation.solutions.metrics`` (the same metrics as the warm starts). Statistics use per-instance means
over runs (n = instances), paired by instance: Wilcoxon signed-rank with Holm correction, 95% t-intervals.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from evaluation import tables  # noqa: E402
from evaluation.solutions import metrics, read_solution  # noqa: E402
from instance import Instance  # noqa: E402

LABELS = {"lb_sgc": "LB-SGC", "lb_sgc_share": "LB-SGC share", "gbs_max_sharing": "GBS",
          "gbs_cluster_lpt": "GBS cluster-LPT"}
BASELINE = "lb_sgc"  # the current default LNS start
KEYS = ["makespan", "fetches", "robot_work", "robot_util_pct", "workload_imb_pct", "makespan_imb_pct"]


def size_of(inst: str) -> str:
    _, s, _, _, o, _ = inst.split("-")
    return f"{s} st. / {o} ord."


def load(out: str):
    starts = pd.read_csv(os.path.join(out, "warm_starts.csv"))
    runs, traj = [], []
    for start in LABELS:
        for d in sorted(glob.glob(os.path.join(out, start, "precalculated_instances", "*"))):
            inst_name = os.path.basename(d)
            inst = Instance.from_pickle(os.path.join(d, "instance.pkl"))
            exps = sorted(glob.glob(os.path.join(d, "experiments", "*", "experiment_dataframe.pkl")))
            if not exps:
                continue
            exp = os.path.dirname(exps[-1])  # latest experiment (one per start and instance expected)
            cfg = json.load(open(os.path.join(exp, "experiment_config.json")))
            df = pd.read_pickle(os.path.join(exp, "experiment_dataframe.pkl"))
            for run_id, g in df.groupby("run_id"):
                g = g.sort_values("iteration")
                m = metrics(*read_solution(os.path.join(exp, f"run{run_id}-solution.pkl")), inst)
                if m["makespan"] != g.best.iloc[-1]:
                    raise ValueError(f"{exp} run {run_id}: solution makespan {m['makespan']} != best {g.best.iloc[-1]}")
                runs.append(dict(start=start, instance=inst_name, size=size_of(inst_name), run=run_id,
                                 iters=len(g), iter_time=g.total_iter_time.mean(),
                                 lns_model=cfg["lns_model"], **{k: m[k] for k in KEYS}))
                traj.append(pd.DataFrame(dict(start=start, instance=inst_name, size=size_of(inst_name),
                                              run=run_id, iteration=g.iteration.values, best=g.best.values)))
    return starts, pd.DataFrame(runs), pd.concat(traj)


def holm(p):
    order, adj, running = np.argsort(p), np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def paired(a, b) -> dict:
    d = np.asarray(a, float) - np.asarray(b, float)
    n, mean, sd = len(d), d.mean(), d.std(ddof=1)
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return dict(n=n, mean_diff=mean, ci_low=mean - half, ci_high=mean + half,
                wins=int((d < 0).sum()), losses=int((d > 0).sum()), p_wilcoxon=p)


def write(df: pd.DataFrame, path: str, digits: int = 2) -> str:
    shown = df.copy()
    for c in shown.columns:
        if shown[c].dtype.kind == "f":
            shown[c] = shown[c].map(lambda x: f"{x:.{digits}f}" if abs(x) >= 1e-3 or x == 0 else f"{x:.2e}")
    df.to_csv(path + ".csv", index=False)
    text = tables.to_markdown(shown)
    with open(path + ".md", "w") as f:
        f.write(text)
    return text


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "results", "warmstart_benchmark")
    dst = os.path.join(out, "analysis")
    os.makedirs(dst, exist_ok=True)
    starts, runs, traj = load(out)
    runs.to_csv(os.path.join(dst, "runs.csv"), index=False)
    print(f"{len(runs)} runs, {runs.instance.nunique()} instances, starts: {sorted(runs.start.unique())}, "
          f"iterations per run: {sorted(runs.iters.unique())}\n")

    final = runs.groupby(["start", "instance", "size"])[KEYS + ["iter_time"]].mean().reset_index()
    st = starts.rename(columns={k: f"start_{k}" for k in KEYS})[["start", "instance"] + [f"start_{k}" for k in KEYS]]
    both = final.merge(st, on=["start", "instance"])
    both["lns_gain_pct"] = 100 * (both.start_makespan - both.makespan) / both.start_makespan
    both["W_kept"] = both.robot_work / both.start_robot_work
    # final makespan relative to the robot-work bound W / MoveCap (1 = robots never idle)
    movecap = {i: Instance.from_pickle(os.path.join(out, BASELINE, "precalculated_instances", i, "instance.pkl")).movecap
               for i in both.instance.unique()}
    both["C_over_W_bound"] = both.makespan / (both.robot_work / both.instance.map(movecap))
    base = both[both.start == BASELINE].set_index("instance")
    for k in ("makespan", "fetches", "robot_work", "start_makespan", "start_robot_work"):
        both[f"d_{k}_pct"] = 100 * (both[k] / both.instance.map(base[k]) - 1)

    # 1. overall: warm start -> LNS final, relative to LB-SGC
    rows = []
    for start, label in LABELS.items():
        for size, g in [("all", both[both.start == start])] + list(both[both.start == start].groupby("size")):
            rows.append({"Start": label, "Size": size, "n": len(g),
                         "Start makespan vs LB-SGC [%]": g.d_start_makespan_pct.mean(),
                         "Final makespan vs LB-SGC [%]": g.d_makespan_pct.mean(),
                         "Final fetches vs LB-SGC [%]": g.d_fetches_pct.mean(),
                         "Final robot work vs LB-SGC [%]": g.d_robot_work_pct.mean(),
                         "LNS gain over its start [%]": g.lns_gain_pct.mean(),
                         "Robot work kept": g.W_kept.mean(),
                         "C / (W/MoveCap)": g.C_over_W_bound.mean(),
                         "Workload imb. start → final [%]": f"{g.start_workload_imb_pct.mean():.1f} → {g.workload_imb_pct.mean():.1f}",
                         "Makespan imb. start → final [%]": f"{g.start_makespan_imb_pct.mean():.1f} → {g.makespan_imb_pct.mean():.1f}",
                         "s / iter": g.iter_time.mean()})
    print(write(pd.DataFrame(rows), os.path.join(dst, "t1_overall")), "\n")

    # 2. per instance: final makespan (mean of runs)
    wide = both.pivot(index="instance", columns="start", values="makespan")[list(LABELS)]
    t2 = wide.rename(columns=LABELS).reset_index()
    t2.insert(1, "size", t2.instance.map(size_of))
    t2["best start"] = wide.idxmin(axis=1).map(LABELS).values
    print(write(t2, os.path.join(dst, "t2_per_instance_final_makespan"), digits=1), "\n")

    # 3. paired tests on final makespan (in % of LB-SGC's final makespan, so instances weigh equally)
    rel = both.pivot(index="instance", columns="start", values="d_makespan_pct")
    tests = [(s, BASELINE, f"{LABELS[s]} vs LB-SGC") for s in LABELS if s != BASELINE]
    tests += [("gbs_max_sharing", "lb_sgc_share", "GBS vs LB-SGC share"),
              ("gbs_cluster_lpt", "lb_sgc_share", "GBS cluster-LPT vs LB-SGC share"),
              ("gbs_cluster_lpt", "gbs_max_sharing", "GBS cluster-LPT vs GBS (balance effect)")]
    t3 = pd.DataFrame([{"Comparison (A − B), final makespan [pp of LB-SGC]": label, **paired(rel[a], rel[b])}
                       for a, b, label in tests])
    t3["p_holm"] = holm(t3.p_wilcoxon.values)
    print(write(t3, os.path.join(dst, "t3_paired_tests"), digits=4), "\n")

    # 4. what carries over: start -> final, per instance and start (correlations over all 40 pairs)
    r_w = stats.pearsonr(both.d_start_robot_work_pct, both.d_makespan_pct)
    r_c = stats.pearsonr(both.d_start_makespan_pct, both.d_makespan_pct)
    r_i = stats.pearsonr(both.start_workload_imb_pct, both.workload_imb_pct)
    t4 = pd.DataFrame([
        {"Relation (40 start × instance pairs)": "start robot work → final makespan (both vs LB-SGC)", "r": r_w[0], "p": r_w[1]},
        {"Relation (40 start × instance pairs)": "start makespan → final makespan (both vs LB-SGC)", "r": r_c[0], "p": r_c[1]},
        {"Relation (40 start × instance pairs)": "start workload imb. → final workload imb.", "r": r_i[0], "p": r_i[1]}])
    print(write(t4, os.path.join(dst, "t4_carry_over"), digits=3), "\n")

    # 5. figure: best makespan over iterations, relative to LB-SGC's final makespan per instance
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref = base.makespan
    traj["rel"] = 100 * (traj.best / traj.instance.map(ref) - 1)
    curve = traj.groupby(["size", "start", "iteration"]).rel.mean().reset_index()
    sizes = sorted(curve["size"].unique())
    fig, axes = plt.subplots(1, len(sizes), figsize=(5 * len(sizes), 3.6), sharey=False)
    for ax, size in zip(np.atleast_1d(axes), sizes):
        for start, label in LABELS.items():
            c = curve[(curve["size"] == size) & (curve.start == start)]
            ax.plot(c.iteration, c.rel, label=label)
        ax.axhline(0, color="grey", lw=0.8, ls=":")
        ax.set_title(size)
        ax.set_xlabel("LNS iteration")
        ax.set_ylabel("best makespan vs final LB-SGC [%]")
    np.atleast_1d(axes)[0].legend(frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(dst, f"fig1_makespan_over_iterations.{ext}"), dpi=150)
    # iterations each start needs to reach LB-SGC's final makespan (per run, first iteration at or below it)
    hit = traj.assign(ok=traj.best <= traj.instance.map(ref)).query("ok").groupby(["start", "instance", "run"]).iteration.min()
    reach = hit.groupby("start").agg(["count", "median"]).rename(index=LABELS)
    reach["count"] = reach["count"].astype(str) + f" / {traj.groupby(['start', 'instance', 'run']).ngroups // len(LABELS)}"
    print("Runs reaching LB-SGC's final makespan (and median iteration):\n" + reach.to_string(), "\n")
    print(f"written to {dst}")


if __name__ == "__main__":
    main()
