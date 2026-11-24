"""Analyse an evolve-benchmark run: ``analyze.py <out> [--instances-dir precalculated_instances]``.

Writes ``<out>/analysis/summary.md`` (tables) and ``<out>/analysis/convergence.png``. Reference
points are recomputed per instance: the GBS seeds (``gbs_max_sharing`` = overlap partition, lpt,
cluster_lpt) and the robot-work lower bounds (one visit per SKU; one visit per (station, SKU) of a
partition).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from heuristics.evolve import robot_work  # noqa: E402
from heuristics.gbs import run_gbs  # noqa: E402
from instance import Instance  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from run import work_bounds  # noqa: E402


def md(df: pd.DataFrame, fmt: str = ".4f") -> str:
    """Markdown table without the optional `tabulate` dependency."""
    df = df.copy()
    df.columns = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in df.columns]
    idx = [" ".join(map(str, i)) if isinstance(i, tuple) else str(i) for i in df.index]
    cell = lambda v: format(v, fmt) if isinstance(v, (float, np.floating)) else str(v)  # noqa: E731
    out = ["| | " + " | ".join(df.columns) + " |", "|---" * (len(df.columns) + 1) + "|"]
    out += ["| " + i + " | " + " | ".join(cell(v) for v in row) + " |" for i, row in zip(idx, df.itertuples(index=False))]
    return "\n".join(out)


def seed_table(names: list[str], inst_dir: str) -> pd.DataFrame:
    rows = []
    for name in names:
        inst = Instance.from_pickle(os.path.join(inst_dir, name, "instance.pkl"))
        with open(os.path.join(inst_dir, name, "config.json")) as f:
            horizon = json.load(f)["horizon"]
        for part in ("overlap", "lpt", "cluster_lpt"):
            with contextlib.redirect_stdout(io.StringIO()):
                sol = run_gbs(inst, horizon=horizon, partition=part)
            glb, plb = work_bounds(inst, sol.meta["assignment"])
            rows.append(dict(instance=name, seed=part, makespan=sol.makespan, work=robot_work(sol, inst),
                             work_lb_global=glb, work_lb_partition=plb, movecap=inst.movecap))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--instances-dir", default="precalculated_instances")
    a = ap.parse_args()
    runs = pd.DataFrame([json.loads(line) for line in open(os.path.join(a.out, "runs.jsonl"))])
    os.makedirs(os.path.join(a.out, "analysis"), exist_ok=True)
    assert (runs.violations == 0).all() and runs.feasible.all(), "invalid evolved solutions"
    seeds = seed_table(sorted(runs.instance.unique()), a.instances_dir)
    seeds.to_csv(os.path.join(a.out, "analysis", "seeds.csv"), index=False)

    sd = seeds.set_index(["instance", "seed"])
    ref = pd.DataFrame({
        "C_gbs": sd.xs("overlap", level="seed").makespan,
        "W_gbs": sd.xs("overlap", level="seed").work,
        "C_clpt": sd.xs("cluster_lpt", level="seed").makespan,
        "W_clpt": sd.xs("cluster_lpt", level="seed").work,
        "C_bestseed": seeds.groupby("instance").makespan.min(),
        "W_bestseed": seeds.groupby("instance").work.min(),
        "W_lb_global": seeds.groupby("instance").work_lb_global.first(),
        "W_lb_part_clpt": sd.xs("cluster_lpt", level="seed").work_lb_partition,
    })
    r = runs.join(ref, on="instance")
    r["dC_gbs"] = r.makespan / r.C_gbs - 1
    r["dC_best"] = r.makespan / r.C_bestseed - 1
    r["dW_gbs"] = r.work / r.W_gbs - 1
    r["dW_best"] = r.work / r.W_bestseed - 1
    r["C_over_WMC"] = r.makespan / (r.work / r.movecap)
    # share of the gap between the best seed's W and the evolved partition's own bound that was closed
    r["W_gap_closed"] = (r.W_bestseed - r.work) / (r.W_bestseed - r.work_lb_partition)
    r.to_csv(os.path.join(a.out, "analysis", "runs.csv"), index=False)

    lines = ["# Evolve benchmark", "",
             f"{runs.instance.nunique()} instances, {runs.seed.nunique()} seeds, "
             f"{int(runs.evals.median())} evaluations per run, all solutions valid.", "",
             "## Reference points (GBS seeds, means over instances)", ""]
    s2 = seeds.join(ref[["C_gbs", "W_gbs"]], on="instance")
    s2["dC_gbs"] = s2.makespan / s2.C_gbs - 1
    s2["dW_gbs"] = s2.work / s2.W_gbs - 1
    s2["C_over_WMC"] = s2.makespan / (s2.work / s2.movecap)
    s2["W_vs_part_lb"] = s2.work / s2.work_lb_partition - 1
    s2["W_vs_global_lb"] = s2.work / s2.work_lb_global - 1
    t = s2.groupby("seed")[["dC_gbs", "dW_gbs", "C_over_WMC", "W_vs_part_lb", "W_vs_global_lb"]].mean()
    lines += [md(t, ".4f"), ""]

    lines += ["## Arms (mean over instances and seeds; d* relative change)", "",
              "dC_gbs / dW_gbs: vs gbs_max_sharing; dC_best / dW_best: vs the best of the 3 seeds on that "
              "metric. W_gap_closed: (W_bestseed - W) / (W_bestseed - partition bound). "
              "p: one-sided Wilcoxon over instances (seed-averaged) that the arm beats the best seed.", ""]
    agg = []
    for arm, g in r.groupby("arm"):
        per_inst = g.groupby("instance")[["dC_best", "dW_best"]].mean()
        row = g[["dC_gbs", "dC_best", "dW_gbs", "dW_best", "C_over_WMC", "W_gap_closed", "runtime"]].mean()
        row["p_C"] = wilcoxon(per_inst.dC_best, alternative="less").pvalue if (per_inst.dC_best != 0).any() else 1
        row["p_W"] = wilcoxon(per_inst.dW_best, alternative="less").pvalue if (per_inst.dW_best != 0).any() else 1
        row["C_wins"] = f"{(per_inst.dC_best < 0).sum()}/{len(per_inst)}"
        row["W_wins"] = f"{(per_inst.dW_best < 0).sum()}/{len(per_inst)}"
        agg.append(row.rename(arm))
    lines += [md(pd.DataFrame(agg), ".4f"), ""]

    lines += ["## By instance size (seed- and instance-mean)", ""]
    t = r.groupby(["arm", "stations"])[["dC_best", "dW_best", "C_over_WMC"]].mean().unstack("stations")
    lines += [md(t, ".4f"), ""]

    lines += ["## Convergence: improvement over the start, share of the final improvement reached", ""]
    fracs = (0.1, 0.25, 0.5, 0.75)
    conv = []
    for _, row in r.iterrows():
        h = row.history
        f0, fN = h[0][2][0], h[-1][2][0]
        rec = {"arm": row.arm}
        for q in fracs:
            e = q * row.evals
            fe = [x[2][0] for x in h if x[0] <= e][-1]
            rec[f"{int(q * 100)}%"] = (f0 - fe) / (f0 - fN) if f0 != fN else 1.0
        rec["evals_of_last_improvement"] = h[-1][0] / row.evals
        conv.append(rec)
    lines += [md(pd.DataFrame(conv).groupby("arm").mean(), ".3f"), ""]

    lines += ["## Operator success (child better than parent, summed over runs)", ""]
    ops = {}
    for _, row in r.iterrows():
        for op, (n, k) in row.op_stats.items():
            o = ops.setdefault((row.arm, op), [0, 0])
            o[0] += n
            o[1] += k
    t = pd.Series({k: v[1] / v[0] for k, v in ops.items()}).unstack()
    lines += [md(t, ".4f"), ""]
    with open(os.path.join(a.out, "analysis", "summary.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    grid = np.linspace(0, 1, 101)
    for metric_i, (ax, title) in enumerate(zip(axes, ("makespan fitness arms: C", "work fitness arms: W"))):
        for arm, g in r.groupby("arm"):
            if ("work" in arm) != (metric_i == 1):
                continue
            curves = []
            for _, row in g.iterrows():
                h = row.history
                f0 = h[0][2][0]
                curves.append([[x[2][0] for x in h if x[0] <= q * row.evals][-1] / f0 - 1 for q in grid])
            ax.plot(grid * g.evals.median(), np.mean(curves, axis=0) * 100, label=arm)
        ax.set_title(title)
        ax.set_xlabel("evaluations")
        ax.set_ylabel("% vs best seed (mean)")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "analysis", "convergence.png"), dpi=130)


if __name__ == "__main__":
    main()
