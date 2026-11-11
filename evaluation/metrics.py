"""Method-agnostic metrics on the long run table (see data.py).

Conventions
  BKS                   best makespan any method found on the instance (all runs)
  gap to BKS            (X - BKS) / BKS * 100           0 = best known, lower is better
  impr. over start      (H - X) / H * 100               H = LNS start heuristic
  Per-instance comparisons use each method's mean over its runs on that instance.
"""

import numpy as np
import pandas as pd
from scipy import stats

from .data import values_at


def add_reference(runs, instances):
    """Attach BKS and the start-heuristic value to every run."""
    bks = runs.groupby("instance").final.min().rename("bks")
    out = runs.merge(bks, on="instance", how="left")
    out = out.merge(instances[["instance", "start_value"]], on="instance", how="left")
    out["gap_final"] = 100 * (out.final - out.bks) / out.bks
    out["impr_final"] = 100 * (out.start_value - out.final) / out.start_value
    return out


def at_times(runs, times):
    """Value, gap to BKS and improvement over start at each time point (per run)."""
    out = runs.copy()
    for T in times:
        val = np.array([values_at(r.t, r.v, T)[0] for r in runs.itertuples()])
        out[f"val_{T}"] = val
        out[f"gap_{T}"] = 100 * (val - out.bks) / out.bks
        out[f"impr_{T}"] = 100 * (out.start_value - val) / out.start_value
    return out


def curves(runs, grid, kind="gap"):
    """Per-run curve on a time grid: gap to BKS (or improvement over start)."""
    vals = np.vstack([values_at(r.t, r.v, grid) for r in runs.itertuples()])
    if kind == "gap":
        ref = runs.bks.values[:, None]
        return 100 * (vals - ref) / ref
    ref = runs.start_value.values[:, None]
    return 100 * (ref - vals) / ref


def instance_means(runs, col="final"):
    """method x instance table of mean `col` over runs."""
    return runs.pivot_table(index="instance", columns="method", values=col, aggfunc="mean")


def ranks(runs, col="final"):
    """Rank of each method per instance (1 = best; ties share the average rank)."""
    return instance_means(runs, col).rank(axis=1, method="average")


def wins(runs, col="final"):
    """Instances on which a method's mean is the best (ties count for all tied)."""
    m = instance_means(runs, col)
    return m.eq(m.min(axis=1), axis=0).sum()


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, min(1.0, (len(p) - i) * p[j]))
        adj[j] = running
    return adj


def paired_tests(runs, reference, others, col="final"):
    """Wilcoxon signed-rank of `reference` vs each other method on per-instance means.

    Returns rows: method, n, median diff (other - reference), wins/ties/losses of the
    reference, p, Holm-adjusted p.
    """
    m = instance_means(runs, col)
    rows = []
    for o in others:
        d = (m[o] - m[reference]).dropna()
        if (d != 0).sum() == 0:
            p = 1.0
        else:
            p = stats.wilcoxon(d, zero_method="zsplit").pvalue
        rows.append(dict(method=o, n=len(d), median_diff=d.median(),
                         ref_better=(d > 0).sum(), ties=(d == 0).sum(),
                         ref_worse=(d < 0).sum(), p=p))
    df = pd.DataFrame(rows)
    if len(df):
        df["p_holm"] = holm(df.p.values)
    return df
