"""RDI-LB share vs. LNS on the Large instances: is the difference real, and where does RDI win?

Usage: python scripts/rdi_vs_lns_large.py [runs.csv] [--lns LNS-6] [--size Large]

Part 1 (significance), per instance, with d = LNS run - RDI-LB share (in % of BKS):
  one-sample t-test and exact sign test of d against 0, 95 % CI of the mean,
  TOST equivalence test for margins of 0.25 % and 0.5 % of BKS.
Across instances:
  one-way ANOVA of d by instance (is the difference instance-dependent?),
  paired t / Wilcoxon on instance means, random-effects (DerSimonian-Laird) mean
  difference with its 95 % CI.

RDI-LB share is deterministic, so all variance comes from the LNS seeds.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MARGINS = (0.25, 0.5)  # equivalence margins, % of BKS


def tost(d, margin):
    """Two one-sided t-tests: p for |mean(d)| < margin."""
    n, m, se = len(d), d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    if se == 0:
        return 0.0 if abs(m) < margin else 1.0
    p_lo = 1 - stats.t.cdf((m + margin) / se, n - 1)
    p_hi = stats.t.cdf((m - margin) / se, n - 1)
    return max(p_lo, p_hi)


def sign_test(d):
    pos, neg = (d > 0).sum(), (d < 0).sum()
    return stats.binomtest(min(pos, neg), pos + neg, 0.5).pvalue if pos + neg else 1.0


def per_instance(runs, lns, ref="RDI-LB share"):
    rows, diffs = [], {}
    for inst, g in runs.groupby("instance"):
        bks = g.bks.iloc[0]
        r = g[g.method == ref].final.iloc[0]
        l = g[g.method == lns].final.values
        d = 100 * (l - r) / bks  # > 0: RDI better
        diffs[inst] = d
        t = stats.ttest_1samp(d, 0)
        ci = stats.t.interval(0.95, len(d) - 1, d.mean(), d.std(ddof=1) / np.sqrt(len(d)))
        row = {"instance": inst, "RDI-LB share": r, f"{lns} mean": l.mean(),
               f"{lns} min": l.min(), f"{lns} max": l.max(),
               "diff mean [% BKS]": d.mean(), "CI low": ci[0], "CI high": ci[1],
               "RDI better in runs": f"{(d > 0).sum()}/{len(d)}",
               "p t-test": t.pvalue, "p sign": sign_test(d)}
        for mg in MARGINS:
            row[f"p equiv ±{mg}%"] = tost(d, mg)
        rows.append(row)
    df = pd.DataFrame(rows)
    from evaluation.metrics import holm
    df.insert(df.columns.get_loc("p t-test") + 1, "p t Holm", holm(df["p t-test"].values))
    return df, diffs


def across(diffs):
    groups = list(diffs.values())
    f = stats.f_oneway(*groups)
    means = np.array([g.mean() for g in groups])
    var = np.array([g.var(ddof=1) / len(g) for g in groups])
    # DerSimonian-Laird random effects
    w = 1 / var
    fixed = (w * means).sum() / w.sum()
    q = (w * (means - fixed) ** 2).sum()
    k = len(means)
    tau2 = max(0.0, (q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    w_re = 1 / (var + tau2)
    re = (w_re * means).sum() / w_re.sum()
    se_re = np.sqrt(1 / w_re.sum())
    out = {
        "ANOVA F (difference depends on instance)": f.statistic, "ANOVA p": f.pvalue,
        "mean of instance diffs [% BKS]": means.mean(),
        "paired t p (instance means)": stats.ttest_1samp(means, 0).pvalue,
        "Wilcoxon p (instance means)": stats.wilcoxon(means).pvalue,
        "random-effects mean [% BKS]": re,
        "random-effects 95% CI": (re - 1.96 * se_re, re + 1.96 * se_re),
        "between-instance SD tau [% BKS]": np.sqrt(tau2),
        "within-instance SE (mean) [% BKS]": np.sqrt(var.mean()),
        "I2 [%]": 100 * max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0,
    }
    for mg in MARGINS:
        out[f"equivalence ±{mg}% shown (RE CI inside)"] = (
            re - 1.96 * se_re > -mg and re + 1.96 * se_re < mg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="?", default="results/lns_config_comparison/data/runs.csv")
    ap.add_argument("--lns", nargs="+", default=["LNS-6", "LNS-5", "LNS-2"])
    ap.add_argument("--size", default="Large")
    a = ap.parse_args()
    runs = pd.read_csv(a.runs)
    runs = runs[runs["size"] == a.size]
    pd.set_option("display.width", 220, "display.max_columns", 30)
    for lns in a.lns:
        tab, diffs = per_instance(runs, lns)
        print(f"\n=== {lns} vs RDI-LB share ({a.size}); diff > 0 means RDI-LB share is better")
        print(tab.round(4).to_string(index=False))
        for k, v in across(diffs).items():
            print(f"  {k}: {v if isinstance(v, (bool, np.bool_)) else np.round(v, 4)}")


if __name__ == "__main__":
    main()
