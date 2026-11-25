"""Summarise scripts/lb_calibration/run.py output.

    .venv/bin/python scripts/lb_calibration/analyze.py results/lb_calibration

Per instance and per size (mean):
  W_lane_vs_min   lane bound vs the exact minimum W (sequence relaxation, when proven optimal)
  W_bestC_vs_min  robot work of the best-makespan schedule vs the minimum W (the trade-off)
  C_gap           best known C (CP min-makespan, heuristics, evolved GBS) vs C_lb
  W_gap           best known W (CP min-work, heuristics, evolved GBS) vs the best W bound
"""

import json
import os
import sys

import pandas as pd


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else "results/lb_calibration"
    rows = [json.loads(l) for l in open(os.path.join(root, "rows.jsonl"))]
    out = []
    for r in rows:
        cpC, cpW = r["cp_C"], r["cp_W"]
        cands_C = [(r["heur_C"], None), (r["evolve_C"], None), (cpC["value"], cpC.get("work"))]
        best_C = min(c for c, _ in cands_C if c is not None)
        best_W = min(w for w in (r["heur_W"], r["evolve_W"], cpW["value"]) if w is not None)
        w_lb = max(r["W_lane"], r["W_seq"])
        w_min = r["W_seq_ub"] if r["W_seq_optimal"] else None
        out.append(dict(
            size=r["size"], seed=r["seed"], lines=r["lines"], hot=r["hot_skus"],
            W_global=r["W_global"], W_lane=r["W_lane"], W_seq=r["W_seq"], W_seq_opt=r["W_seq_optimal"],
            W_seq_ub=r["W_seq_ub"], W_best=best_W, W_best_from="cp" if best_W == cpW["value"] else "heur/evolve",
            evolve_W=r["evolve_W"], heur_W=r["heur_W"],
            C_lb=r["C_lb"], C_best=best_C, cpC_status=cpC["status"], cpC_bound=cpC["bound"],
            W_at_cpC=cpC.get("work"),
            W_lane_vs_min=100 * (r["W_lane"] / w_min - 1) if w_min else None,
            W_bestC_vs_min=100 * (cpC["work"] / w_min - 1) if w_min and cpC.get("work") else None,
            W_relax_ub_vs_best=100 * (r["W_seq_ub"] / best_W - 1) if r["W_seq_ub"] else None,
            W_gap=100 * (best_W / w_lb - 1),
            C_gap=100 * (best_C / r["C_lb"] - 1),
        ))
    df = pd.DataFrame(out)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(df.round(1).to_string(index=False))
    cols = ["lines", "hot", "W_lane_vs_min", "W_bestC_vs_min", "W_relax_ub_vs_best", "W_gap", "C_gap"]
    agg = df.groupby("size", sort=False)[cols].mean().round(1)
    agg["seq_opt"] = df.groupby("size", sort=False).W_seq_opt.sum()
    agg["cp_proved"] = df.groupby("size", sort=False).cpC_status.apply(lambda s: (s == "Optimal").sum())
    print("\nMeans per size (S-L-orders-SKUs-movecap):")
    print(agg.to_string())
    md = os.path.join(root, "summary.md")
    with open(md, "w") as f:
        f.write("# Lower-bound calibration\n\n```\n" + agg.to_string() + "\n```\n\n```\n" + df.round(1).to_string(index=False) + "\n```\n")
    print(f"\nwrote {md}")


if __name__ == "__main__":
    main()
