"""Experiment evaluation suite.

  python -m evaluation compare <root> [--out DIR] [--times 60 300 900]
                               [--baselines CP RDI] [--configs LNS-1 LNS-3]
                               [--pattern 'GEN*-10-*'] [--png-tables]
      Every LNS configuration, CP and heuristic found under <root> (a folder of
      precalculated instance folders). Default output: results/compare_<root name>.

  python -m evaluation scaling [results/lns_scaling] [--out DIR] [--png-tables]
      Full-model vs. subproblem overhead from benchmark_lns_scaling.py.
"""

import argparse
import os


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m evaluation",
                                 description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="report", required=True)

    c = sub.add_parser("compare", help="LNS configs vs. CP vs. heuristics")
    c.add_argument("root")
    c.add_argument("--out")
    c.add_argument("--times", type=float, nargs="+", default=[60, 300, 900],
                   help="time points [s]; the largest is the budget for 'final'")
    c.add_argument("--baselines", nargs="+", default=["CP", "RDI"],
                   help="methods shown next to the best LNS config in the reduced variants")
    c.add_argument("--configs", nargs="+", help="only these LNS ids (e.g. LNS-1 LNS-4)")
    c.add_argument("--pattern", default="GEN*", help="glob for instance folders")
    c.add_argument("--png-tables", action="store_true", help="also render tables to PNG")

    s = sub.add_parser("scaling", help="LNS overhead scaling (full vs. subproblem)")
    s.add_argument("root", nargs="?", default="results/lns_scaling")
    s.add_argument("--out")
    s.add_argument("--png-tables", action="store_true")

    a = ap.parse_args(argv)
    if a.report == "compare":
        from .reports.compare import Comparison
        times = [int(t) if float(t).is_integer() else t for t in a.times]
        out = a.out or os.path.join("results", "compare_" + os.path.basename(
            os.path.normpath(a.root)))
        cmp = Comparison(a.root, out, times, a.baselines, a.pattern, a.png_tables,
                         a.configs).run()
        print(f"Best LNS config: {cmp.look[cmp.best_lns]['label']}")
        print(f"Wrote tables, figures and data to {out}")
    else:
        from .reports.scaling import run
        out = run(a.root, a.out, a.png_tables)
        print(f"Wrote tables and figures to {out}")


if __name__ == "__main__":
    main()
