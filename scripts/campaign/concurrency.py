"""Pilot: how much does a subproblem LNS iteration slow down when N jobs run at once?

    $PY scripts/campaign/concurrency.py --root results/campaign --src results/campaign/code \\
        --instance GEN1-10-3-28750-400-40 --levels 1 4 7 8 --iters 30

Runs N identical LNS processes (BASE configuration, start gbs_cluster_lpt, one run of ``--iters``
iterations), pinned to cores 0..N-1, for every N in ``--levels`` one after another, in
``<root>/runs/P0_concurrency/N<k>/``. Prints, per level and relative to N = 1, the median iteration time,
its search part (capped by ``--iter-time-limit``) and overhead part (build, extraction, presolve, merge),
and the final makespan. The search being capped, a slowdown shows as more overhead and as less search
work per iteration (a worse ``final_best`` at equal iterations). This decides ``--slots``: with a time budget, a slowdown lowers the
number of iterations every job gets, so every comparison must run at one fixed level.
Takes about (iters x ~5 s + setup) per level, ~15 min in total for the defaults.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pandas as pd  # noqa: E402

from experiments import BASE  # noqa: E402
from run import flags  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--instance", default="GEN1-10-3-28750-400-40")
    ap.add_argument("--start", default="gbs_cluster_lpt")
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 4, 7, 8])
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--py", default=os.environ.get("PY", sys.executable))
    a = ap.parse_args()
    root, src = os.path.abspath(a.root), os.path.abspath(a.src)
    rows = []
    for n in a.levels:
        procs = []
        for i in range(n):
            cwd = os.path.join(root, "runs", "P0_concurrency", f"N{n}", f"p{i}")
            folder = os.path.join(cwd, "precalculated_instances", a.instance)
            if os.path.exists(folder):
                shutil.rmtree(folder)
            os.makedirs(os.path.join(folder, "experiments"))
            for d, fn in ((os.path.join(root, "instances", a.instance), "instance.pkl"),
                          (os.path.join(root, "instances", a.instance), "instance_summary.txt"),
                          (os.path.join(root, "instances", a.instance), "config.json"),
                          (os.path.join(root, "starts", a.start, a.instance), "heuristic_solution.pkl")):
                shutil.copyfile(os.path.join(d, fn), os.path.join(folder, fn))
            cmd = ["taskset", "-c", str(i)] if shutil.which("taskset") else []
            cmd += [a.py, os.path.join(src, "matheuristic.py"), a.instance, "--runs", "1", "--iters", str(a.iters),
                    *flags(BASE)]
            procs.append((cwd, subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                                env=dict(os.environ, OMP_NUM_THREADS="1", MPLBACKEND="Agg"))))
        print(f"N={n}: {n} processes running ...", flush=True)
        for cwd, p in procs:
            p.wait()
            df_path = sorted(glob.glob(os.path.join(cwd, "precalculated_instances", a.instance, "experiments", "*",
                                                    "experiment_dataframe.pkl")))
            if p.returncode != 0 or not df_path:
                print(f"  process in {cwd} failed (rc {p.returncode})")
                continue
            df = pd.read_pickle(df_path[-1])
            # the search is capped by --iter-time-limit, so a slowdown shows in the overhead (model build,
            # extraction, presolve, merge) and in less search work per iteration, not in the search time
            rows.append(dict(N=n, iter=df.total_iter_time.median(), search=df.solve_cpp_search.median(),
                             overhead=(df.total_iter_time - df.solve_cpp_search).median(),
                             final_best=df.best.iloc[-1]))
    res = pd.DataFrame(rows).groupby("N").median()
    for c in list(res.columns):
        res[c + "_vs_N1"] = res[c] / res[c].iloc[0]
    print(res.round(3).to_string())
    out = os.path.join(root, "runs", "P0_concurrency", "summary.csv")
    res.to_csv(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
