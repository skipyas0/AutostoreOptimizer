"""Collect campaign results into flat tables (one row per LNS run / CP solve, plus anytime trajectories).

    $PY scripts/campaign/collect.py --root results/campaign --exp P0_lns P0_cp      # -> <root>/collected/

Writes, per experiment ``<exp>``:

- ``<root>/collected/<exp>_runs.csv``: experiment, arm, start, instance, run, the final incumbent's
  ``evaluation.solutions.metrics`` (makespan, fetches, robot_work, balance, ...), iterations, median
  iteration / search / overhead time, setup time, the start's metrics (``start_*``), the job's peak RSS and
  wall time (``_done`` marker), and the lower bounds (``bounds.csv``) if present.
- ``<root>/collected/<exp>_traj.csv.gz``: (run, t, iteration, best makespan) at every improvement, t in
  seconds since the run started (setup + cumulative iteration time; CP: solver time + build time), for
  anytime curves and equal-time comparisons. Every run starts with a t = 0 row holding the warm start.

The final makespan is checked against the LNS log's best value (a mismatch raises).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

import pandas as pd  # noqa: E402

from evaluation.solutions import metrics, read_solution  # noqa: E402
from experiments import EXPERIMENTS  # noqa: E402
from instance import Instance  # noqa: E402
from run import build_jobs  # noqa: E402

KEYS = ["makespan", "fetches", "robot_work", "robot_util_pct", "workload_imb_pct", "makespan_imb_pct"]
SETUP_RE = re.compile(r"Setup time before first iteration: ([\d.]+)s")


def setup_times(exp_dir: str) -> dict[int, float]:
    out = {}
    for path in glob.glob(os.path.join(exp_dir, "run_*.log")):
        k = int(re.search(r"run_(\d+)\.log$", path).group(1))
        with open(path, errors="replace") as f:
            m = SETUP_RE.search(f.read())
        if m:
            out[k] = float(m.group(1))
    return out


def collect_job(j, inst: Instance, done: dict, start_m: dict):
    runs, traj = [], []
    base = dict(experiment=j.exp.name, arm=j.arm, start=j.start, instance=j.instance,
                peak_rss_gb=done.get("peak_rss_gb"), job_wall_s=done.get("wall_s"),
                **{f"start_{k}": start_m.get(k) for k in KEYS})
    if j.cp:
        for seed in j.exp.seeds:
            d = os.path.join(j.folder, "cp_mono", f"seed{seed}")
            if not os.path.exists(os.path.join(d, "kpis.json")):
                continue
            kp = json.load(open(os.path.join(d, "kpis.json")))
            pr = json.load(open(os.path.join(d, "progress.json")))
            runs.append(dict(base, run=seed, iters=None, setup_s=kp["build_time"],
                             **{k: kp.get(k) for k in KEYS}))
            traj.append(dict(experiment=j.exp.name, arm=j.arm, start=j.start, instance=j.instance, run=seed,
                             t=0.0, iteration=None, best=float(pr["start_makespan"])))
            for r in pr["records"]:
                traj.append(dict(experiment=j.exp.name, arm=j.arm, start=j.start, instance=j.instance, run=seed,
                                 t=pr["build_time"] + r["time"], iteration=None, best=r["best"]))
        return runs, traj
    exps = sorted(glob.glob(os.path.join(j.folder, "experiments", "*", "experiment_dataframe.pkl")))
    if not exps:
        return runs, traj
    exp_dir = os.path.dirname(exps[-1])  # latest experiment folder of the job
    df = pd.read_pickle(exps[-1])
    setups = setup_times(exp_dir)
    for run_id, g in df.groupby("run_id"):
        g = g.sort_values("iteration")
        sol = os.path.join(exp_dir, f"run{run_id}-solution.npz")
        if not os.path.exists(sol):
            sol = os.path.join(exp_dir, f"run{run_id}-solution.pkl")
        m = metrics(*read_solution(sol), inst)
        if m["makespan"] != g.best.iloc[-1]:
            raise ValueError(f"{exp_dir} run {run_id}: solution makespan {m['makespan']} != best {g.best.iloc[-1]}")
        setup = setups.get(int(run_id), 0.0)
        runs.append(dict(base, run=int(run_id), iters=len(g), setup_s=setup,
                         iter_s=g.total_iter_time.median(), search_s=g.solve_cpp_search.median(),
                         overhead_s=(g.total_iter_time - g.solve_cpp_search).median(),
                         active_orders=g.num_optimized.median(), **{k: m[k] for k in KEYS}))
        t = setup + g.total_iter_time.cumsum().values
        best = g.best.values
        traj.append(dict(experiment=j.exp.name, arm=j.arm, start=j.start, instance=j.instance, run=int(run_id),
                         t=0.0, iteration=-1, best=float(start_m.get("makespan", best[0]))))  # the warm start
        keep = [0] + [i for i in range(1, len(best)) if best[i] < best[i - 1]] + [len(best) - 1]
        for i in sorted(set(keep)):
            traj.append(dict(experiment=j.exp.name, arm=j.arm, start=j.start, instance=j.instance, run=int(run_id),
                             t=float(t[i]), iteration=int(g.iteration.values[i]), best=float(best[i])))
    return runs, traj


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--exp", nargs="+", required=True, choices=list(EXPERIMENTS))
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    out = os.path.join(root, "collected")
    os.makedirs(out, exist_ok=True)
    starts = pd.read_csv(os.path.join(root, "starts.csv")).set_index(["start", "instance"])
    bounds = pd.read_csv(os.path.join(root, "bounds.csv")) if os.path.exists(os.path.join(root, "bounds.csv")) else None
    cache: dict[str, Instance] = {}
    for name in a.exp:
        runs, traj = [], []
        for j in build_jobs(root, [name], "*", "*", "*"):
            if j.status() != "ok":
                continue
            done = json.load(open(j.done_file))
            if j.instance not in cache:
                cache[j.instance] = Instance.from_pickle(os.path.join(root, "instances", j.instance, "instance.pkl"))
            sm = starts.loc[(j.start, j.instance)].to_dict() if (j.start, j.instance) in starts.index else {}
            r, t = collect_job(j, cache[j.instance], done, sm)
            runs += r
            traj += t
        df = pd.DataFrame(runs)
        if bounds is not None and len(df):
            df = df.merge(bounds[["instance", "W_lb", "C_lb"]], on="instance", how="left")
        df.to_csv(os.path.join(out, f"{name}_runs.csv"), index=False)
        pd.DataFrame(traj).to_csv(os.path.join(out, f"{name}_traj.csv.gz"), index=False)
        print(f"{name}: {len(df)} runs from {df.instance.nunique() if len(df) else 0} instances -> {out}")


if __name__ == "__main__":
    main()
