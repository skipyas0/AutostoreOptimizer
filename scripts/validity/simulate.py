"""Validity check, step 3: execute every static schedule in the simulator.

    .venv/bin/python scripts/validity/simulate.py [--out results/validity]

For every <out>/<regime>/seed<N>/plans/<method>.json, and every LNS final solution under
<out>/<regime>/lns/ (written into plans/lns_<start>_run<k>.json first), runs the simulator in each
execution MODE, plus the simulator's own heuristics (NATIVE, no plan). Writes <out>/sim.csv.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bridge  # noqa: E402
from prepare import LAYOUT  # noqa: E402

import pandas as pd  # noqa: E402

# execution modes of a plan (simulator parameters)
MODES = {
    "full_b3": {"planMode": "full", "buffer": 3},    # the whole schedule, bins requested 3 ahead per station
    "full_b6": {"planMode": "full", "buffer": 6},
    "assign_b3": {"planMode": "assign", "buffer": 3},  # only order -> station and opening order
}
# the simulator's own heuristics (no plan)
NATIVE = {
    "sim_fcfs_b3": {"orderAssign": "fcfs", "buffer": 3},
    "sim_sharing_b3": {"orderAssign": "sharing", "buffer": 3},
    "sim_sharing_b6": {"orderAssign": "sharing", "buffer": 6},
}


def lns_plans(out: str) -> None:
    """Write the final LNS solutions as plans (plans/lns_<start>_run<k>.json), once."""
    import result_store

    for exp in glob.glob(os.path.join(out, "*", "lns", "precalculated_instances", "VAL-*", "experiments", "*")):
        name = exp.split(os.sep)[-3]
        _, regime, seed = name.split("-")
        d = os.path.join(out, regime, f"seed{seed}")
        cfg = json.load(open(os.path.join(exp, "..", "..", "config.json")))
        sim_skus = json.load(open(os.path.join(d, "sku_map.json")))
        n_st = len(json.load(open(os.path.join(d, "sim", "geometry.json")))["stations"])
        for f in glob.glob(os.path.join(exp, "run*-solution.*")):
            run = os.path.basename(f).split("-")[0]
            dst = os.path.join(d, "plans", f"lns_{cfg['heuristic']}_{run}.json")
            if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(f):
                continue
            sol = result_store.load(f)
            vs = sol.var_sols if hasattr(sol, "var_sols") else sol
            plan = bridge.pad_plan(bridge.plan_from_solution(vs, os.path.basename(dst)[:-5], sim_skus), n_st)
            json.dump(plan, open(dst, "w"))


def run_one(job) -> dict:
    d, regime, seed, method, mode, params, plan_path = job
    inst = os.path.join(d, "sim", f"instance-seed{seed}.json")
    plan = json.load(open(plan_path)) if plan_path else None
    try:
        r = bridge.simulate(inst, LAYOUT, plan, params)
    except RuntimeError as e:
        r = dict(sim_done=False, sim_problems=str(e)[-300:])
    return dict(regime=regime, seed=seed, method=method, mode=mode, **r)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(bridge.REPO, "results", "validity"))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    lns_plans(args.out)
    jobs = []
    for d in sorted(glob.glob(os.path.join(args.out, "*", "seed*"))):
        regime, seed = d.split(os.sep)[-2], int(d.split("seed")[-1])
        for p in sorted(glob.glob(os.path.join(d, "plans", "*.json"))):
            method = os.path.basename(p)[:-5]
            for mode, params in MODES.items():
                jobs.append((d, regime, seed, method, mode, params, p))
        for method, params in NATIVE.items():
            jobs.append((d, regime, seed, method, "native", params, None))
    with Pool(args.workers) as pool:
        rows = pool.map(run_one, jobs, chunksize=4)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "sim.csv"), index=False)
    bad = df[~df.sim_done.astype(bool)]
    print(f"{len(df)} simulator runs, {len(bad)} not finished")
    if len(bad):
        print(bad[["regime", "seed", "method", "mode", "sim_problems"]].head(10).to_string())


if __name__ == "__main__":
    main()
