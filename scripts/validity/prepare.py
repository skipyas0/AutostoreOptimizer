"""Validity check, step 1: simulator instances -> static instances -> static schedules -> station plans.

    .venv/bin/python scripts/validity/prepare.py [--out results/validity] [--regimes ports robots] [--seeds 1-10]

Per regime and seed, writes <out>/<regime>/seed<N>/:
    sim/instance-seed<N>.json, sim/geometry.json   the simulator instance (wsim --dump)
    cal/                                          calibration run (the simulator's own heuristic, cycles)
    calibration.json, sku_map.json                fitted times; simulator SKU of each static SKU
    plans/<method>.json                           each static schedule as a simulator station plan
    static.csv                                    static metrics of every schedule
and an LNS root <out>/<regime>/lns/precalculated_instances/VAL-<regime>-<N>/ (instance.pkl, config.json,
heuristic_solution.pkl = the LNS warm start, experiments/), for run_lns.sh.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import pickle
import subprocess
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bridge  # noqa: E402

REPO = bridge.REPO
from evaluation.solutions import metrics  # noqa: E402
from heuristics import run_heuristic, validate_solution  # noqa: E402

LAYOUT = ["--site", "40,40,16,5"]  # 40 x 40 storage columns, 16 deep, 5 stations (about 20 000 bins)
BASE = {"lanes": 3, "orders": 300, "skus": 10000}
# robots = MoveCap; 'ports': ports are the bottleneck (robots mostly idle), 'robots': robots are
REGIMES = {"ports": {"robots": 40}, "robots": {"robots": 12}}
METHODS = [
    "sgc", "sgc_share", "lb_sgc", "lb_sgc_share", "ama_two_phase", "lb_ama_two_phase_singles",
    "cfss", "cfss_share", "rdi", "rdi_lb_share",
    "gbs_sequential", "gbs_no_defer", "gbs_max_sharing", "gbs_lpt", "gbs_critical_path",
    "gbs_cluster_lpt", "gbs_regret", "gbs_regret_lc",
]
LNS_START = "gbs_regret_lc"
HORIZON = 20000


def static_metrics(sol, inst) -> dict:
    visits = [(s, e.sku) for s, evs in sol.bin_events.items() for e in evs]
    picks = [(s, k, a, b) for (o, s, k), (a, b) in sol.pick_events.items()]
    return metrics(visits, picks, inst)


def prepare_one(job) -> list[dict]:
    out, regime, seed = job
    d = os.path.join(out, regime, f"seed{seed}")
    params = {**BASE, **REGIMES[regime]}
    inst_path = bridge.dump(os.path.join(d, "sim"), LAYOUT, params, [seed])[0]
    geo = json.load(open(os.path.join(d, "sim", "geometry.json")))
    sim = json.load(open(inst_path))
    cal = bridge.calibrate(inst_path, LAYOUT, os.path.join(d, "cal"))
    json.dump(cal, open(os.path.join(d, "calibration.json"), "w"), indent=1)
    inst, sim_skus, _ = bridge.static_instance(sim, geo, cal, seed)
    json.dump(sim_skus, open(os.path.join(d, "sku_map.json"), "w"))
    inst.to_pickle(os.path.join(d, "instance.pkl"))

    os.makedirs(os.path.join(d, "plans"), exist_ok=True)
    rows, sols = [], {}
    for h in METHODS:
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            sol = run_heuristic(h, inst, horizon=HORIZON)
        runtime = time.perf_counter() - t0
        bad = validate_solution(sol, inst, horizon=HORIZON) if sol.feasible else ["infeasible"]
        if bad:
            print(f"{regime} seed {seed} {h}: invalid ({bad[0][:80]}), skipped", flush=True)
            continue
        sols[h] = sol
        plan = bridge.pad_plan(bridge.plan_from_solution(sol, h, sim_skus), len(geo["stations"]))
        json.dump(plan, open(os.path.join(d, "plans", f"{h}.json"), "w"))
        rows.append(dict(regime=regime, seed=seed, method=h, runtime=runtime, **static_metrics(sol, inst)))
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(d, "static.csv"), index=False)

    # LNS root, warm-started from LNS_START
    name = f"VAL-{regime}-{seed}"
    root = os.path.join(out, regime, "lns", "precalculated_instances", name)
    os.makedirs(os.path.join(root, "experiments"), exist_ok=True)
    inst.to_pickle(os.path.join(root, "instance.pkl"))
    with open(os.path.join(root, "heuristic_solution.pkl"), "wb") as f:
        pickle.dump(sols[LNS_START], f)
    config = {"stations": len(inst.S), "lanes": len(inst.L), "orders": len(inst.O), "skus": len(inst.K),
              "movecap": inst.movecap, "symmetry_breaking": False, "heuristic": LNS_START, "horizon": HORIZON,
              "alpha": 1.0, "beta": 1.0, "gen_seed": seed, "source": "warehouse-sim validity check",
              "layout": LAYOUT, "params": params}
    json.dump(config, open(os.path.join(root, "config.json"), "w"))
    with open(os.path.join(root, "instance_summary.txt"), "w") as f:
        f.write(f"{name}: simulator instance {inst_path}\n")
    print(f"{regime} seed {seed}: {len(rows)} schedules, calibration makespan {cal['cal_makespan']:.0f} s", flush=True)
    return rows


def parse_seeds(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(REPO, "results", "validity"))
    ap.add_argument("--regimes", nargs="+", default=list(REGIMES))
    ap.add_argument("--seeds", default="1-10")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    jobs = [(args.out, r, s) for r in args.regimes for s in parse_seeds(args.seeds)]
    with Pool(min(args.workers, len(jobs))) as pool:
        rows = [r for rs in pool.map(prepare_one, jobs) for r in rs]
    commit = lambda p: subprocess.run(["git", "-C", p, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()  # noqa: E731
    json.dump({"created": time.strftime("%Y-%m-%d %H:%M"), "repo_commit": commit(REPO),
               "wsim_commit": commit(os.path.join(REPO, "warehouse-sim")), "layout": LAYOUT, "base": BASE,
               "regimes": {r: REGIMES[r] for r in args.regimes}, "seeds": args.seeds, "methods": METHODS,
               "lns_start": LNS_START, "horizon": HORIZON}, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    print(f"{len(rows)} schedules written under {args.out}")


if __name__ == "__main__":
    main()
