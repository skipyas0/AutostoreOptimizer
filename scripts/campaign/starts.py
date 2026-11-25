"""Campaign step 2: warm starts (and lower bounds) for every instance, computed once and shared by all experiments.

    .venv/bin/python scripts/campaign/starts.py --root results/campaign -j 14                 # all starts, all sets
    .venv/bin/python scripts/campaign/starts.py --root results/campaign --sets eval --starts gbs_regret_lc
    .venv/bin/python scripts/campaign/starts.py --root results/campaign --bounds -j 14         # lane W / C bounds

Reads ``<root>/instances/sets.json`` (``instances.py``) and writes, per start S and instance I,

    <root>/starts/S/I/heuristic_solution.pkl   the warm start (validated; never overwritten)
    <root>/starts/S/I/heuristic_summary.txt    same format as the precalculation
    <root>/starts/S/I/metrics.json             evaluation.solutions.metrics + runtime

and ``<root>/starts.csv`` with one row per (start, instance). Finished pairs are skipped, so the script can be
re-run after an interruption or with more starts. ``--bounds`` writes ``<root>/bounds.csv`` instead (the lane
bound on robot work W and the makespan bound of ``lower_bounds.py``, CP-SAT with 1 worker per instance).

Starts are not timed for the LNS comparison (jobs run in parallel with ``-j``); ``runtime`` is informative only.
The two evolution starts are the expensive ones (10 000 / 1 500 GBS decodings: ~15-30 min / ~3-5 min per
instance at 400-800 orders).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

import pandas as pd  # noqa: E402

from evaluation.solutions import metrics  # noqa: E402
from heuristics import run_heuristic, validate_solution  # noqa: E402
from instance import Instance  # noqa: E402

EVOLVE_SEEDS = ("overlap", "lpt", "cluster_lpt", "regret")
# label -> (registry name, parameters); the label is the start's name in every experiment
STARTS = {
    "lb_sgc": ("lb_sgc", {}),  # the LNS start of all experiments before Nov 2026
    "lb_sgc_share": ("lb_sgc_share", {}),
    "rdi_lb_share": ("rdi_lb_share", {}),
    "cfss_share": ("cfss_share", {}),
    "gbs_no_defer": ("gbs_no_defer", {}),
    "gbs_max_sharing": ("gbs_max_sharing", {}),
    "gbs_cluster_lpt": ("gbs_cluster_lpt", {}),
    "gbs_regret": ("gbs_regret", {}),
    "gbs_regret_lc": ("gbs_regret_lc", {}),
    # evolution, seeded with all four GBS partitions; work fitness = the W-minimising start,
    # makespan fitness = the falsification arm (low C at about the seeds' W)
    "gbs_evolve_work": ("gbs_evolve_work", {"max_evals": 10000, "seed_partitions": EVOLVE_SEEDS}),
    "gbs_evolve_makespan": ("gbs_evolve", {"max_evals": 1500, "seed_partitions": EVOLVE_SEEDS}),
}
COST = {"gbs_evolve_work": 100, "gbs_evolve_makespan": 15, "rdi_lb_share": 5}  # rough, for scheduling only


def solve_start(root: str, start: str, inst_name: str) -> dict:
    dst = os.path.join(root, "starts", start, inst_name)
    done = os.path.join(dst, "metrics.json")
    if os.path.exists(done):
        with open(done) as f:
            return json.load(f)
    src = os.path.join(root, "instances", inst_name)
    inst = Instance.from_pickle(os.path.join(src, "instance.pkl"))
    with open(os.path.join(src, "config.json")) as f:
        horizon = json.load(f)["horizon"]
    name, params = STARTS[start]
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        sol = run_heuristic(name, inst, horizon=horizon, **params)
    elapsed = time.perf_counter() - t0
    violations = validate_solution(sol, inst, horizon=horizon)
    if not sol.feasible or violations:
        raise RuntimeError(f"{start} on {inst_name}: infeasible or invalid warm start: {violations[:3]}")
    visits = [(s, ev.sku) for s, evs in sol.bin_events.items() for ev in evs]
    picks = [(s, k, a, b) for (o, s, k), (a, b) in sol.pick_events.items()]
    row = dict(start=start, instance=inst_name, heuristic=name, params=json.dumps(params, sort_keys=True),
               runtime=elapsed, **metrics(visits, picks, inst))
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "heuristic_solution.pkl"), "wb") as f:
        pickle.dump(sol, f)
    with open(os.path.join(dst, "heuristic_summary.txt"), "w") as f:
        f.write(f"{start} Result | Feasible: {sol.feasible} | Makespan: {sol.makespan} | "
                f"BinEvents: {sol.total_moves // 2} | Time: {elapsed:.3f}s\n")
    with open(done + ".tmp", "w") as f:
        json.dump(row, f)
    os.replace(done + ".tmp", done)  # metrics.json last: its presence marks a finished start
    return row


def solve_bounds(root: str, inst_name: str, time_limit: float) -> dict:
    from lower_bounds import makespan_lb, work_lb_lane

    inst = Instance.from_pickle(os.path.join(root, "instances", inst_name, "instance.pkl"))
    t0 = time.perf_counter()
    wb = work_lb_lane(inst, time_limit=time_limit, workers=1)
    cb = makespan_lb(inst, work_lb=wb.value)
    return dict(instance=inst_name, W_lb=wb.value, W_lb_global=wb.global_lb, W_lb_optimal=wb.optimal,
                C_lb=cb.value, C_lb_picker=cb.picker, C_lb_robot=cb.robot, runtime=time.perf_counter() - t0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join(REPO, "results", "campaign"))
    ap.add_argument("--sets", nargs="+", default=None, help="instance sets (default: all in sets.json)")
    ap.add_argument("--starts", nargs="+", default=list(STARTS), choices=list(STARTS))
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--bounds", action="store_true", help="compute lower bounds instead of starts")
    ap.add_argument("--bound-time", type=float, default=300.0, help="CP-SAT time limit per lane bound [s]")
    a = ap.parse_args()

    with open(os.path.join(a.root, "instances", "sets.json")) as f:
        sets = json.load(f)
    names = [n for s in (a.sets or list(sets)) for n in sets[s]]
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(a.jobs) as ex:
        if a.bounds:
            futs = {ex.submit(solve_bounds, a.root, n, a.bound_time): n for n in names}
        else:
            jobs = sorted(((s, n) for s in a.starts for n in names),
                          key=lambda j: -COST.get(j[0], 1) * int(j[1].split("-")[4]))  # expensive first
            futs = {ex.submit(solve_start, a.root, s, n): (s, n) for s, n in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rows.append(fut.result())
            except Exception as e:  # report and continue; a re-run retries it
                print(f"FAILED {futs[fut]}: {e}", flush=True)
                continue
            r = rows[-1]
            el = time.time() - t0
            what = r["instance"] if a.bounds else f"{r['start']:20s} {r['instance']}"
            print(f"[{i}/{len(futs)}] {what} ({r['runtime']:.0f} s) | elapsed {el / 60:.0f} min, "
                  f"naive ETA {el / i * (len(futs) - i) / 60:.0f} min", flush=True)
    out = os.path.join(a.root, "bounds.csv" if a.bounds else "starts.csv")
    df = pd.DataFrame(rows)
    if os.path.exists(out) and not a.bounds:  # keep rows of starts / sets not in this call
        old = pd.read_csv(out)
        df = pd.concat([old, df]).drop_duplicates(["start", "instance"], keep="last")
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
