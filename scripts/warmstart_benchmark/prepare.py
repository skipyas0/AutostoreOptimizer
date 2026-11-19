"""Warm-start benchmark, step 1: one LNS root per warm-start heuristic.

    .venv/bin/python scripts/warmstart_benchmark/prepare.py            # defaults below
    .venv/bin/python scripts/warmstart_benchmark/prepare.py --starts lb_sgc_share gbs_max_sharing

For every start heuristic H and source instance folder GEN... this creates

    <out>/<H>/precalculated_instances/GEN.../
        instance.pkl, instance_summary.txt   symlinks to the source folder
        config.json                          copy with "heuristic": H
        heuristic_solution.pkl / _summary    H's solution = the LNS warm start (validated)
        experiments/                         empty; LNS writes its runs here

so ``matheuristic.py GEN...`` run from ``<out>/<H>`` starts from H, and every root can be
read by ``python -m evaluation compare <out>/<H>/precalculated_instances``. It also writes
``<out>/warm_starts.csv`` (the structure metrics of every warm start) and ``<out>/manifest.json``.
Takes a few seconds per instance.
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

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

import pandas as pd  # noqa: E402

from evaluation.solutions import metrics  # noqa: E402
from heuristics import validate_solution  # noqa: E402
from instance import Instance  # noqa: E402
from precalculate_instances_for_lns import solve_warmstart_heuristic  # noqa: E402

DEFAULT_STARTS = ["lb_sgc", "lb_sgc_share", "gbs_max_sharing", "gbs_cluster_lpt"]
# the instances of results/fetch_benchmark (5 and 10 stations, generator seeds 1-5)
DEFAULT_INSTANCES = [f"GEN{s}-{st}-3-28750-{o}-40" for s in range(1, 6) for st, o in ((5, 300), (10, 400))]


def prepare_one(src: str, root: str, start: str) -> dict:
    name = os.path.basename(src.rstrip("/"))
    dst = os.path.join(root, start, "precalculated_instances", name)
    os.makedirs(os.path.join(dst, "experiments"), exist_ok=True)
    for f in ("instance.pkl", "instance_summary.txt"):
        link = os.path.join(dst, f)
        if not os.path.lexists(link):
            os.symlink(os.path.join(os.path.abspath(src), f), link)

    with open(os.path.join(src, "config.json")) as f:
        config = json.load(f)
    config["heuristic"] = start
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(config, f)

    inst = Instance.from_pickle(os.path.join(src, "instance.pkl"))
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        sol, label = solve_warmstart_heuristic(start, inst, config)
    elapsed = time.perf_counter() - t0
    violations = validate_solution(sol, inst, horizon=config["horizon"])
    if not sol.feasible or violations:
        raise RuntimeError(f"{start} on {name}: infeasible or invalid warm start: {violations[:3]}")

    with open(os.path.join(dst, "heuristic_solution.pkl"), "wb") as f:
        pickle.dump(sol, f)
    with open(os.path.join(dst, "heuristic_summary.txt"), "w") as f:  # same format as precalculation
        f.write(f"{label} Result | Feasible: {sol.feasible} | Makespan: {sol.makespan} | "
                f"BinEvents: {sol.total_moves // 2} | Time: {elapsed:.3f}s\n")

    visits = [(s, ev.sku) for s, evs in sol.bin_events.items() for ev in evs]
    picks = [(s, k, a, b) for (o, s, k), (a, b) in sol.pick_events.items()]
    return dict(start=start, instance=name, runtime=elapsed, **metrics(visits, picks, inst))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(REPO, "results", "warmstart_benchmark"))
    ap.add_argument("--source", default=os.path.join(REPO, "precalculated_instances"),
                    help="folder with the source instance folders")
    ap.add_argument("--instances", nargs="+", default=DEFAULT_INSTANCES)
    ap.add_argument("--starts", nargs="+", default=DEFAULT_STARTS, help="registry names of the warm-start heuristics")
    args = ap.parse_args()

    rows = []
    for name in args.instances:
        for start in args.starts:
            rows.append(prepare_one(os.path.join(args.source, name), args.out, start))
            r = rows[-1]
            print(f"{start:18s} {name}: makespan {r['makespan']}, fetches {r['fetches']}, "
                  f"workload imb. {r['workload_imb_pct']:.1f}%", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "warm_starts.csv"), index=False)
    commit = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({"warm_starts_commit": commit, "starts": args.starts, "instances": args.instances,
                   "created": time.strftime("%Y-%m-%d %H:%M")}, f, indent=2)
    print(df.groupby("start", sort=False)[["makespan", "fetches", "robot_work", "workload_imb_pct", "makespan_imb_pct"]]
          .mean().round(1).to_string())


if __name__ == "__main__":
    main()
