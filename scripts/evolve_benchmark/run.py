"""Evolve-benchmark: does evolving GBS order sequences beat the GBS starts?

    .venv/bin/python scripts/evolve_benchmark/run.py --out results/evolve_benchmark   # ~35 min on 10 cores

For each instance (the 15 LNS benchmark instances GEN1-5 x 5/10/15 stations), arm and seed it runs
``run_gbs_evolve`` with a fixed evaluation budget and writes one JSON line per job to
``<out>/runs.jsonl`` (fitness, makespan, robot work W, lower bounds on W, convergence history).
The GBS seeds (overlap / lpt / cluster_lpt partitions, rule max_sharing) are recorded per job.
Analysis: ``scripts/evolve_benchmark/analyze.py <out>``.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
from multiprocessing import Pool

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from heuristics import validate_solution  # noqa: E402
from heuristics.evolve import robot_work, run_gbs_evolve  # noqa: E402
from instance import Instance  # noqa: E402

INSTANCES = [f"GEN{s}-{st}-3-28750-{o}-40" for s in range(1, 6) for st, o in ((5, 300), (10, 400), (15, 600))]
ARMS = {
    "ls_makespan": dict(method="ls", fitness="makespan", window=3),
    "ga_makespan": dict(method="ga", fitness="makespan", window=3),
    "ls_work": dict(method="ls", fitness="work", window=3),
    "ga_work": dict(method="ga", fitness="work", window=3),
    "ga_makespan_bias": dict(method="ga", fitness="makespan", window=3, evolve_bias=True),
}


def work_bounds(inst: Instance, assignment: dict[int, list[int]]) -> tuple[int, int]:
    """(one visit per SKU, one visit per (station, SKU) of `assignment`) robot work."""
    w = {k: inst.rt[k] + inst.rt_ret[k] for o in inst.O for k in inst.orders_req[o]}
    global_lb = sum(w.values())
    part_lb = sum(w[k] for os_ in assignment.values() for k in {k for o in os_ for k in inst.orders_req[o]})
    return global_lb, part_lb


def job(args: tuple[str, str, int, str, int]) -> dict:
    name, arm, seed, inst_dir, evals = args
    inst = Instance.from_pickle(os.path.join(inst_dir, name, "instance.pkl"))
    with open(os.path.join(inst_dir, name, "config.json")) as f:
        horizon = json.load(f)["horizon"]
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        sol = run_gbs_evolve(inst, horizon=horizon, max_evals=evals, seed=seed, **ARMS[arm])
    runtime = time.perf_counter() - t0
    viol = validate_solution(sol, inst, horizon=horizon)
    m = sol.meta
    glb, plb = work_bounds(inst, m["sequence"])
    return dict(
        instance=name, stations=len(inst.S), orders=len(inst.O), movecap=inst.movecap, arm=arm, seed=seed,
        evals=m["evals"], runtime=runtime, feasible=sol.feasible, violations=len(viol),
        makespan=sol.makespan, work=robot_work(sol, inst), visits=sol.total_moves // 2,
        work_lb_global=glb, work_lb_partition=plb,
        seed_fitness={k: list(v) for k, v in m["seed_fitness"].items()},
        history=[[e, t, list(f)] for e, t, f in m["history"]], op_stats=m["op_stats"], n_bias=m["n_bias"],
        sequence={str(s): v for s, v in m["sequence"].items()},
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/evolve_benchmark")
    ap.add_argument("--instances-dir", default="precalculated_instances")
    ap.add_argument("--instances", nargs="*", default=INSTANCES)
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--evals", type=int, default=1500)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "runs.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["instance"], r["arm"], r["seed"]))
    todo = [(i, arm, s, a.instances_dir, a.evals) for i in a.instances for arm in a.arms for s in range(a.seeds)
            if (i, arm, s) not in done]
    todo.sort(key=lambda j: -int(j[0].split("-")[1]))  # big instances first: better packing
    print(f"{len(todo)} jobs ({len(done)} already done), {a.jobs} workers", flush=True)
    t0 = time.perf_counter()
    with Pool(a.jobs) as pool, open(path, "a") as f:
        for n, r in enumerate(pool.imap_unordered(job, todo), 1):
            f.write(json.dumps(r) + "\n")
            f.flush()
            el = time.perf_counter() - t0
            print(f"[{n}/{len(todo)}] {r['instance']} {r['arm']} s{r['seed']}: C={r['makespan']} W={r['work']} "
                  f"viol={r['violations']} {r['runtime']:.0f}s | elapsed {el / 60:.1f} min, "
                  f"ETA {el / n * (len(todo) - n) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
