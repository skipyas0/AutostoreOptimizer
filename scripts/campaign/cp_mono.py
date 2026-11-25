"""Monolithic CP baseline: the full C++ CP model, warm-started from the instance folder's heuristic solution.

    python scripts/campaign/cp_mono.py <instance folder> --time-limit 3600 --seed 1 [--no-symmetry-breaking]

The folder is a campaign job folder (``instance.pkl``, ``config.json``, ``heuristic_solution.pkl``, as for
matheuristic.py). Writes ``<folder>/cp_mono/seed<seed>/``:

- ``progress.json``: one record per incumbent (time, best makespan, bound, gap), from CP Optimizer's listener.
  The clock starts at the solve call; model building is reported separately as ``build_time``.
- ``solution.npz``: the final solution (result_store format) and ``kpis.json``: makespan, robot work and the
  other ``evaluation.solutions.metrics`` of the final solution, checked with ``validate_solution``.
- ``cplog.txt``: CP Optimizer's log.

Same solve settings as the precalculated CP runs (``precalculate_instances_for_lns``): C++ backend, makespan
objective, 1 worker, symmetry breaking on (it helps a monolithic search; LNS runs without it).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

import cpp_cp_model  # noqa: E402
import result_store  # noqa: E402
from evaluation.solutions import metrics  # noqa: E402
from heuristics import HeuristicSolution, validate_solution  # noqa: E402
from instance import Instance  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--time-limit", type=float, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--symmetry-breaking", action=argparse.BooleanOptionalAction, default=True)
    a = ap.parse_args()

    out = os.path.join(a.folder, "cp_mono", f"seed{a.seed}")
    os.makedirs(out, exist_ok=True)
    inst = Instance.from_pickle(os.path.join(a.folder, "instance.pkl"))
    with open(os.path.join(a.folder, "config.json")) as f:
        horizon = json.load(f)["horizon"]
    with open(os.path.join(a.folder, "heuristic_solution.pkl"), "rb") as f:
        start = pickle.load(f)

    t0 = time.perf_counter()
    mdl, handles = cpp_cp_model.build_model(inst, add_symmetry_breaking=a.symmetry_breaking, horizon=horizon)
    sp = start.to_cpp_warmstart(mdl, handles)  # strict: raises on an invalid warm start
    listener = cpp_cp_model.ProgressCollector()
    mdl.add_solver_listener(listener)
    mdl.set_starting_point(sp)
    build_time = time.perf_counter() - t0
    print(f"model built in {build_time:.1f} s; solving for {a.time_limit:.0f} s (seed {a.seed})", flush=True)

    res = mdl.solve(time_limit=a.time_limit, workers=a.workers, random_seed=a.seed, log_verbosity="Normal",
                    log_filename=os.path.join(out, "cplog.txt"), collect_progress=True)
    solve_time = time.perf_counter() - t0 - build_time
    progress = [{k: v for k, v in r.items() if k != "delta"} for r in listener.records]
    with open(os.path.join(out, "progress.json"), "w") as f:
        json.dump({"build_time": build_time, "solve_time": solve_time, "start_makespan": start.makespan,
                   "records": progress}, f)

    sol = HeuristicSolution.from_cp(res, handles) if res else None
    if sol is None:
        kpis = {"status": "no solution"}
    else:
        if res.get_solution() is not None:
            result_store.dump(res.get_solution(), os.path.join(out, "solution.npz"))
        violations = validate_solution(sol, inst, horizon=horizon)
        visits = [(s, ev.sku) for s, evs in sol.bin_events.items() for ev in evs]
        picks = [(s, k, x, y) for (o, s, k), (x, y) in sol.pick_events.items()]
        kpis = {"status": str(res.get_solve_status()), "violations": violations[:10], **metrics(visits, picks, inst)}
    kpis.update(build_time=build_time, solve_time=solve_time, seed=a.seed, time_limit=a.time_limit,
                symmetry_breaking=a.symmetry_breaking)
    with open(os.path.join(out, "kpis.json"), "w") as f:
        json.dump(kpis, f, indent=1)
    print(json.dumps({k: kpis.get(k) for k in ("status", "makespan", "robot_work", "fetches")}), flush=True)


if __name__ == "__main__":
    main()
