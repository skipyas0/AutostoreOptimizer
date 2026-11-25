"""Lower-bound calibration on small instances.

    .venv/bin/python scripts/lb_calibration/run.py --out results/lb_calibration [--cp-time 120]

For each small generated instance it computes:
  * lower_bounds.py bounds: W_global, W_lane, C_lb;
  * the exact optimum of the time-free sequence relaxation (seq_relaxation_lb, CP-SAT): W_seq;
  * the monolithic CP model (CP Optimizer, docplex), warm-started from the best heuristic:
    min makespan C* and min robot work W* (objective replaced by the weighted fetch count),
    each with its status and proven bound;
  * the best heuristic C and W.
One JSON line per instance goes to <out>/rows.jsonl; finished instances are skipped.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

import cp_model  # noqa: E402
from datagen import generate_data  # noqa: E402
from heuristics import run_heuristic  # noqa: E402
from heuristics.evolve import robot_work, run_gbs_evolve  # noqa: E402
from lower_bounds import makespan_lb, seq_relaxation_lb, visit_work, work_lb_lane  # noqa: E402

H = 10000
HEURS = ["lb_sgc", "lb_sgc_share", "gbs_max_sharing", "gbs_cluster_lpt", "rdi"]
# (S, L, orders, skus, movecap); skus small enough that several SKUs are shared by > L orders
SIZES = [(1, 1, 8, 30, 4), (1, 2, 10, 30, 4), (2, 1, 10, 40, 6), (2, 2, 12, 40, 6), (2, 2, 16, 60, 8), (3, 2, 18, 80, 10)]
SEEDS = [1, 2, 3]


def solve_cp(inst, warm, objective: str, time_limit: float, workers: int) -> dict:
    mdl, handles = cp_model.build_model(inst, add_symmetry_breaking=True, horizon=H, objective_func="makespan")
    w = visit_work(inst)
    work = mdl.sum(w[k] * mdl.presence_of(iv) for (s, k, e), iv in handles["F"].items() if k in w)
    mdl.add_kpi(work, "work")
    if objective == "work":
        mdl.remove(mdl.get_objective())
        mdl.minimize(work)
    sp = warm.to_warmstart(mdl, handles, strict=False)
    if sp is not None:
        mdl.set_starting_point(sp)
    t0 = time.perf_counter()
    res = mdl.solve(TimeLimit=time_limit, Workers=workers, LogVerbosity="Quiet", RandomSeed=1)
    out = dict(status=res.get_solve_status(), runtime=time.perf_counter() - t0, value=None, bound=None)
    if res.is_solution():
        out["value"] = int(round(res.get_objective_values()[0]))
        out["bound"] = int(round(res.get_objective_bounds()[0]))
        kpis = res.get_kpis()
        out["makespan"] = int(kpis["makespan"])
        out["fetches"] = int(kpis["bin_fetches"])
        out["work"] = int(kpis["work"])
    return out


def one(S, L, O, K, mc, seed, cp_time, workers, evolve_evals) -> dict:
    inst = generate_data(num_stations=S, lanes_per_station=L, num_orders=O, num_skus=K, seed=seed,
                         sku_popularity_skew=1.5, movecap=mc)
    wb = work_lb_lane(inst, time_limit=30, workers=workers)
    best_c = best_w = None
    for h in HEURS:
        with contextlib.redirect_stdout(io.StringIO()):
            sol = run_heuristic(h, inst, horizon=H)
        if not sol.feasible:
            continue
        if best_c is None or sol.makespan < best_c[0].makespan:
            best_c = (sol, h)
        w = robot_work(sol, inst)
        if best_w is None or w < best_w[1]:
            best_w = (sol, w, h)
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        ev = run_gbs_evolve(inst, horizon=H, method="ls", fitness="work", window=3, max_evals=evolve_evals, seed=seed)
    ev_rt = time.perf_counter() - t0
    ev_w = robot_work(ev, inst) if ev.feasible else None
    if ev_w is not None and ev_w < best_w[1]:
        best_w = (ev, ev_w, "gbs_evolve_work")
    cpC = solve_cp(inst, best_c[0], "makespan", cp_time, workers)
    cpW = solve_cp(inst, best_w[0], "work", cp_time, workers)
    sq = seq_relaxation_lb(inst, time_limit=cp_time, workers=workers, hint=best_w[0])
    cb = makespan_lb(inst, work_lb=max(wb.value, sq.value))
    lines = sum(len(set(inst.orders_req[o])) for o in inst.O)
    return dict(
        size=f"{S}-{L}-{O}-{K}-{mc}", seed=seed, lines=lines, skus=len(visit_work(inst)),
        hot_skus=wb.hot_skus, W_global=wb.global_lb, W_lane=wb.value,
        W_seq=sq.value, W_seq_optimal=sq.optimal, W_seq_ub=sq.upper, seq_runtime=sq.runtime,
        C_lb=cb.value, C_lb_picker=cb.picker, C_lb_robot=cb.robot,
        evolve_W=ev_w, evolve_C=ev.makespan if ev.feasible else None, evolve_runtime=ev_rt,
        heur_C=best_c[0].makespan, heur_C_name=best_c[1], heur_W=best_w[1], heur_W_name=best_w[2],
        cp_C=cpC, cp_W=cpW,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/lb_calibration")
    ap.add_argument("--cp-time", type=float, default=120)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--evolve-evals", type=int, default=10000)
    ap.add_argument("--sizes", nargs="*", default=None, help="S-L-O-K-mc strings (default: SIZES)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "rows.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            done = {(r["size"], r["seed"]) for r in map(json.loads, f)}
    sizes = [tuple(map(int, s.split("-"))) for s in a.sizes] if a.sizes else SIZES
    for sz in sizes:
        for seed in SEEDS:
            key = ("-".join(map(str, sz)), seed)
            if key in done:
                continue
            row = one(*sz, seed, a.cp_time, a.workers, a.evolve_evals)
            with open(path, "a") as f:
                f.write(json.dumps(row) + "\n")
            c, w = row["cp_C"], row["cp_W"]
            print(f"{key} W: global {row['W_global']} lane {row['W_lane']} seq {row['W_seq']}"
                  f"{'' if row['W_seq_optimal'] else '?'} | CP W* {w['value']} (bd {w['bound']}, {w['status']}) heur {row['heur_W']} evolve {row['evolve_W']}"
                  f" || C: lb {row['C_lb']} | CP C* {c['value']} (bd {c['bound']}, {c['status']}) heur {row['heur_C']}",
                  flush=True)


if __name__ == "__main__":
    main()
