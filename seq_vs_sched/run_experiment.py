"""Benchmark: fetch-optimal sequences vs. makespan-optimal schedules.

Per instance, in this order:
  SEQ[a]        sequencing model, arm a in {none, balance}                 -> fetches f_a
  a|l           scheduling model restricted to SEQ[a] at level l           -> makespan M_{a,l}
                (greedy timing of SEQ[a] warm-starts level sequence, which warm-starts
                visits, which warm-starts assignment)
  cold          scheduling model, min makespan, lb_sgc warm start only     -> unbiased structure
  ref           scheduling model, min makespan, warm start = best schedule -> best known makespan M*
  fetch_opt     (optional) scheduling model, min bin fetches               -> timed fetch optimum

The monolithic CP rarely proves makespan optimality even on toy instances, so
"optimal" means best known (M* = min over all schedules found; "ref" reaches
it by construction). "cold" is reported separately because "ref" starts from a
fetch-optimal schedule and may inherit its structure.

Questions answered:
  Q1  how many fetches does the makespan-optimal schedule spend?   f_ref / f*
      and how similar is its sequence to the fetch-optimal one?    sim_{cold,ref}_vs_<arm>_*
  Q2  how good is the schedule that follows the fetch-optimal sequence?   M_ratio_<arm>|<level>

Usage (from the repo root):
  python -m seq_vs_sched.run_experiment --stations 2 3 --movecaps 4 1000 --seeds 1 2 3 4 5
  python -m seq_vs_sched.run_experiment --summarize seq_vs_sched/results/<run>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from statistics import mean, median

import cp_model
from datagen import generate_data
from heuristics import run_heuristic
from heuristics.solution import HeuristicSolution

from .fix_schedule import LEVELS, constrain_schedule_to_sequence, greedy_schedule_from_sequence
from .metrics import compare, describe
from .seq_model import solve_sequence_model
from .sequence import SequenceSolution, sequence_from_schedule

HERE = os.path.dirname(os.path.abspath(__file__))


def _solve_schedule(inst, cfg, time_limit, objective="makespan", seq=None, level=None, warm=None):
    """One scheduling CP solve. `warm`: HeuristicSolution starting point, or "heuristic" for lb_sgc.

    Returns (result dict, HeuristicSolution or None).
    """
    t0 = time.perf_counter()
    mdl, h = cp_model.build_model(inst, add_symmetry_breaking=True, horizon=0, objective_func=objective)
    if seq is not None:
        constrain_schedule_to_sequence(mdl, h, seq, level)
    if warm == "heuristic":
        warm = run_heuristic("lb_sgc", inst, horizon=h["horizon"])
    if warm is not None and warm.feasible:
        mdl.set_starting_point(warm.to_warmstart(mdl, h))
    res = mdl.solve(TimeLimit=time_limit, Workers=cfg["workers"], RandomSeed=cfg["cp_seed"],
                    LogVerbosity="Quiet", log_output=None)
    out = {"status": res.get_solve_status() if res else "None", "wall_time": time.perf_counter() - t0}
    if not res or not res.is_solution():
        return out, None
    kpis = res.get_kpis()
    out.update(makespan=kpis["makespan"], bin_fetches=kpis["bin_fetches"],
               bound=(res.get_objective_bounds() or [None])[0], solve_time=res.get_solve_time())
    return out, HeuristicSolution.from_cp(res, h)


# Most restrictive first: each level warm-starts the next (they are nested relaxations).
_LEVEL_CHAIN = ("sequence", "visits", "assignment")


def run_instance(params: dict, cfg: dict) -> dict:
    inst = generate_data(
        num_stations=params["stations"], lanes_per_station=params["lanes"], num_orders=params["orders"],
        num_skus=params["skus"], seed=params["seed"], movecap=params["movecap"],
    )
    n_lanes = len(inst.L)
    rec = {"params": params, "n_lines": sum(map(len, inst.orders_req.values())),
           "n_distinct_skus": len({k for v in inst.orders_req.values() for k in v}), "seq": {}, "sched": {}}
    seqs: dict[str, SequenceSolution] = {}
    schedules = []  # (makespan, fetches, HeuristicSolution) of every schedule found

    def record(key, out, hs, ref_seq=None):
        if hs is not None:
            s = sequence_from_schedule(hs, meta={"source": key})
            out.update(describe(s, n_lanes), solution=s.to_dict())
            if ref_seq is not None:
                out["vs_seq"] = compare(s, ref_seq)  # level=sequence: must be identical
            else:
                out["vs_seq"] = {arm: compare(s, q) for arm, q in seqs.items()}
            schedules.append((out["makespan"], out["bin_fetches"], hs))
        rec["sched"][key] = out

    # --- 1. sequencing model arms
    for arm in cfg["arms"]:
        s = solve_sequence_model(inst, secondary=arm, time_limit=cfg["seq_time"], workers=cfg["workers"], seed=cfg["cp_seed"])
        if s is None:
            rec["seq"][arm] = {"status": "no solution"}
            continue
        errs = s.validate(inst)
        assert not errs, errs
        seqs[arm] = s
        rec["seq"][arm] = {**s.meta, **describe(s, n_lanes), "solution": s.to_dict()}

    # --- 2. schedules restricted to each fetch-optimal sequence (Q2)
    for arm, s in seqs.items():
        warm = greedy_schedule_from_sequence(inst, s)  # always feasible for every level
        rec["seq"][arm]["greedy_makespan"] = warm.makespan
        for level in [lv for lv in _LEVEL_CHAIN if lv in cfg["levels"]]:
            out, hs = _solve_schedule(inst, cfg, cfg["fixed_time"], seq=s, level=level, warm=warm)
            record(f"{arm}|{level}", out, hs, ref_seq=s)
            warm = hs or warm

    # --- 3. makespan-optimal schedule, cold: lb_sgc warm start only (unbiased structure, Q1)
    out, hs = _solve_schedule(inst, cfg, cfg["sched_time"], warm="heuristic")
    record("cold", out, hs)

    # --- 4. makespan reference: warm start from the best schedule found so far
    if schedules:
        best = min(schedules, key=lambda t: (t[0], t[1]))[2]
        out, hs = _solve_schedule(inst, cfg, cfg["sched_time"], warm=best)
        record("ref", out, hs)

    # --- 5. optional: fetch-minimal *timed* schedule (lane hand-over inside a visit is allowed here)
    if cfg["sched_fetch"] and schedules:
        best = min(schedules, key=lambda t: (t[1], t[0]))[2]
        out, hs = _solve_schedule(inst, cfg, cfg["sched_time"], objective="bin_fetches", warm=best)
        record("fetch_opt", out, hs)
    return rec


# ---------------------------------------------------------------------------- summary
def flat_row(rec: dict) -> dict:
    p, seq, sch = rec["params"], rec["seq"], rec["sched"]
    nan = math.nan
    row = {**{k: p[k] for k in ("stations", "lanes", "orders", "skus", "movecap", "seed")},
           "lines": rec["n_lines"], "distinct_skus": rec["n_distinct_skus"]}
    f_best = min((v["fetches"] for v in seq.values() if "fetches" in v), default=nan)
    row.update(f_seq_best=f_best, f_seq_lb=max((v.get("fetch_bound") or 0 for v in seq.values()), default=nan))
    for arm, v in seq.items():
        row[f"f_seq_{arm}"] = v.get("fetches")
        row[f"maxst_seq_{arm}"] = v.get("max_station_fetches")
        row[f"status_seq_{arm}"] = v.get("status")

    found = [o["makespan"] for o in sch.values() if o.get("makespan") is not None and o is not sch.get("fetch_opt")]
    M = min(found, default=None)  # best known makespan (normally the "ref" solve)
    row.update(M_best=M, M_lb=max((o.get("bound") or 0 for k, o in sch.items() if k in ("cold", "ref")), default=nan),
               status_ref=sch.get("ref", {}).get("status"))
    for key in ("cold", "ref"):
        o = sch.get(key, {})
        row[f"M_{key}"] = o.get("makespan")
        row[f"M_ratio_{key}"] = o["makespan"] / M if o.get("makespan") and M else nan
        row[f"f_{key}"] = o.get("bin_fetches")
        row[f"f_ratio_{key}"] = o["bin_fetches"] / f_best if o.get("bin_fetches") and f_best else nan
        row[f"maxst_{key}"] = o.get("max_station_fetches")
        row[f"seq_lane_feasible_{key}"] = o.get("seq_lane_feasible")
        for arm, cmpd in o.get("vs_seq", {}).items():
            for k, v in cmpd.items():
                row[f"sim_{key}_vs_{arm}_{k}"] = v
    for key, o in sch.items():
        if "|" in key:
            row[f"M_{key}"] = o.get("makespan")
            row[f"status_{key}"] = o.get("status")
            row[f"M_ratio_{key}"] = o["makespan"] / M if o.get("makespan") and M else nan
    if "fetch_opt" in sch:
        o = sch["fetch_opt"]
        row["f_sched_fetchopt"] = o.get("bin_fetches")
        row["f_ratio_sched_fetchopt"] = o["bin_fetches"] / f_best if o.get("bin_fetches") and f_best else nan
        row["M_ratio_sched_fetchopt"] = o["makespan"] / M if o.get("makespan") and M else nan
    return row


def summarize(run_dir: str) -> list[dict]:
    recs = []
    for fn in sorted(os.listdir(run_dir)):
        if fn.startswith("inst_") and fn.endswith(".json"):
            with open(os.path.join(run_dir, fn)) as f:
                recs.append(json.load(f))
    rows = [flat_row(r) for r in recs]
    if not rows:
        print("no results")
        return rows
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(run_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    def agg(rs, key):
        vals = [r[key] for r in rs if isinstance(r.get(key), (int, float)) and not math.isnan(r[key])]
        return f"{mean(vals):.3f} / {median(vals):.3f} / {max(vals):.3f}" if vals else "-"

    groups = {}
    for r in rows:
        groups.setdefault((r["stations"], r["lanes"], r["orders"], r["skus"], r["movecap"]), []).append(r)
    ratio_keys = [c for c in cols if c.startswith("M_ratio_") or c.startswith("f_ratio_")]
    sim_keys = [c for c in cols if c.startswith("sim_") and not c.endswith("order_prec_pairs")]
    lines = ["# Sequencing vs. scheduling", "",
             "Cells: mean / median / max over seeds. M_ratio_x = makespan of x / best known makespan; "
             "f_ratio_x = fetches of x / best fetch-optimal sequence.", ""]
    for g, rs in sorted(groups.items()):
        lines += [f"## S={g[0]} L={g[1]} O={g[2]} K={g[3]} MoveCap={g[4]}  ({len(rs)} instances)", "",
                  "| metric | mean / median / max |", "|---|---|"]
        for k in ["f_seq_best", "f_seq_lb", "f_ref", "M_best", "M_lb"] + ratio_keys + sim_keys:
            lines.append(f"| {k} | {agg(rs, k)} |")
        n_opt = sum(r["status_ref"] == "Optimal" for r in rs)
        n_fopt = sum(r["f_seq_best"] == r["f_seq_lb"] for r in rs)
        lines += ["", f"Fetch optimum proven on {n_fopt}/{len(rs)}; makespan proven optimal on {n_opt}/{len(rs)}; "
                  f"makespan-optimal (ref) sequence fits the lanes in position semantics on "
                  f"{sum(bool(r.get('seq_lane_feasible_ref')) for r in rs)}/{len(rs)}.", ""]
    text = "\n".join(lines)
    with open(os.path.join(run_dir, "summary.md"), "w") as f:
        f.write(text)
    print(text)
    return rows


# ---------------------------------------------------------------------------- CLI
def _job(args):
    params, cfg, path = args
    rec = run_instance(params, cfg)
    rec["cfg"] = cfg
    with open(path, "w") as f:
        json.dump(rec, f, indent=1, default=str)
    o, c = rec["sched"].get("ref", {}), rec["sched"].get("cold", {})
    print(f"[done] {os.path.basename(path)}  f_seq={[v.get('fetches') for v in rec['seq'].values()]}  "
          f"M_cold={c.get('makespan')} M_ref={o.get('makespan')} ({o.get('status')}) f_ref={o.get('bin_fetches')}  "
          + "  ".join(f"{k}:{v.get('makespan')}" for k, v in rec["sched"].items() if "|" in k), flush=True)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stations", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--lanes", type=int, nargs="+", default=[4])
    ap.add_argument("--orders", type=int, nargs="+", default=[20])
    ap.add_argument("--skus", type=int, nargs="+", default=[100])
    ap.add_argument("--movecaps", type=int, nargs="+", default=[4, 1000],
                    help="robot-bound vs. effectively uncapped")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--arms", nargs="+", default=["none", "balance"], choices=["none", "balance"])
    ap.add_argument("--levels", nargs="+", default=list(LEVELS), choices=list(LEVELS))
    ap.add_argument("--seq-time", type=float, default=30)
    ap.add_argument("--sched-time", type=float, default=60)
    ap.add_argument("--fixed-time", type=float, default=30)
    ap.add_argument("--sched-fetch", action="store_true", help="also minimise fetches with the scheduling model")
    ap.add_argument("--workers", type=int, default=4, help="CP workers per solve")
    ap.add_argument("--jobs", type=int, default=1, help="instances solved in parallel")
    ap.add_argument("--cp-seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="run folder (default results/<timestamp>)")
    ap.add_argument("--summarize", default=None, help="only aggregate an existing run folder")
    a = ap.parse_args()

    if a.summarize:
        summarize(a.summarize)
        return

    run_dir = a.out or os.path.join(HERE, "results", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    cfg = {"arms": a.arms, "levels": a.levels, "seq_time": a.seq_time, "sched_time": a.sched_time,
           "fixed_time": a.fixed_time, "sched_fetch": a.sched_fetch, "workers": a.workers, "cp_seed": a.cp_seed}
    jobs = []
    for S in a.stations:
        for L in a.lanes:
            for O in a.orders:
                for K in a.skus:
                    for mc in a.movecaps:
                        for seed in a.seeds:
                            params = dict(stations=S, lanes=L, orders=O, skus=K, movecap=mc, seed=seed)
                            path = os.path.join(run_dir, f"inst_S{S}_L{L}_O{O}_K{K}_mc{mc}_seed{seed}.json")
                            if not os.path.exists(path):  # resume
                                jobs.append((params, cfg, path))
    print(f"{len(jobs)} instances -> {run_dir}", flush=True)
    if a.jobs > 1:
        with ProcessPoolExecutor(a.jobs) as ex:
            list(ex.map(_job, jobs))
    else:
        for j in jobs:
            _job(j)
    summarize(run_dir)


if __name__ == "__main__":
    main()
