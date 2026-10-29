"""How the per-iteration overhead of the two LNS model variants scales with instance size.

Overhead = everything in an LNS iteration except the CP search and the strategy:
  * full model:  freeze-constraint generation/update + API + CP extraction of the
                 full (partially frozen) model
  * subproblem:  context extraction + reduced-model build + API + CP extraction of
                 the reduced model + merge back into the full solution
Setup time (model build + warm start before the first iteration) is recorded too.

Part A ("lns"):  real LNS runs of both variants on an experiment-1-style size sweep
                 (C++ backend, experiment-1 strategy settings, search-only limit).
Part B ("nbh"):  subproblem only, random neighbourhoods of fixed order fractions,
                 to find the neighbourhood size at which its overhead reaches the
                 full model's extraction cost.

Usage:
  python benchmark_lns_scaling.py                       # both parts, default sweep
  python benchmark_lns_scaling.py --parts lns --sizes 1x25,5x150
  python analyze_lns_scaling.py results/lns_scaling     # tables + figures
"""

import argparse
import json
import os
import pickle
import random
import re
import time

import pandas as pd
from loguru import logger

DEFAULT_SIZES = "1x25,2x50,3x100,5x150,5x300,8x300,10x400,15x600,20x800"
DEFAULT_FRACTIONS = "0.05,0.1,0.2,0.4,0.7,1.0"

# Instance generator settings of experiment 1 (only stations/orders vary)
INSTANCE_BASE = {
    "lanes": 3, "skus": 28750, "movecap": 40, "symmetry_breaking": True,
    "heuristic": "load_balanced_greedy", "skew": 1.0, "rt_model": "depth_based",
    "pick_model": "variable", "pick_touch_time": 4, "return_model": "balancing",
    "bin_model": "popularity_correlated", "max_bins": 8, "grid_depth": 16,
    "horizon": 10000, "alpha": 1.0, "beta": 1.0, "cp_seeds": [], "workers": 1,
}

# LNS settings of experiment 1, except the iteration limit (search-only, short:
# the search itself is not what is measured here)
LNS_BASE = {
    "runs": 1, "stagnation_th": 9999, "severity_step": 1, "severity_backup": "none",
    "max_severity": 5, "iter_time_mode": "search", "delta_freezing": True,
    "starting_point_for_frozen": True, "rrt_threshold": 0.05, "time_limit": None,
    "backend": "cpp", "improvement_constr": True, "presolve": "On", "workers": 1,
    "search_type": "Auto", "strat_choice": "uniform", "strat_preset": "all",
    "default_timeslice_width": 100, "default_n_lanes": 2, "default_percent": 0.1,
    "cp_objective": "makespan", "cp_objective_phase_duration": 0,
    "cp_objective_phase_schedule": "m", "render_video": False, "only_first_sol": False,
    "validate_merges": False,
}


def make_instance(root, stations, orders, seed):
    """Instance folder in the precalculated layout (no CP baselines). Reused if present."""
    from datagen import generate_data
    from precalculate_instances_for_lns import solve_warmstart_heuristic

    cfg = {**INSTANCE_BASE, "stations": stations, "orders": orders, "gen_seed": seed}
    name = f"GEN{seed}-{stations}-{cfg['lanes']}-{cfg['skus']}-{orders}-{cfg['movecap']}"
    path = os.path.join(root, name)
    if os.path.exists(os.path.join(path, "heuristic_solution.pkl")):
        return path
    os.makedirs(os.path.join(path, "experiments"), exist_ok=True)
    instance = generate_data(
        num_stations=stations, lanes_per_station=cfg["lanes"], num_orders=orders,
        num_skus=cfg["skus"], seed=seed, movecap=cfg["movecap"],
        pick_touch_time=cfg["pick_touch_time"], return_model=cfg["return_model"],
        max_bins_per_sku=cfg["max_bins"], sku_popularity_skew=cfg["skew"],
        retrieval_time_model=cfg["rt_model"], pick_time_model=cfg["pick_model"],
        bin_count_model=cfg["bin_model"], grid_depth=cfg["grid_depth"],
    )
    instance.to_pickle(os.path.join(path, "instance.pkl"))
    heur, _ = solve_warmstart_heuristic(cfg["heuristic"], instance, cfg)
    with open(os.path.join(path, "heuristic_solution.pkl"), "wb") as f:
        pickle.dump(heur, f)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(cfg, f)
    return path


def _setup_time(run_dir):
    with open(os.path.join(run_dir, "run_1.log")) as f:
        m = re.search(r"Setup time before first iteration: ([\d.]+)s", f.read())
    return float(m.group(1)) if m else float("nan")


def run_lns(path, lns_model, iters, search_limit):
    """One LNS run; returns (per-iteration DataFrame, run-level dict)."""
    from matheuristic import Solver

    cfg = {**LNS_BASE, "iters": iters, "iter_time_limit": search_limit,
           "lns_model": lns_model}
    t0 = time.perf_counter()
    solver = Solver(cfg, instance_path=path)
    wall = time.perf_counter() - t0
    run_dir = solver.vlg.path
    df = pd.read_pickle(os.path.join(run_dir, "experiment_dataframe.pkl"))
    df = df[df["total_iter_time"] > 0].copy()
    df["n_unfrozen_orders"] = df["unfrozen_orders"].apply(
        lambda s: len(s) if hasattr(s, "__len__") else -1)
    df = df.drop(columns=["unfrozen_orders", "unfrozen_skus", "discovered_solutions"],
                 errors="ignore")
    run = {
        "setup_time": _setup_time(run_dir), "wall_time": wall,
        "num_variables": solver.vlg.num_variables, "best": solver.best_result,
        "run_dir": run_dir,
    }
    return df, run


def neighbourhood_sweep(path, fractions, reps, search_limit, seed):
    """Subproblem overhead per stage for random order neighbourhoods of given sizes."""
    import subproblem
    from instance import Instance  # noqa: F401  (unpickling)
    from matheuristic import prepare_model_subproblem

    with open(os.path.join(path, "instance.pkl"), "rb") as f:
        inst = pickle.load(f)
    with open(os.path.join(path, "heuristic_solution.pkl"), "rb") as f:
        heur = pickle.load(f)
    with open(os.path.join(path, "config.json")) as f:
        horizon = json.load(f)["horizon"]
    t0 = time.perf_counter()
    _, handles, sol = prepare_model_subproblem({"horizon": horizon}, inst, heur)
    setup = time.perf_counter() - t0

    cfg = {"iter_time_limit": search_limit, "iter_time_mode": "search", "workers": 1,
           "presolve": "On", "search_type": "Auto"}
    rng = random.Random(seed)
    O = list(handles["O"])
    rows = []
    for frac in fractions:
        for rep in range(reps):
            active = rng.sample(O, max(1, round(frac * len(O))))
            t0 = time.perf_counter()
            ctx = subproblem.extract_context(sol, handles, inst, active)
            t1 = time.perf_counter()
            mdl, sub_handles = subproblem.build_submodel(ctx, "cpp", horizon, "makespan")
            t2 = time.perf_counter()
            raw, _ = subproblem.solve_submodel(mdl, sub_handles, "cpp", "makespan", -1, cfg)
            t3 = time.perf_counter()
            subproblem.to_full_result(sol, ctx, raw, sub_handles, handles, "cpp", "makespan")
            t4 = time.perf_counter()
            total, extraction, search = subproblem._timings(raw, "cpp")
            rows.append({
                "fraction": frac, "rep": rep, "n_active": len(active),
                "n_shared_picks": len(ctx.shared_picks),
                "t_context": t1 - t0, "t_build": t2 - t1,
                "t_extraction": extraction, "t_search": search,
                "t_api": max(0.0, (t3 - t2) - total), "t_merge": t4 - t3,
                "status": raw.get_solve_status(), "sub_setup_time": setup,
            })
            logger.info(f"  frac {frac:.2f} rep {rep}: build {t2 - t0:.2f}s, "
                        f"extraction {extraction:.2f}s, merge {t4 - t3:.2f}s")
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", default=DEFAULT_SIZES,
                   help=f"Comma-separated STATIONSxORDERS (default: {DEFAULT_SIZES})")
    p.add_argument("--seeds", default="1", help="Instance generator seeds (default: 1)")
    p.add_argument("--parts", default="lns,nbh", help="lns and/or nbh (default: both)")
    p.add_argument("--iters", type=int, default=12, help="LNS iterations per run (default: 12)")
    p.add_argument("--search-limit", type=float, default=1.0,
                   help="Search-only CP limit per iteration in s (default: 1.0)")
    p.add_argument("--fractions", default=DEFAULT_FRACTIONS,
                   help=f"Part B neighbourhood order fractions (default: {DEFAULT_FRACTIONS})")
    p.add_argument("--reps", type=int, default=3, help="Part B samples per fraction (default: 3)")
    p.add_argument("--out", default="results/lns_scaling", help="Output directory")
    args = p.parse_args()

    sizes = [tuple(int(x) for x in s.split("x")) for s in args.sizes.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    parts = set(args.parts.split(","))
    fractions = [float(f) for f in args.fractions.split(",")]
    inst_root = os.path.join(args.out, "instances")
    os.makedirs(inst_root, exist_ok=True)
    with open(os.path.join(args.out, "benchmark_config.json"), "w") as f:
        json.dump({**vars(args), "instance_base": INSTANCE_BASE, "lns_base": LNS_BASE}, f,
                  indent=2)

    iters_csv = os.path.join(args.out, "lns_iterations.csv")
    runs_csv = os.path.join(args.out, "lns_runs.csv")
    nbh_csv = os.path.join(args.out, "nbh_sweep.csv")

    for seed in seeds:
        for stations, orders in sizes:
            logger.info(f"=== {stations} stations / {orders} orders, seed {seed} ===")
            path = make_instance(inst_root, stations, orders, seed)
            meta = {"stations": stations, "orders": orders, "seed": seed,
                    "instance": os.path.basename(path)}
            if "lns" in parts:
                for lns_model in ("full", "subproblem"):
                    df, run = run_lns(path, lns_model, args.iters, args.search_limit)
                    for k, v in meta.items():
                        df[k] = v
                    df["lns_model"] = lns_model
                    df["num_variables"] = run["num_variables"]
                    df.to_csv(iters_csv, mode="a", index=False,
                              header=not os.path.exists(iters_csv))
                    pd.DataFrame([{**meta, "lns_model": lns_model, **run}]).to_csv(
                        runs_csv, mode="a", index=False, header=not os.path.exists(runs_csv))
            if "nbh" in parts:
                df = neighbourhood_sweep(path, fractions, args.reps, args.search_limit, seed)
                for k, v in meta.items():
                    df[k] = v
                df.to_csv(nbh_csv, mode="a", index=False, header=not os.path.exists(nbh_csv))


if __name__ == "__main__":
    main()
