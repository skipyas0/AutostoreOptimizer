#!/usr/bin/env python3
"""
Single-run benchmark for AutoStore — CP model, heuristic, or heuristic warm-start.

Each call runs exactly ONE configuration:
  (parameter, level, seed)
and writes a JSON with solve metadata + full improvement curve.

Modes
-----
  cp         — CP Optimizer model only (v4 or v5, selectable via --model-version)
  heuristic  — SGC heuristic only (selectable via --heuristic-module)
  warmstart  — SGC heuristic used as CpoStartingPoint, then CP Optimizer solves

This script is designed to be launched from a Slurm job array, passing
--task-id = $SLURM_ARRAY_TASK_ID.
"""

import argparse
import importlib
import json
import os
import sys
import time
import traceback
from datetime import datetime
# ----------------------------------------------------------
# Reference configuration (your reference scenario)
# ----------------------------------------------------------
REFERENCE_CONFIG = {
    "stations": 4,
    "lanes": 4,
    "orders": 40,
    "pick": 4,
    "timelimit": 1800,
    "symmetry_breaking": True,
}
REFERENCE_CONFIG["skus"] = REFERENCE_CONFIG["stations"] * 5000
REFERENCE_CONFIG["movecap"] = REFERENCE_CONFIG["skus"] // 1000

# ----------------------------------------------------------
# Parameter levels for scaling
# ----------------------------------------------------------

PARAM_LEVELS = {
    "stations": [1, 2, 4, 6, 8, 10],
    # "stations": [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    "lanes": [1, 2, 4, 6, 8, 10],
    # "lanes": [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    "orders": [10, 20, 40, 60, 80, 90, 100, 120, 140, 160, 180, 200],
    # "orders": [600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 3500, 4000, 4500, 5000],
    # "orders": [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000],
    # "orders": [140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500],
    # "orders": [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200],
    "movecap": [1, 2, 5, 10, 15, 20, 40, 60],
}
SEEDS = [0, 1, 2, 4, 5]

# Explicit order of parameters (so enumeration is deterministic)
PARAM_ORDER = ["stations", "lanes", "orders", "movecap"]


def enumerate_combinations():
    """
    Create a flat, deterministic list of all (parameter, value, seed) combos.

    Returns a list of dicts:
        [
          {"parameter": "stations", "value": 1, "seed": 0},
          {"parameter": "stations", "value": 1, "seed": 1},
          ...
        ]
    """
    combos = []
    for param in PARAM_ORDER:
        levels = PARAM_LEVELS[param]
        for val in levels:
            for seed in SEEDS:
                combos.append({
                    "parameter": param,
                    "value": val,
                    "seed": seed,
                })
    return combos


def build_config(parameter: str, level: int, seed: int, timelimit: int) -> dict:
    """
    Build a solve_instance() config for a single run, based on the reference.
    """
    cfg = dict(REFERENCE_CONFIG)
    cfg["timelimit"] = timelimit
    cfg["seed"] = seed
    cfg["symmetry_breaking"] = True
    cfg["verbose"] = True
    cfg["collect_progress"] = True

    if parameter == "stations":
        cfg["stations"] = level
        # cfg["skus"] = 3 * cfg["orders"]
        cfg["skus"] = level * 5000
        cfg["movecap"] = max(10, int(cfg["skus"] / 1250))
    elif parameter == "lanes":
        cfg["lanes"] = level
        # cfg["skus"] = 3 * cfg["orders"]
        # cfg["movecap"] = 2 * cfg["orders"]
    elif parameter == "orders":
        cfg["orders"] = level
        # cfg["skus"] = 3 * level  # your "3×orders" rule
        # cfg["movecap"] = 2 * level
    elif parameter == "movecap":
        cfg["movecap"] = level
        # cfg["skus"] = 3 * cfg["orders"]
    else:
        raise ValueError(f"Unsupported parameter: {parameter}")

    return cfg


def _run_warmstart(cfg: dict, heuristic_module: str) -> dict:
    """Generate data once, run heuristic, inject as CpoStartingPoint, solve CP.

    Returns a result dict with the standard CP keys plus a ``"heuristic"``
    sub-dict containing the heuristic's solve_time and objective_value.
    """
    from datagen import generate_data
    from cp_model import (
        build_model, inject_warmstart, ProgressCollector, validate_warmstart,
    )

    heur = importlib.import_module(heuristic_module)

    num_stations = cfg.get("stations", 1)
    lanes_per_station = cfg.get("lanes", 2)
    num_orders = cfg.get("orders", 7)
    num_skus = cfg.get("skus", 5)
    seed = cfg.get("seed", 42)
    pick_touch_time = cfg.get("pick", 4)
    timelimit = cfg.get("timelimit", 3600)
    add_symmetry_breaking = cfg.get("symmetry_breaking", True)
    horizon = cfg.get("horizon", 10000)
    move_cap = cfg.get("movecap", None)

    # 1. Generate data (single call shared by heuristic and CP)
    S, L, K, orders_req, rt, p, N = generate_data(
        num_stations=num_stations,
        lanes_per_station=lanes_per_station,
        num_orders=num_orders,
        num_skus=num_skus,
        seed=seed,
        pick_touch_time=pick_touch_time,
    )
    rt_ret = dict(rt)
    O = sorted(orders_req.keys())

    # 2. Run heuristic
    # We call solve_heuristic_instance so that each heuristic can handle its own hyperparameters.
    if hasattr(heur, "solve_heuristic_instance"):
        try:
            heur_summary, heur_sol = heur.solve_heuristic_instance(cfg, return_raw=True)
        except TypeError:
            # Fallback if return_raw is not implemented (though we added it to all)
            raise RuntimeError("solve_heuristic_instance must support return_raw=True for warmstart.")
    else:
        # Fallback to run_sgc
        t_heur = time.perf_counter()
        heur_sol = heur.run_sgc(
            S, L, K, O, orders_req, rt, rt_ret, p, N,
            horizon=horizon, move_cap=move_cap,
        )
        heur_time = time.perf_counter() - t_heur
        heur_summary = {
            "status": "Feasible" if heur_sol.feasible else "Infeasible",
            "solve_time": heur_time,
            "objective_value": float(heur_sol.makespan) if heur_sol.feasible else None,
        }

    # 3. Build CP model
    num_vars = 0
    try:
        mdl, handles = build_model(
            S, L, K, orders_req, rt, p,
            rt_return=rt_ret,
            add_symmetry_breaking=add_symmetry_breaking,
            horizon=horizon,
            move_cap=move_cap,
            N=N,
        )
        num_vars = len(mdl.get_all_variables())
    except Exception as exc:
        print(f"Model build failed: {exc}")
        print(sys.exc_info())
        traceback.print_exc()

        return {
            "status": "ModelBuildError",
            "solve_time": 0.0,
            "objective_value": None,
            "num_vars": num_vars,
            "progress": [],
            "heuristic": heur_summary,
        }

    # 4. Inject warm start (only if heuristic succeeded)
    collector = ProgressCollector()
    mdl.add_solver_listener(collector)
    solve_kwargs = dict(
        Workers=1,
        TimeLimit=timelimit,
        LogVerbosity="Terse",
        solve_with_search_next=True,
    )
    if heur_sol.feasible:
        # Debug: Validate heuristic consistency with CP constraints
        violations = validate_warmstart(heur_sol, heur_sol.pick_events, handles)
        if violations:
            print(f"Warmstart Violations Found ({len(violations)}):")
            for v in violations[:10]:
                print(f" - {v}")
            if len(violations) > 10:
                print(" ... and more.")

        try:
            sp = inject_warmstart(heur_sol, heur_sol.pick_events, mdl, handles)
            mdl.set_starting_point(sp)
            print(f"Warm start injected (heuristic makespan={heur_sol.makespan})")
        except Exception as exc:
            traceback.print_exc()
            print(f"inject_warmstart failed ({exc}); falling back to cold start.")
            # exit(211)

    # 5. Solve CP
    try:
        cp_sol = mdl.solve(**solve_kwargs)
        status = cp_sol.get_solve_status()
        solve_time = cp_sol.get_solve_time()
        obj_val = None
        if status in ("Optimal", "Feasible"):
            obj_val = cp_sol.get_objective_values()[0]
        elif status == "Unknown":
            status = "TimeLimit"
        return {
            "status": status,
            "solve_time": solve_time,
            "objective_value": obj_val,
            "num_vars": num_vars,
            "progress": collector.records,
            "heuristic": heur_summary,
        }
    except Exception as exc:
        print(f"Solver crashed: {exc}")
        return {
            "status": "Crash",
            "solve_time": 0.0,
            "objective_value": None,
            "num_vars": num_vars,
            "progress": collector.records,
            "heuristic": heur_summary,
        }


def main():
    ap = argparse.ArgumentParser(
        description="Single-run AutoStore benchmark (CP, heuristic, or warmstart).")
    ap.add_argument(
        "--task-id",
        type=int,
        help="Index of the combination to run (usually SLURM_ARRAY_TASK_ID).",
    )
    ap.add_argument(
        "--timelimit",
        type=int,
        default=3600,
        help="Time limit per run (seconds). Ignored for pure heuristic mode.",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default="logs/v4_scaling_single",
        help="Directory to store JSON result files.",
    )
    ap.add_argument(
        "--print-count",
        action="store_true",
        help="Print number of combinations and exit (for sanity-checking array size).",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute and overwrite existing JSON file for this combo.",
    )
    ap.add_argument(
        "--model-version",
        type=str,
        default="v5",
        help="CP model version to use: v4 or v5 (only relevant for cp/warmstart modes).",
    )
    ap.add_argument(
        "--mode",
        type=str,
        default="cp",
        choices=["cp", "heuristic", "warmstart"],
        help=(
            "Benchmark mode: "
            "'cp' — CP model only; "
            "'heuristic' — SGC heuristic only; "
            "'warmstart' — heuristic as CpoStartingPoint then CP."
        ),
    )
    ap.add_argument(
        "--heuristic-module",
        type=str,
        default="autostore_heuristic",
        help="Python module name for the heuristic (used in heuristic/warmstart modes).",
    )

    args = ap.parse_args()

    combos = enumerate_combinations()
    total = len(combos)

    if args.print_count:
        print(f"Total combinations: {total}")
        for idx, combo in enumerate(combos):
            print(idx, combo["parameter"], combo["value"], combo["seed"])
        return

    if args.task_id is None:
        raise SystemExit("You must pass --task-id (e.g., from SLURM_ARRAY_TASK_ID).")

    if args.task_id < 0 or args.task_id >= total:
        raise SystemExit(
            f"task-id {args.task_id} is out of range [0, {total - 1}]. "
            "Adjust --array in your Slurm script."
        )

    combo = combos[args.task_id]
    parameter = combo["parameter"]
    value = combo["value"]
    seed = combo["seed"]

    # Build config for this one run
    cfg = build_config(parameter, value, seed, args.timelimit)

    # Add some context from Slurm if present
    slurm_array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    slurm_array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    slurm_job_id = os.environ.get("SLURM_JOB_ID")

    fname = f"{args.mode}_{parameter}_val{value}_seed{seed}.json"
    out_path = os.path.join(args.output_dir, fname)

    # Skip if file already exists and overwrite not requested
    if os.path.exists(out_path) and not args.overwrite:
        print(
            f"Skipping combo (mode={args.mode}, param={parameter}, value={value}, seed={seed}) "
            f"because {out_path} already exists. Use --overwrite to recompute."
        )
        return

    # ----------------------------------------------------------------
    # Dispatch based on mode
    # ----------------------------------------------------------------
    if args.mode == "cp":
        if args.model_version == "v4":
            from order_station_assign_v4_benchmarkable import solve_instance
            print("Running Single-SKU v4 CP model")
        elif args.model_version == "v5":
            from cp_model import solve_instance
            print("Running Multi-SKU v5 CP model")
        else:
            raise ValueError(f"Unsupported model version: {args.model_version}")
        res = solve_instance(cfg)

    elif args.mode == "heuristic":
        heur = importlib.import_module(args.heuristic_module)
        print(f"Running heuristic: {args.heuristic_module}")
        res = heur.solve_heuristic_instance(cfg)

    elif args.mode == "warmstart":
        print(f"Running warmstart: {args.heuristic_module} + v5 CP model")
        res = _run_warmstart(cfg, args.heuristic_module)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # ----------------------------------------------------------------
    # Build and write JSON record
    # ----------------------------------------------------------------
    record = {
        "meta": {
            "mode": args.mode,
            "heuristic_module": args.heuristic_module if args.mode != "cp" else None,
            "model_version": args.model_version if args.mode != "heuristic" else None,
            "parameter": parameter,
            "value": value,
            "seed": seed,
            "reference_config": REFERENCE_CONFIG,
            "timelimit": args.timelimit,
            "task_id": args.task_id,
            "slurm_array_job_id": slurm_array_job_id,
            "slurm_array_task_id": slurm_array_task_id,
            "slurm_job_id": slurm_job_id,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "config": cfg,
        "result": {
            "status": res.get("status"),
            "solve_time": res.get("solve_time"),
            "objective_value": res.get("objective_value"),
            "num_vars": res.get("num_vars"),
            "total_moves": res.get("total_moves"),  # heuristic only; None for CP
        },
        "progress": res.get("progress", []),
    }
    # warmstart: attach heuristic sub-result
    if args.mode == "warmstart" and "heuristic" in res:
        record["heuristic"] = res["heuristic"]

    os.makedirs(args.output_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    print(f"Saved result to {out_path}")


if __name__ == "__main__":
    main()
