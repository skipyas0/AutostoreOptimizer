#!/usr/bin/env python3
"""
Local benchmark runner for AutoStore heuristics.
Executes multiple heuristic modules across the standard parameter sweep locally.
Designed for sub-second heuristics where SLURM scheduling overhead is too high.

Usage:
    python benchmark_heuristic_local.py --modules autostore_heuristic --output-dir results/heuristic_v1
"""

import argparse
import importlib
import json
import os
import time
import sys
import concurrent.futures
from datetime import datetime
from typing import List, Dict, Any

# Add repository root and script directory to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from benchmark_v4_single import PARAM_LEVELS, REFERENCE_CONFIG, SEEDS, PARAM_ORDER, build_config


def enumerate_combinations() -> List[Dict[str, Any]]:
    """
    Create a flat list of all (parameter, value, seed) combos.
    Same as in benchmark_v4_single.py for consistency.
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


def run_single_instance(module_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dynamically import and run a heuristic module.
    Expected interface: module.solve_heuristic_instance(config) -> result_dict
    """
    try:
        mod = importlib.import_module(module_name)
    except ImportError as e:
        return {
            "status": "ImportError",
            "error": str(e),
            "solve_time": 0.0,
            "objective_value": None
        }

    if not hasattr(mod, "solve_heuristic_instance"):
        return {
            "status": "InterfaceError",
            "error": f"Module {module_name} missing solve_heuristic_instance()",
            "solve_time": 0.0,
            "objective_value": None
        }

    try:
        # Heuristics are fast, so we trust them to manage their own time or return quickly
        res = mod.solve_heuristic_instance(config)
        return res
    except Exception as e:
        return {
            "status": "Crash",
            "error": str(e),
            "solve_time": 0.0,
            "objective_value": None
        }


def process_task(task_args: dict) -> dict:
    mod_name = task_args["mod_name"]
    cfg = task_args["cfg"]
    out_path = task_args["out_path"]
    param = task_args["param"]
    val = task_args["val"]
    seed = task_args["seed"]
    overwrite = task_args["overwrite"]

    if os.path.exists(out_path) and not overwrite:
        return {
            "status": "Skipped",
            "mod_name": mod_name,
            "param": param,
            "val": val,
            "seed": seed,
            "solve_time": 0.0,
            "objective_value": None
        }

    res = run_single_instance(mod_name, cfg)

    record = {
        "meta": {
            "mode": "heuristic_local",
            "module": mod_name,
            "parameter": param,
            "value": val,
            "seed": seed,
            "generated_at": datetime.now().isoformat(),
            "reference_config": REFERENCE_CONFIG
        },
        "config": cfg,
        "result": res
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    return {
        "status": res.get("status"),
        "mod_name": mod_name,
        "param": param,
        "val": val,
        "seed": seed,
        "solve_time": res.get("solve_time", 0.0),
        "objective_value": res.get("objective_value", None)
    }


def main():
    parser = argparse.ArgumentParser(description="Local runner for AutoStore heuristics")
    parser.add_argument(
        "--modules",
        nargs="+",
        help="List of python module names to benchmark",
        default=["heuristic_gbs_critical_path", "heuristic_gbs_max_sharing", "heuristic_rdi_sgc_best_score", "heuristic_rdi_sgc_sharing_degree"]
    )
    parser.add_argument(
        "--output-dir",
        default="results/heuristic_local/greedy_and_ama_serial",
        help="Directory to save JSON results"
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Run only one instance per parameter for testing"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing result files"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() - 1 or 4,
        help="Number of parallel workers"
    )

    args = parser.parse_args()

    # 1. Prepare combinations
    combos = enumerate_combinations()
    if args.test_run:
        print("Test run mode: selecting 1 combo per parameter...")
        # Take just one combo per parameter type to verify pipeline
        seen_params = set()
        test_combos = []
        for c in combos:
            if c["parameter"] not in seen_params:
                test_combos.append(c)
                seen_params.add(c["parameter"])
        combos = test_combos

    print(f"Plan: Run {len(combos)} configurations across {len(args.modules)} modules.")
    print(f"Output directory: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Execute
    start_global = time.perf_counter()
    count = 0

    tasks = []
    for combo in combos:
        param = combo["parameter"]
        val = combo["value"]
        seed = combo["seed"]
        cfg = build_config(param, val, seed, timelimit=3600)

        for mod_name in args.modules:
            fname = f"{mod_name}_{param}_val{val}_seed{seed}.json"
            out_path = os.path.join(args.output_dir, fname)

            if os.path.exists(out_path) and not args.overwrite:
                if args.verbose:
                    print(f"Skipping {fname} (exists)")
                continue

            tasks.append({
                "mod_name": mod_name,
                "cfg": cfg,
                "out_path": out_path,
                "param": param,
                "val": val,
                "seed": seed,
                "overwrite": args.overwrite
            })

    total_tasks = len(tasks)
    if total_tasks == 0:
        print("All tasks skipped. Exiting.")
        return

    print(f"Executing {total_tasks} tasks using {args.workers} workers...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {executor.submit(process_task, t): t for t in tasks}

        for future in concurrent.futures.as_completed(future_to_task):
            count += 1
            t = future_to_task[future]
            try:
                res = future.result()
                if args.verbose:
                    print(f"[{count:3d}/{total_tasks}] {t['mod_name']} - {t['param']}={t['val']}, seed={t['seed']} -> {res['status']} ({res['solve_time']:.4f}s), Makespan: {res['objective_value']}")
                else:
                    sys.stdout.write(f"\rProgress: [{count:3d}/{total_tasks}] completed...")
                    sys.stdout.flush()
            except Exception as e:
                print(f"\nTask failed for {t['mod_name']} ({t['param']}={t['val']}, seed={t['seed']}): {e}")

    elapsed = time.perf_counter() - start_global
    print(f"\nBenchmark completed in {elapsed:.2f}s.")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
