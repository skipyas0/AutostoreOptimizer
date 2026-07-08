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
        print(f"Running {module_name} with config: {config}")
        res = mod.solve_heuristic_instance(config)
        return res
    except Exception as e:
        return {
            "status": "Crash",
            "error": str(e),
            "solve_time": 0.0,
            "objective_value": None
        }


def main():
    parser = argparse.ArgumentParser(description="Local runner for AutoStore heuristics")
    parser.add_argument(
        "--modules",
        nargs="+",
        help="List of python module names to benchmark",
        default=["autostore_heuristic", "heuristic_ama_sgc",
                 "heuristic_ama_sgc_2phase", "heuristic_cfss_sgc", "heuristic_rdi_sgc", "heuristic_rdi_sgc_sharing_degree", "heuristic_rdi_sgc_best_score", "heuristic_gbs", "heuristic_gbs_critical_path", "heuristic_gbs_max_sharing"],
    )
    parser.add_argument(
        "--output-dir",
        default="./results/heuristic_local/full_final",
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
        "--no_verbose",
        action="store_false",
        help="Print detailed progress"
    )

    args = parser.parse_args()
    verbose = args.no_verbose

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
    total_tasks = len(combos) * len(args.modules)

    for i, combo in enumerate(combos):
        param = combo["parameter"]
        val = combo["value"]
        seed = combo["seed"]

        # Build config just like the benchmarks
        # timelimit is dummy for heuristics usually, but we pass it for consistency
        cfg = build_config(param, val, seed, timelimit=3600)

        for mod_name in args.modules:
            # File naming: {module}_{param}_val{value}_seed{seed}.json
            # This allows multiple heuristics to coexist in the same folder if needed
            fname = f"{mod_name}_{param}_val{val}_seed{seed}.json"
            out_path = os.path.join(args.output_dir, fname)

            if os.path.exists(out_path) and not args.overwrite:
                if verbose:
                    print(f"Skipping {fname} (exists)")
                count += 1
                continue

            print(f"Running {mod_name} | {param}={val} seed={seed} ... {out_path=}")

            # Run
            res = run_single_instance(mod_name, cfg)

            # Save
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

            if verbose:
                print(f" Done ({res.get('solve_time', 0):.4f}s) -> {res.get('status')}, Makespan: {res.get('objective_value')}")

            count += 1
            if count % 10 == 0:
                print(f"Progress: {count}/{total_tasks} completed...")

    elapsed = time.perf_counter() - start_global
    print(f"\nBenchmark completed in {elapsed:.2f}s.")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
