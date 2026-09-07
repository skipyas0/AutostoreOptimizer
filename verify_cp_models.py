#!/usr/bin/env python3
"""Verification script to prove that Python and C++ CP models are identical.

Builds CP models from both cp_model.py (Python/docplex) and cpp_cp_model.py
(C++/pybind11), exports them to .cpo files, normalizes them via
cp_model_utils.canonicalize_cpo, and diffs them line-by-line.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import cp_model
import cpp_cp_model
from cp_model_utils import canonicalize_cpo, compare_cpo_files
from instance import Instance


def create_test_instances() -> list[dict]:
    """Creates diverse test instance configurations to verify model equivalence

    under different options (symmetry breaking, move cap, auto-horizon, etc.).
    """
    tests = []

    # Test 1: Minimal instance, no symmetry breaking, fixed horizon
    inst1 = Instance(
        S=[0, 1],
        L=[0, 1],
        K=[0, 1],
        orders_requirements={0: [0, 1], 1: [0]},
        rt={0: 10, 1: 20},
        p={0: 4, 1: 4},
        N={0: 1, 1: 1},
        movecap=5,
        seed=42,
    )
    tests.append(
        {
            "name": "minimal_no_symm",
            "instance": inst1,
            "add_symmetry_breaking": False,
            "horizon": 100,
            "description": "2 stations, 2 lanes, 2 SKUs, 2 orders, no symmetry breaking, horizon=100",
        }
    )

    # Test 2: Minimal instance, symmetry breaking enabled, movecap=5
    inst2 = Instance(
        S=[0, 1],
        L=[0, 1],
        K=[0, 1],
        orders_requirements={0: [0, 1], 1: [0]},
        rt={0: 10, 1: 20},
        p={0: 4, 1: 4},
        N={0: 1, 1: 1},
        movecap=5,
        seed=42,
    )
    tests.append(
        {
            "name": "minimal_with_symm",
            "instance": inst2,
            "add_symmetry_breaking": True,
            "horizon": 100,
            "description": "2 stations, 2 lanes, 2 SKUs, 2 orders, with symmetry breaking, horizon=100",
        }
    )

    # Test 3: Auto-horizon computation (horizon=0), movecap=None
    inst3 = Instance(
        S=[0, 1],
        L=[0, 1],
        K=[0, 1],
        orders_requirements={0: [0], 1: [1], 2: [0, 1]},
        rt={0: 15, 1: 25},
        p={0: 5, 1: 6},
        N={0: 1, 1: 2},
        movecap=10,
        seed=123,
    )
    inst3._movecap = None
    tests.append(
        {
            "name": "auto_horizon_no_movecap",
            "instance": inst3,
            "add_symmetry_breaking": False,
            "horizon": 0,
            "description": "2 stations, 2 lanes, 2 SKUs, 3 orders, horizon=0 (auto-computed), movecap=None",
        }
    )


    # Test 4: Multiple SKUs and orders with varying bin counts N[k]
    inst4 = Instance(
        S=[0, 1],
        L=[0, 1, 2],
        K=[0, 1, 2, 3],
        orders_requirements={
            0: [0, 1],
            1: [1, 2],
            2: [2, 3],
            3: [0, 3],
        },
        rt={0: 10, 1: 15, 2: 20, 3: 25},
        p={0: 3, 1: 4, 2: 5, 3: 6},
        N={0: 1, 1: 2, 2: 1, 3: 3},
        movecap=6,
        seed=777,
    )
    tests.append(
        {
            "name": "multi_sku_varying_bins",
            "instance": inst4,
            "add_symmetry_breaking": True,
            "horizon": 150,
            "description": "2 stations, 3 lanes, 4 SKUs, 4 orders, varying N[k], with symmetry breaking",
        }
    )

    # Test 5: 3 stations, 3 lanes, 5 SKUs, custom rt_ret
    inst5 = Instance(
        S=[0, 1, 2],
        L=[0, 1, 2],
        K=[0, 1, 2, 3, 4],
        orders_requirements={
            0: [0, 2],
            1: [1, 3],
            2: [0, 4],
            3: [2, 3, 4],
        },
        rt={0: 8, 1: 12, 2: 16, 3: 20, 4: 24},
        p={0: 4, 1: 4, 2: 4, 3: 4, 4: 4},
        N={0: 1, 1: 1, 2: 2, 3: 2, 4: 1},
        movecap=8,
        seed=999,
        rt_ret={0: 10, 1: 14, 2: 18, 3: 22, 4: 26},
    )
    tests.append(
        {
            "name": "three_stations_custom_rt_ret",
            "instance": inst5,
            "add_symmetry_breaking": True,
            "horizon": 180,
            "description": "3 stations, 3 lanes, 5 SKUs, 4 orders, custom rt_ret, with symmetry breaking",
        }
    )

    return tests


def verify_instance(
    test_config: dict, output_dir: Path, verbose: bool = False
) -> bool:
    """Verifies that Python and C++ CP models produce equivalent CPO models for

    a given instance.
    """
    name = test_config["name"]
    inst = test_config["instance"]
    symm = test_config["add_symmetry_breaking"]
    horizon = test_config["horizon"]
    desc = test_config["description"]

    test_dir = output_dir / name
    test_dir.mkdir(parents=True, exist_ok=True)

    py_raw_path = str(test_dir / "model_python.cpo")
    cpp_raw_path = str(test_dir / "model_cpp.cpo")
    py_canon_path = str(test_dir / "model_python_canonical.cpo")
    cpp_canon_path = str(test_dir / "model_cpp_canonical.cpo")

    # 1. Build Python CP model & export CPO
    py_model, _ = cp_model.build_model(inst, symm, horizon)
    py_model.export_model(py_raw_path)

    # 2. Build C++ CP model & export CPO
    cpp_model_obj, _ = cpp_cp_model.build_model(inst, symm, horizon)
    cpp_model_obj.export_model(cpp_raw_path)

    # 3. Canonicalize and Diff
    identical, diff_lines = compare_cpo_files(
        py_cpo_path=py_raw_path,
        cpp_cpo_path=cpp_raw_path,
        py_canonical_path=py_canon_path,
        cpp_canonical_path=cpp_canon_path,
    )

    # Count statements in canonical model
    with open(py_canon_path) as f:
        py_canon_lines = [
            l.strip()
            for l in f
            if l.strip() and not l.strip().startswith("//")
        ]
    with open(cpp_canon_path) as f:
        cpp_canon_lines = [
            l.strip()
            for l in f
            if l.strip() and not l.strip().startswith("//")
        ]

    num_vars = len([l for l in py_canon_lines if "intervalVar" in l])
    num_constraints = len(
        [
            l
            for l in py_canon_lines
            if "intervalVar" not in l and not l.startswith("minimize")
        ]
    )

    status_str = "MATCH (IDENTICAL)" if identical else "MISMATCH"
    color = "\033[92m" if identical else "\033[91m"
    reset = "\033[0m"

    print(f"\n{'-'*70}")
    print(f"Test: {name}")
    print(f"Config: {desc}")
    print(
        f"Artifacts: {test_dir}/{{model_python_canonical.cpo, model_cpp_canonical.cpo}}"
    )
    print(f"Elements: {num_vars} variables, {num_constraints} constraints, 1 objective")
    print(f"Result: {color}{status_str}{reset} (Diff lines: {len(diff_lines)})")

    if not identical or verbose:
        if diff_lines:
            print(f"\nUnified Diff for {name}:")
            for line in diff_lines[:30]:
                sys.stdout.write(line)
            if len(diff_lines) > 30:
                print(f"... ({len(diff_lines) - 30} more diff lines)")

    return identical


def main():
    parser = argparse.ArgumentParser(
        description="Verify equivalence of Python and C++ CP models via CPO export & diff."
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="cpo_verification_results",
        help="Directory to save exported and canonicalized .cpo files",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose diff and inspection details",
    )
    parser.add_argument(
        "--test",
        "-t",
        type=str,
        default=None,
        help="Name of specific test configuration to run (default: run all)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_tests = create_test_instances()
    if args.test:
        all_tests = [t for t in all_tests if t["name"] == args.test]
        if not all_tests:
            print(f"Error: test '{args.test}' not found in test suite.")
            sys.exit(1)

    print("======================================================================")
    print("CP Model Equivalence Verification: Python (docplex) vs C++ (pybind11)")
    print("======================================================================")
    print(f"Output directory for .cpo artifacts: {out_dir.resolve()}")
    print(f"Running {len(all_tests)} verification scenario(s)...")

    results = {}
    for test in all_tests:
        ok = verify_instance(test, out_dir, verbose=args.verbose)
        results[test["name"]] = ok

    print("\n" + "=" * 70)
    print("SUMMARY OF VERIFICATION RESULTS:")
    print("=" * 70)
    all_passed = True
    for name, passed in results.items():
        status = "PASSED (100% IDENTICAL)" if passed else "FAILED"
        color = "\033[92m" if passed else "\033[91m"
        reset = "\033[0m"
        print(f"  {name:<32}: {color}{status}{reset}")
        if not passed:
            all_passed = False

    print("=" * 70)
    if all_passed:
        print("\033[92mALL TESTS PASSED: Python and C++ CP models are truly equivalent!\033[0m\n")
        sys.exit(0)
    else:
        print("\033[91mSOME TESTS FAILED: Differences detected!\033[0m\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
