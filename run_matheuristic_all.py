#!/usr/bin/env python3
"""
Batch runner for matheuristic.py — runs on all precalculated instance folders.

Usage:
    python run_matheuristic_all.py                          # default args
    python run_matheuristic_all.py --backend ortools         # override backend
    python run_matheuristic_all.py --runs 5 --iters 100      # override other args
    python run_matheuristic_all.py --skip GEN42-2-2-500-50-10  # skip specific folder
"""

import argparse
import os
import subprocess
import sys

PRECALCULATED_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "precalculated_instances"
)
MATHEURISTIC_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "matheuristic.py"
)

# Default CPLEX installation path on the server
DEFAULT_CPLEX_DIR = "/home/kloudvoj/ibm/ILOG/CPLEX_Studio222"


def build_cplex_env(cplex_dir):
    """Build an environment dict with CPLEX library paths set for subprocesses."""
    env = os.environ.copy()

    # Set CPLEX_DIR so docplex can find the installation
    env["CPLEX_DIR"] = cplex_dir

    # Determine architecture subdirectory
    if sys.platform == "darwin":
        cplex_lib_base = os.path.join(cplex_dir, "cplex", "lib")
        if os.path.exists(os.path.join(cplex_lib_base, "arm64_osx")):
            arch_dir = "arm64_osx"
        else:
            arch_dir = "x86-64_osx"
        lib_path_var = "DYLD_LIBRARY_PATH"
    else:
        arch_dir = "x86-64_linux"
        lib_path_var = "LD_LIBRARY_PATH"

    fmt_dir = "static_pic"

    # Collect all CPLEX library directories
    lib_dirs = []
    for sub in ("cplex", "cpoptimizer", "concert"):
        base = os.path.join(cplex_dir, sub, "lib")
        candidate = os.path.join(base, arch_dir, fmt_dir)
        if os.path.exists(candidate):
            lib_dirs.append(candidate)
        else:
            candidate = os.path.join(base, arch_dir)
            if os.path.exists(candidate):
                lib_dirs.append(candidate)
            else:
                lib_dirs.append(base)

    # Prepend to the library path variable
    existing = env.get(lib_path_var, "")
    new_path = ":".join(lib_dirs)
    if existing:
        env[lib_path_var] = f"{new_path}:{existing}"
    else:
        env[lib_path_var] = new_path

    return env


def get_instance_folders():
    """Return sorted list of instance folder names under precalculated_instances/."""
    if not os.path.isdir(PRECALCULATED_DIR):
        print(f"Error: Directory not found: {PRECALCULATED_DIR}", file=sys.stderr)
        sys.exit(1)

    folders = sorted(
        f
        for f in os.listdir(PRECALCULATED_DIR)
        if os.path.isdir(os.path.join(PRECALCULATED_DIR, f)) and not f.startswith(".")
    )
    return folders


def main():
    parser = argparse.ArgumentParser(
        description="Run matheuristic.py on all precalculated instance folders."
    )
    # Arguments forwarded to matheuristic.py
    parser.add_argument(
        "--backend",
        type=str,
        default="docplex",
        help="Optimizer backend (default: docplex)",
    )
    parser.add_argument(
        "--runs", type=int, default=10, help="Number of runs per instance (default: 10)"
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=300,
        help="Number of iterations per run (default: 300)",
    )
    parser.add_argument(
        "--stagnation-th",
        type=int,
        default=50,
        dest="stagnation_th",
        help="Stagnation threshold (default: 50)",
    )
    parser.add_argument(
        "--severity-step",
        type=int,
        default=1,
        dest="severity_step",
        help="Severity increment step (default: 1)",
    )
    parser.add_argument(
        "--max-severity",
        type=int,
        default=5,
        dest="max_severity",
        help="Upper bound for severity (default: 5)",
    )
    parser.add_argument(
        "--iter-time-limit",
        type=float,
        default=2.0,
        dest="iter_time_limit",
        help="Per-iteration time limit in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of workers for CP Optimizer"
    )
    parser.add_argument(
        "--presolve",
        type=str,
        default="Auto",
        choices=["Auto", "On", "Off"],
        help="Presolve setting for CP Optimizer",
    )
    parser.add_argument(
        "--search-type",
        type=str,
        default="Auto",
        choices=["Auto", "DepthFirst", "Restart", "MultiPoint"],
        help="Search type for CP Optimizer",
    )
    parser.add_argument(
        "--delta-freezing",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="delta_freezing",
        help="Use delta freezing (default: True)",
    )
    parser.add_argument(
        "--improvement-constr",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="improvement_constr",
        help="Add improvement constraint (default: False)",
    )
    parser.add_argument(
        "--starting-point-for-frozen",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="starting_point_for_frozen",
        help="Initialize starting point for frozen vars (default: True)",
    )
    parser.add_argument(
        "--eps-greedy-prob",
        type=float,
        default=0.2,
        dest="eps_greedy_prob",
        help="Epsilon-greedy acceptance probability (default: 0.2)",
    )

    # Batch-specific arguments
    parser.add_argument(
        "--cplex-dir",
        type=str,
        default=DEFAULT_CPLEX_DIR,
        help=f"CPLEX installation directory (default: {DEFAULT_CPLEX_DIR})",
    )
    parser.add_argument(
        "--skip",
        type=str,
        nargs="*",
        default=[],
        help="List of instance folder names to skip",
    )
    parser.add_argument(
        "--only",
        type=str,
        nargs="*",
        default=[],
        help="Only run these instance folder names (overrides --skip)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without executing",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run instances sequentially (default: parallel with N-1 workers)",
    )

    args = parser.parse_args()

    instance_folders = get_instance_folders()

    if args.only:
        instance_folders = [f for f in instance_folders if f in args.only]
    elif args.skip:
        instance_folders = [f for f in instance_folders if f not in args.skip]

    if not instance_folders:
        print("No instance folders to process.", file=sys.stderr)
        sys.exit(0)

    # Build the base command for matheuristic.py
    base_cmd = [sys.executable, MATHEURISTIC_SCRIPT]
    forwarded_args = [
        ("--backend", args.backend),
        ("--runs", str(args.runs)),
        ("--iters", str(args.iters)),
        ("--stagnation-th", str(args.stagnation_th)),
        ("--severity-step", str(args.severity_step)),
        ("--max-severity", str(args.max_severity)),
        ("--iter-time-limit", str(args.iter_time_limit)),
        ("--workers", str(args.workers)),
        ("--presolve", args.presolve),
        ("--search-type", args.search_type),
    ]
    # BooleanOptionalAction flags
    if args.delta_freezing:
        base_cmd.append("--delta-freezing")
    else:
        base_cmd.append("--no-delta-freezing")

    if args.improvement_constr:
        base_cmd.append("--improvement-constr")
    else:
        base_cmd.append("--no-improvement-constr")

    if args.starting_point_for_frozen:
        base_cmd.append("--starting-point-for-frozen")
    else:
        base_cmd.append("--no-starting-point-for-frozen")

    base_cmd.extend(f"--eps-greedy-prob {args.eps_greedy_prob}".split())

    print("=" * 72)
    print("Batch Matheuristic Runner")
    print(f"Instances to process ({len(instance_folders)}):")
    for f in instance_folders:
        print(f"  - {f}")
    print(f"Backend: {args.backend}")
    print(f"Runs per instance: {args.runs}, Iters per run: {args.iters}")
    print(f"Iter time limit: {args.iter_time_limit}s")
    print("=" * 72)

    if args.dry_run:
        print("\n[Dry run] Would execute:")
        for folder in instance_folders:
            cmd = base_cmd + [folder]
            print(f"  $ {' '.join(cmd)}")
        sys.exit(0)

    # Build CPLEX environment for subprocesses
    cplex_env = build_cplex_env(args.cplex_dir)

    # Run instances
    if args.sequential:
        for folder in instance_folders:
            cmd = base_cmd + [folder]
            print(f"\n{'─' * 72}")
            print(f"Running: {folder}")
            print(f"{'─' * 72}")
            result = subprocess.run(
                cmd, cwd=os.path.dirname(MATHEURISTIC_SCRIPT), env=cplex_env
            )
            if result.returncode != 0:
                print(
                    f"ERROR: {folder} failed with exit code {result.returncode}",
                    file=sys.stderr,
                )
                print(result.stderr.decode() if result.stderr else "")
    else:
        import concurrent.futures

        max_workers = max(1, os.cpu_count() - 1) if os.cpu_count() else 2
        print(
            f"\nRunning {len(instance_folders)} instances in parallel (max {max_workers} workers)..."
        )

        def run_instance(folder):
            cmd = base_cmd + [folder]
            result = subprocess.run(
                cmd,
                cwd=os.path.dirname(MATHEURISTIC_SCRIPT),
                capture_output=True,
                text=True,
                env=cplex_env,
            )
            return folder, result.returncode, result.stdout, result.stderr

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            fut_to_folder = {
                executor.submit(run_instance, f): f for f in instance_folders
            }
            for future in concurrent.futures.as_completed(fut_to_folder):
                folder = fut_to_folder[future]
                try:
                    folder, rc, stdout, stderr = future.result()
                    status = "OK" if rc == 0 else f"FAILED (rc={rc})"
                    print(f"  [{status}] {folder}")
                    if rc != 0:
                        print(f"    stderr: {stderr[:500]}" if stderr else "")
                except Exception as e:
                    print(f"  [ERROR] {folder}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
