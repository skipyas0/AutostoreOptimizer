#!/usr/bin/env python3
"""
AutoStore Adaptive Multi-Attribute SGC (AMA-SGC) 2-Phase Heuristic.

This is a wrapper around heuristic_ama_sgc.py that defaults the mode
to "two_phase" (28 runs) instead of "full_grid" (192 runs).
"""

from heuristic_ama_sgc import run_ama_sgc
import time
from instance import Instance
from datagen import generate_data
import argparse

from autostore_heuristic import validate_solution


def solve_heuristic_instance(config: dict, return_raw: bool = False):
    """Run one AMA-SGC instance described by *config*, defaulting to two_phase.

    Accepts the same config keys as the base SGC's ``solve_heuristic_instance``
    plus optional 'mode' (which defaults to 'two_phase' here).
    Returns a dict with the same top-level keys so benchmark scripts can
    treat both implementations interchangeably.
    """

    num_stations = config.get("stations", 1)
    lanes_per_station = config.get("lanes", 2)
    num_orders = config.get("orders", 7)
    num_skus = config.get("skus", 5)
    seed = config.get("seed", 42)
    pick_touch_time = config.get("pick", 4)
    horizon = config.get("horizon", 10000)
    move_cap = config.get("movecap", None)
    alpha = config.get("alpha", 1.0)
    beta = config.get("beta", 0.0)
    # The default changes here:
    mode = config.get("mode", "two_phase")

    instance = generate_data(
        num_stations=num_stations,
        lanes_per_station=lanes_per_station,
        num_orders=num_orders,
        num_skus=num_skus,
        seed=seed,
        pick_touch_time=pick_touch_time,
    )
    S, L, K, orders_req, rt, p, N = instance
    O = instance.O

    t0 = time.perf_counter()
    sol, best_config, all_runs = run_ama_sgc(
        instance,
        horizon=horizon, move_cap=move_cap, ALPHA=alpha, BETA=beta,
        mode=mode,
    )
    elapsed = time.perf_counter() - t0
    status = "Feasible" if sol.feasible else "Infeasible"
    violations = validate_solution(
        sol, instance, horizon=horizon, move_cap=move_cap
    )
    if violations:
        print(f"VALIDATION FAILED ({len(violations)} violations)")
        status = "Invalid"
        for v in violations[:10]:
            print(f"  Violation: {v}")
    else:
        print("Validation PASSED")

    res = {
        "status": status,
        "solve_time": elapsed,
        "objective_value": float(sol.makespan) if sol.feasible else None,
        "num_vars": 0,
        "progress": [],
        "total_moves": sol.total_moves,
        "winning_config": best_config,
        "all_runs": all_runs,
    }
    return (res, sol) if return_raw else res


def main() -> None:
    """CLI entry point for running the AMA-SGC 2-Phase heuristic standalone."""

    ap = argparse.ArgumentParser(
        description="AMA-SGC Heuristic for AutoStore Task B (v5) [2-Phase Default]"
    )
    ap.add_argument("--stations", type=int, default=4)
    ap.add_argument("--lanes", type=int, default=4)
    ap.add_argument("--orders", type=int, default=160)
    ap.add_argument("--skus", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pick", type=int, default=4)
    ap.add_argument("--movecap", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument(
        "--mode", choices=["full_grid", "two_phase"], default="two_phase",
        help="full_grid: 192 runs; two_phase: 28 runs (faster, may miss interactions)",
    )
    ap.add_argument("--verbose", action="store_true", help="Print winning attribute combination")
    ap.add_argument("--no_vis", action="store_true", help="Skip HTML schedule visualisation")
    args = ap.parse_args()

    print("Generating data...")
    instance = generate_data(
        num_stations=args.stations,
        lanes_per_station=args.lanes,
        num_orders=args.orders,
        num_skus=args.skus,
        seed=args.seed,
        pick_touch_time=args.pick,
    )
    S, L, K, orders_req, rt, p, N = instance
    O = instance.O

    print(f"Stations={len(S)}, Lanes={len(L)}, SKUs={len(K)}, Orders={len(O)}, RobotLimit={args.movecap}\n")

    print(f"\nRunning AMA-SGC heuristic (alpha={args.alpha}, beta={args.beta}, mode={args.mode})...")
    t0 = time.perf_counter()
    sol, best_config, all_runs = run_ama_sgc(
        instance,
        horizon=args.horizon, move_cap=args.movecap,
        ALPHA=args.alpha, BETA=args.beta,
        mode=args.mode, verbose=True,
    )
    elapsed = time.perf_counter() - t0

    oa, od, ba, bd = best_config
    def dir_str(d): return "desc" if d else "asc"
    print(f"\n=== AMA-SGC Result ===")
    print(f"Feasible:    {sol.feasible}")
    print(f"Makespan:    {sol.makespan}")
    print(f"Total bin events (moves/2): {sol.total_moves // 2}")
    print(f"Time:        {elapsed:.4f}s")
    print(f"Winning:     order={oa}({dir_str(od)}), bin={ba}({dir_str(bd)})")

    violations = validate_solution(
        sol, instance, horizon=args.horizon, move_cap=args.movecap
    )
    if violations:
        print(f"VALIDATION FAILED ({len(violations)} violations)")
        for v in violations[:10]:
            print(f"  Violation: {v}")
    else:
        print("Validation PASSED")
    
    if not args.no_vis:
        try:
            from schedule_visualizer import plot_schedule
            from autostore_heuristic import build_viz_handles
            mock_sol, handles = build_viz_handles(
                sol, instance
            )
            plot_schedule(mock_sol, handles)
        except Exception as exc:
            print(f"[VIS] Skipped: {exc}")


if __name__ == "__main__":
    main()
