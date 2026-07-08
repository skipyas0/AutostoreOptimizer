#!/usr/bin/env python3
"""
AutoStore Regret-Based Dynamic Insertion SGC (Best Score Tiebreaker).

This is a wrapper around heuristic_rdi_sgc.py that defaults the tiebreaker
to "best_score" instead of "sum_rt_asc".
"""

from heuristic_rdi_sgc import run_rdi_sgc
import time
from datagen import generate_data
import argparse

from autostore_heuristic import validate_solution


def solve_heuristic_instance(config: dict, return_raw: bool = False):
    """Run one RDI-SGC instance described by *config*, defaulting to best_score."""
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
    
    # Default changed here:
    tiebreaker = config.get("tiebreaker", "best_score")

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

    t0 = time.perf_counter()
    sol = run_rdi_sgc(
        S=S, L=L, K=K, O=O,
        orders_req=orders_req, rt=rt,
        rt_ret=rt_ret, p=p, N=N,
        horizon=horizon, move_cap=move_cap,
        ALPHA=alpha, BETA=beta,
        regret_k=config.get("regret_k", 2),
        use_lazy=config.get("use_lazy", True),
        tiebreaker=tiebreaker
    )
    elapsed = time.perf_counter() - t0
    status = "Feasible" if sol.feasible else "Infeasible"
    violations = validate_solution(
        sol, S, L, K, O, orders_req, rt, rt_ret, p, N,
        horizon=horizon, move_cap=move_cap,
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
    }
    return (res, sol) if return_raw else res


def main() -> None:
    """CLI entry point for running the RDI-SGC heuristic (Best Score) standalone."""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    ap = argparse.ArgumentParser(description="Regret-Based Dynamic Insertion SGC [Best Score Default]")
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
    ap.add_argument("--regret_k", type=int, default=2)
    ap.add_argument("--no_lazy", action="store_true")
    ap.add_argument("--tiebreaker", type=str, choices=["sharing_degree", "best_score", "sum_rt_asc"], default="best_score")
    ap.add_argument("--no_vis", action="store_true")
    args = ap.parse_args()

    print("Generating data...")
    S, L, K, orders_req, rt, p, N = generate_data(
        num_stations=args.stations, lanes_per_station=args.lanes,
        num_orders=args.orders, num_skus=args.skus, seed=args.seed,
        pick_touch_time=args.pick
    )
    rt_ret = dict(rt)
    O = sorted(orders_req.keys())

    print(f"Stations={len(S)}, Lanes={len(L)}, SKUs={len(K)}, Orders={len(O)}, RobotLimit={args.movecap}\n")
    print(f"Running RDI-SGC (regret_k={args.regret_k}, tiebreaker={args.tiebreaker}, lazy={not args.no_lazy})")

    t0 = time.perf_counter()
    sol = run_rdi_sgc(
        S, L, K, O, orders_req, rt, rt_ret, p, N,
        horizon=args.horizon, move_cap=args.movecap,
        ALPHA=args.alpha, BETA=args.beta, regret_k=args.regret_k,
        use_lazy=not args.no_lazy, tiebreaker=args.tiebreaker
    )
    elapsed = time.perf_counter() - t0

    print(f"\n=== RDI-SGC Result ===")
    print(f"Feasible:    {sol.feasible}")
    print(f"Makespan:    {sol.makespan}")
    print(f"Total bin events (moves/2): {sol.total_moves // 2}")
    print(f"Time:        {elapsed:.4f}s")

    violations = validate_solution(
        sol, S, L, K, O, orders_req, rt, rt_ret, p, N,
        horizon=args.horizon, move_cap=args.movecap,
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
                sol, S, L, K, O, orders_req, rt, rt_ret, p,
            )
            plot_schedule(mock_sol, handles)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"[VIS] Skipped: {exc}")


if __name__ == "__main__":
    main()
