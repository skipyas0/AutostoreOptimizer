from schedule_visualizer import plot_schedule
from autostore_heuristic import build_viz_handles, validate_solution
from schedule_visualizer import write_html
from cp_model import build_model, inject_warmstart, extract_and_print_solution, validate_warmstart
from heuristic_rdi_sgc_best_score import run_rdi_sgc
from datagen import generate_data
import sys
import os
import time

# Ensure we can import modules from the project directory
sys.path.insert(0, os.path.abspath('d:/_FEL/CIIRC/Autostore/order_station_assign'))

config = {
    'stations': 2, 'lanes': 4, 'orders': 40, 'pick': 4,
    'timelimit': 200, 'symmetry_breaking': True, 'skus': 20000,
    'movecap': 40, 'seed': 42, 'verbose': True, 'collect_progress': True,
    'horizon': 10000, 'alpha': 1.0, 'beta': 1.0
}


def main():
    print("Generating data...")
    S, L, K, orders_req, rt, p, N = generate_data(
        num_stations=config['stations'],
        lanes_per_station=config['lanes'],
        num_orders=config['orders'],
        num_skus=config['skus'],
        seed=config['seed'],
        pick_touch_time=config['pick']
    )
    O = sorted(orders_req.keys())
    rt_return = dict(rt)
    print(orders_req)

    print("Running RDI-SGC Heuristic...")
    t_heur = time.perf_counter()
    heur_sol = run_rdi_sgc(
        S, L, K, O, orders_req, rt, rt_return, p, N,
        horizon=config['horizon'], move_cap=config['movecap'], ALPHA=config['alpha'], BETA=config['beta']
    )
    print(
        f"Heuristic Time: {time.perf_counter() - t_heur:.2f}s, Feasible: {heur_sol.feasible}, Makespan: {heur_sol.makespan}")

    print("Building CP Model...")
    mdl, handles = build_model(
        S, L, K, orders_req, rt, p, rt_return=rt_return,
        add_symmetry_breaking=config['symmetry_breaking'],
        horizon=config['horizon'], move_cap=config['movecap'], N=N
    )

    print(f"\n=== RDI-SGC Heuristic Result ===")
    print(f"Feasible:    {heur_sol.feasible}")
    print(f"Makespan:    {heur_sol.makespan}")
    print(f"Total bin events (moves/2): {heur_sol.total_moves // 2}")
    print(f"Time:        {time.perf_counter() - t_heur:.4f}s")

    violations = validate_solution(
        heur_sol, S, L, K, O, orders_req, rt, rt_return, p, N,
        horizon=config['horizon'], move_cap=config['movecap'],
    )
    if violations:
        print(f"VALIDATION FAILED ({len(violations)} violations)")
        for v in violations[:10]:
            print(f"  Violation: {v}")
    else:
        print("Validation PASSED")

    try:
        mock_sol, viz_handles = build_viz_handles(
            heur_sol, S, L, K, O, orders_req, rt, rt_return, p,
        )
        plot_schedule(mock_sol, viz_handles)
    except Exception as exc:
        print(f"[VIS] Skipped: {exc}")

    if heur_sol.feasible:
        print("Injecting Warmstart...")
        violations = validate_warmstart(heur_sol, heur_sol.pick_events, handles)
        if violations:
            print(f"Warmstart Violations Found ({len(violations)}):")
            for v in violations[:10]:
                print(f" - {v}")

        try:
            sp = inject_warmstart(heur_sol, heur_sol.pick_events, mdl, handles)
            mdl.set_starting_point(sp)
            print("Successfully injected starting point.")
        except Exception as exc:
            print(f"Failed to inject warmstart: {exc}")

    input("Press Enter to continue to CP solving...")
    print(f"Solving CP Model with {config['timelimit']}s time limit...")
    sol_cp = mdl.solve(
        TimeLimit=config['timelimit'],
        LogVerbosity="Terse"
    )

    if sol_cp:
        print(f"CP Solve Status: {sol_cp.get_solve_status()}")
        extract_and_print_solution(sol_cp, handles)
        if plot_schedule:
            print("Exporting visualization...")
            fig = plot_schedule(sol_cp, handles)
            html_file = "./CP-RDI_solution.html"
            write_html(fig, html_file)
            print(f"\nWrote visualization to {html_file}")
    else:
        print("No solution found by CP.")


if __name__ == '__main__':
    main()
