from datagen import generate_data
from cp_model import build_model, inject_warmstart, extract_and_print_solution
from heuristic_ama_sgc import run_ama_sgc
import sys
import os

config = {
    'stations': 4, 'lanes': 4, 'orders': 40, 'pick': 4,
    'timelimit': 3600, 'symmetry_breaking': True, 'skus': 20000,
    'movecap': 1, 'seed': 0, 'verbose': True, 'collect_progress': True
}


try:
    from schedule_visualizer import plot_schedule
except ImportError:
    plot_schedule = None


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

    print("Running AMA-SGC Heuristic...")
    sol_heur, best_config, all_runs = run_ama_sgc(
        S, L, K, O, orders_req, rt, rt_return, p, N,
        horizon=20000, move_cap=config['movecap'],
        mode="two_phase", verbose=True
    )
    print(f"Heuristic Makespan: {sol_heur.makespan}")

    print("Building CP Model...")
    mdl, handles = build_model(
        S, L, K, orders_req, rt, p, rt_return=rt_return,
        add_symmetry_breaking=config['symmetry_breaking'],
        horizon=20000, move_cap=config['movecap'], N=N
    )

    print("Injecting Warmstart...")
    sp = inject_warmstart(sol_heur, sol_heur.pick_events, mdl, handles)

    print("Solving CP Model...")
    sol_cp = mdl.solve(
        TimeLimit=config['timelimit'],
        StartingPoint=sp,
        LogVerbosity="Normal"
    )

    if sol_cp:
        print(f"CP Solve Status: {sol_cp.get_solve_status()}")
        extract_and_print_solution(sol_cp, handles)
        if plot_schedule:
            print("Exporting visualization...")
            plot_schedule(sol_cp, handles)
    else:
        print("No solution found by CP.")


if __name__ == '__main__':
    main()
