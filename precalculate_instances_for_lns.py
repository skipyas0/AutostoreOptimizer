import json
import os
import pickle
import shutil
import time
from contextlib import redirect_stdout
from itertools import product

from autostore_heuristic import build_viz_handles, validate_solution
from cp_model import (
    ProgressCollector,
    build_model,
    extract_and_print_solution,
    inject_warmstart,
    validate_warmstart,
)
from datagen import generate_data
from heuristic_rdi_sgc import run_rdi_sgc
from schedule_visualizer import plot_schedule, write_html


def precalculate_config(config):
    path = f"precalculated_instances/GEN{config['gen_seed']}-CP{config['cp_seed']}-{config['stations']}-{config['lanes']}-{config['skus']}-{config['orders']}-{config['movecap']}"

    os.mkdir(path)

    with open(f"{path}/config.json", "w+") as f:
        json.dump(config, f)

    instance = generate_data(
        num_stations=config["stations"],
        lanes_per_station=config["lanes"],
        num_orders=config["orders"],
        num_skus=config["skus"],
        seed=config["gen_seed"],
        movecap=config["movecap"],
    )

    instance.to_pickle(f"{path}/instance.pkl")

    with open(f"{path}/instance_summary.txt", "w+") as f, redirect_stdout(f):
        instance.print_summary()

    mdl, handles = build_model(
        instance,
        rt_return=instance.rt_ret,
        add_symmetry_breaking=config["symmetry_breaking"],
        horizon=config["horizon"],
        move_cap=config["movecap"],
    )

    t0 = time.perf_counter()
    heur_sol = run_rdi_sgc(
        instance,
        horizon=config["horizon"],
        move_cap=config["movecap"],
        ALPHA=config["alpha"],
        BETA=config["beta"],
    )
    elapsed = time.perf_counter() - t0

    violations = validate_solution(
        heur_sol, instance, horizon=config["horizon"], move_cap=config["movecap"]
    )
    violations.extend(validate_warmstart(heur_sol, heur_sol.pick_events, handles))

    with open(f"{path}/heuristic_solution.pkl", "wb") as f:
        pickle.dump(heur_sol, f)

    with open(f"{path}/heuristic_summary.txt", "w+") as f:
        f.write(
            f"RDI-SGC Heuristic Result | Feasible: {heur_sol.feasible} | Makespan: {heur_sol.makespan} | BinEvents: {heur_sol.total_moves // 2} | Time: {elapsed}"
        )

    mock_sol, viz_handles = build_viz_handles(heur_sol, instance)
    fig = plot_schedule(mock_sol, viz_handles)
    html_file = f"{path}/Heur-RDI_solution.html"
    write_html(fig, html_file)
    print(f"\nWrote visualization to {html_file}")

    sp = inject_warmstart(heur_sol, heur_sol.pick_events, mdl, handles)

    history_listener = ProgressCollector()
    mdl.add_solver_listener(history_listener)

    mdl.set_starting_point(sp)

    with open(f"{path}/cplog.txt", "w+") as cp_log:
        print(f"Solving CP Model with {config['time_limit']}s time limit...")
        sol_cp = mdl.solve(
            TimeLimit=config["time_limit"],
            LogVerbosity="Verbose",
            log_output=cp_log,
            solve_with_search_next=True,
            RandomSeed=config["cp_seed"],
        )
        cp_log.flush()

    if sol_cp:
        sol = sol_cp.get_solution()
        if sol is not None:
            with open(f"{path}/cp_solution.pkl", "wb") as f:
                pickle.dump(sol, f)
        else:
            print("CP sol is None")

        with open(f"{path}/cp_intermediate_records.pkl", "wb") as f:
            pickle.dump(history_listener.records, f)

        print(f"CP Solve Status: {sol_cp.get_solve_status()}")
        with open(f"{path}/cp_sol.txt", "w+") as f, redirect_stdout(f):
            extract_and_print_solution(sol_cp, handles)
        if plot_schedule:
            print("Exporting visualization...")
            fig = plot_schedule(sol_cp, handles)
            html_file = f"{path}/CP-RDI_solution.html"
            write_html(fig, html_file)
            print(f"\nWrote visualization to {html_file}")
    else:
        print("No solution found by CP.")

    os.mkdir(f"{path}/experiments")
    return path


def check_if_precalculated(config, with_removal=False):
    ret = None
    for past in os.listdir("precalculated_instances"):
        with open(f"precalculated_instances/{past}/config.json", "r") as f:
            past_config = json.load(f)
            if past_config == config:
                if "experiments" in os.listdir(f"precalculated_instances/{past}"):
                    print(f"Config {config} is already precalculated in {past}.")
                    ret = f"precalculated_instances/{past}"
                    break
                elif with_removal:
                    if (
                        input("Found non-completed instance, remove and recalculate?")
                        == "yes"
                    ):
                        shutil.rmtree(f"precalculated_instances/{past}")
                    else:
                        raise ValueError(
                            "Incomplete instance but no recalculate on prompt"
                        )
    return ret


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Precalculate instances for LNS (generates heuristic + CP solutions)."
    )
    parser.add_argument(
        "--time-limit", type=int, default=3600, help="CP solver time limit in seconds"
    )
    parser.add_argument(
        "--gen-seed", type=int, default=42, help="Random seed for data generation"
    )
    parser.add_argument(
        "--cp-seed", type=int, default=42, help="Random seed for CP solver"
    )
    parser.add_argument(
        "--stations",
        type=int,
        nargs="+",
        default=[2, 4],
        help="List of station counts to iterate over",
    )
    parser.add_argument(
        "--lanes",
        type=int,
        nargs="+",
        default=[4],
        help="List of lane counts to iterate over",
    )
    parser.add_argument(
        "--skus",
        type=int,
        nargs="+",
        default=[20000],
        help="List of SKU counts to iterate over",
    )
    parser.add_argument(
        "--orders",
        type=int,
        nargs="+",
        default=[100, 200, 300],
        help="List of order counts to iterate over",
    )
    parser.add_argument(
        "--movecaps",
        type=int,
        nargs="+",
        default=[20, 35],
        help="List of move capacities to iterate over",
    )
    args = parser.parse_args()

    TIME_LIMIT = args.time_limit
    GEN_SEED = args.gen_seed
    CP_SEED = args.cp_seed
    STATIONS = args.stations
    LANES = args.lanes
    SKUS = args.skus
    ORDERS = args.orders
    MOVECAPS = args.movecaps

    print(
        f"Precalculating configs: {TIME_LIMIT=} {GEN_SEED=} {CP_SEED=} {STATIONS=} {LANES=} {SKUS=} {ORDERS=} {MOVECAPS=}"
    )

    for s, l, k, o, m in product(STATIONS, LANES, SKUS, ORDERS, MOVECAPS):
        config = {
            "stations": s,
            "lanes": l,
            "orders": o,
            "symmetry_breaking": True,
            "skus": k,
            "movecap": m,
            "gen_seed": GEN_SEED,
            "cp_seed": CP_SEED,
            "horizon": 10000,
            "alpha": 1.0,
            "beta": 1.0,
            "time_limit": TIME_LIMIT,
        }
        print("This config:", config)

        if check_if_precalculated(config, with_removal=True) is not None:
            continue

        precalculate_config(config)


if __name__ == "__main__":
    main()
