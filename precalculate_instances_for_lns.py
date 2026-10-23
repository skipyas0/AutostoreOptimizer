import datetime as dt
import json
import os
import pickle
import shutil
import time
from contextlib import redirect_stdout
from itertools import product

# Suppress matplotlib font cache warnings
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_config")

from cp_model import extract_and_print_solution
import cpp_cp_model
from datagen import generate_data
from heuristics import HEURISTICS, resolve, run_heuristic, validate_solution, validate_warmstart
from schedule_visualizer import plot_schedule, write_html


# Labels written to heuristic_summary.txt (kept from the pre-refactor version).
HEURISTIC_LABELS = {
    "lb_sgc": "Load-Balanced SGC (Greedy)",
    "sgc": "Standard SGC (Greedy)",
    "rdi": "RDI-SGC",
}


def solve_warmstart_heuristic(heuristic_name: str, instance, config: dict):
    """Solve the instance with the requested warm-start heuristic. Returns (heur_sol, label).

    Any name/alias of the heuristics registry is accepted; additionally
    load_balanced_rdi / random_rdi / similarity_rdi run the CP-based station
    decomposition from decomposition.py.
    """
    h = heuristic_name.lower().replace("-", "_")
    if h in ("load_balanced_rdi", "lb_rdi"):
        from decomposition import solve_load_balanced_station_decomposition

        return solve_load_balanced_station_decomposition(instance), "Load-Balanced RDI"
    if h in ("random_rdi", "random_partition_rdi"):
        from decomposition import solve_station_decomposition

        return solve_station_decomposition(instance, partition_method="random"), "Random-Partitioned RDI"
    if h in ("similarity_rdi", "similarity_partition_rdi"):
        from decomposition import solve_station_decomposition

        return solve_station_decomposition(instance, partition_method="similarity"), "Similarity-Partitioned RDI"

    try:
        key = resolve(h)
    except ValueError:
        raise ValueError(
            f"Unknown heuristic '{heuristic_name}'. Available options: {sorted(HEURISTICS)}, "
            f"load_balanced_rdi, random_rdi, similarity_rdi"
        ) from None
    sol = run_heuristic(
        key,
        instance,
        horizon=config.get("horizon", 10000),
        alpha=config.get("alpha", 1.0),
        beta=config.get("beta", 0.0),
    )
    return sol, HEURISTIC_LABELS.get(key, key)


def precalculate_config(config):
    heur_name = config.get("heuristic", "load_balanced_greedy")
    base_name = f"GEN{config['gen_seed']}-{config['stations']}-{config['lanes']}-{config['skus']}-{config['orders']}-{config['movecap']}"
    path = f"precalculated_instances/{base_name}"
    if os.path.exists(path):
        cfg_file = f"{path}/config.json"
        if os.path.exists(cfg_file):
            try:
                with open(cfg_file, "r") as f:
                    existing_cfg = json.load(f)
                if existing_cfg != config:
                    path = f"precalculated_instances/{base_name}-{heur_name}"
            except Exception:
                pass

    os.makedirs(path, exist_ok=True)

    with open(f"{path}/config.json", "w+") as f:
        json.dump(config, f)

    instance = generate_data(
        num_stations=config["stations"],
        lanes_per_station=config["lanes"],
        num_orders=config["orders"],
        num_skus=config["skus"],
        seed=config["gen_seed"],
        movecap=config["movecap"],
        pick_touch_time=config["pick_touch_time"],
        return_model=config["return_model"],
        max_bins_per_sku=config["max_bins"],
        sku_popularity_skew=config["skew"],
        retrieval_time_model=config["rt_model"],
        pick_time_model=config["pick_model"],
        bin_count_model=config["bin_model"],
        grid_depth=config["grid_depth"],
    )

    instance.to_pickle(f"{path}/instance.pkl")

    with open(f"{path}/instance_summary.txt", "w+") as f, redirect_stdout(f):
        instance.print_summary()

    mdl, handles = cpp_cp_model.build_model(
        instance,
        add_symmetry_breaking=config["symmetry_breaking"],
        horizon=config["horizon"],
    )

    t0 = time.perf_counter()
    heur_sol, heur_label = solve_warmstart_heuristic(heur_name, instance, config)
    elapsed = time.perf_counter() - t0

    violations = validate_solution(heur_sol, instance, horizon=config["horizon"])
    violations.extend(validate_warmstart(heur_sol, heur_sol.pick_events, handles))
    if violations:
        print(
            f"[!] Warning: Heuristic solution has {len(violations)} validation violation(s):"
        )
        for v in violations[:10]:
            print(f"    - {v}")

    with open(f"{path}/heuristic_solution.pkl", "wb") as f:
        pickle.dump(heur_sol, f)

    with open(f"{path}/heuristic_summary.txt", "w+") as f:
        f.write(
            f"{heur_label} Result | Feasible: {heur_sol.feasible} | Makespan: {heur_sol.makespan} | BinEvents: {heur_sol.total_moves // 2} | Time: {elapsed:.3f}s\n"
        )

    mock_sol, viz_handles = heur_sol.to_viz(instance)
    fig = plot_schedule(mock_sol, viz_handles, show=False)
    html_file = f"{path}/heuristic_solution.html"
    write_html(fig, html_file)
    print(f"\nWrote visualization to {html_file}")

    # strict: raises instead of silently warm-starting CP from an infeasible solution
    sp = heur_sol.to_cpp_warmstart(mdl, handles)

    for seed in config["cp_seeds"]:
        mdl, handles = cpp_cp_model.build_model(
            instance,
            add_symmetry_breaking=config["symmetry_breaking"],
            horizon=config["horizon"],
        )

        history_listener = cpp_cp_model.ProgressCollector()
        mdl.add_solver_listener(history_listener)

        mdl.set_starting_point(sp)

        log_path = f"{path}/cplog_{seed}.txt"
        print(
            f"Solving CP Model with {config['time_limit']}s time limit and seed {seed} (C++ Backend)..."
        )
        sol_cp = mdl.solve(
            time_limit=config["time_limit"],
            workers=config["workers"],
            random_seed=seed,
            log_verbosity="Verbose",
            log_filename=log_path,
            collect_progress=True,
        )

        if sol_cp:
            sol = sol_cp.get_solution()
            if sol is not None:
                with open(f"{path}/cp_solution_{seed}.pkl", "wb") as f:
                    pickle.dump(sol, f)
            else:
                print("CP sol is None")

            with open(f"{path}/cp_intermediate_records_{seed}.pkl", "wb") as f:
                pickle.dump(history_listener.records, f)

            print(f"CP Solve Status: {sol_cp.get_solve_status()}")
            with open(f"{path}/cp_sol_{seed}.txt", "w+") as f, redirect_stdout(f):
                extract_and_print_solution(sol_cp, handles)
            if plot_schedule:
                print("Exporting visualization...")
                fig = plot_schedule(sol_cp, handles, show=False)
                html_file = f"{path}/cp_solution_{seed}.html"
                write_html(fig, html_file)
                print(f"\nWrote visualization to {html_file}")
        else:
            print("No solution found by CP.")

    os.makedirs(f"{path}/experiments", exist_ok=True)
    return path


def check_if_precalculated(config, with_removal=False):
    ret = None
    if not os.path.exists("precalculated_instances"):
        return None
    for past in os.listdir("precalculated_instances"):
        folder = f"precalculated_instances/{past}"
        if not os.path.isdir(folder):
            continue
        cfg_file = f"{folder}/config.json"
        if not os.path.exists(cfg_file):
            continue
        try:
            with open(cfg_file, "r") as f:
                past_config = json.load(f)
        except Exception:
            continue
        if past_config == config:
            if "experiments" in os.listdir(folder):
                print(f"Config {config} is already precalculated in {past}.")
                ret = folder
                break
            elif with_removal:
                if (
                    input("Found non-completed instance, remove and recalculate?")
                    == "yes"
                ):
                    shutil.rmtree(folder)
                else:
                    raise ValueError("Incomplete instance but no recalculate on prompt")
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
        "--gen-seeds",
        type=int,
        nargs="+",
        default=[42],
        help="List of random seeds for data generation",
    )
    parser.add_argument(
        "--cp-seeds",
        type=int,
        nargs="+",
        default=[42],
        help="List of CP seeds to try",
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

    parser.add_argument(
        "--order-dist",
        choices=["lognormal", "negbin", "poisson2_to_1_6", "uniform_1_5"],
        default="lognormal",
    )
    parser.add_argument(
        "--skew",
        type=float,
        default=1.0,
        help="Zipf exponent for SKU popularity (1.0 = classic 80/20)",
    )
    parser.add_argument(
        "--rt-model",
        choices=["depth_based", "triangular"],
        default="depth_based",
    )
    parser.add_argument(
        "--pick-model",
        choices=["variable", "constant"],
        default="variable",
    )
    parser.add_argument(
        "--return-model",
        choices=["state_preserving", "balancing"],
        default="balancing",
    )
    parser.add_argument(
        "--bin-model",
        choices=["popularity_correlated", "uniform"],
        default="popularity_correlated",
    )

    parser.add_argument("--pick", type=int, default=4)
    parser.add_argument("--max-bins", type=int, default=8)

    parser.add_argument("--grid-depth", type=int, default=16)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--heuristic",
        type=str,
        default="load_balanced_greedy",
        choices=[
            "load_balanced_greedy",
            "load_balanced_sgc",
            "standard_greedy",
            "standard_sgc",
            "rdi_sgc",
            "load_balanced_rdi",
            "random_rdi",
            "similarity_rdi",
        ],
        help="Warm-starting heuristic to use (default: load_balanced_greedy)",
    )
    args = parser.parse_args()

    TIME_LIMIT = args.time_limit
    GEN_SEEDS = args.gen_seeds
    CP_SEEDS = args.cp_seeds
    STATIONS = args.stations
    LANES = args.lanes
    SKUS = args.skus
    ORDERS = args.orders
    MOVECAPS = args.movecaps

    total = (
        len(GEN_SEEDS)
        * len(CP_SEEDS)
        * len(STATIONS)
        * len(LANES)
        * len(SKUS)
        * len(ORDERS)
        * len(MOVECAPS)
    )
    print(f"Will run {total} CP runs with {TIME_LIMIT}s each")
    print(f"Total duration = {total * TIME_LIMIT / 3600:.01f} h")
    print(
        f"Will be done at: {(dt.datetime.now() + dt.timedelta(seconds=total * TIME_LIMIT)).strftime('%d.%m.%Y %H:%M:%S')}"
    )
    print(
        f"Precalculating configs: heuristic={args.heuristic} {TIME_LIMIT=} {GEN_SEEDS=} {CP_SEEDS=} {STATIONS=} {LANES=} {SKUS=} {ORDERS=} {MOVECAPS=}"
    )

    for gen_seed, s, l, k, o, m in product(
        GEN_SEEDS, STATIONS, LANES, SKUS, ORDERS, MOVECAPS
    ):
        config = {
            "stations": s,
            "lanes": l,
            "orders": o,
            "symmetry_breaking": True,
            "skus": k,
            "movecap": m,
            "heuristic": args.heuristic,
            "order_dist": args.order_dist,
            "skew": args.skew,
            "rt_model": args.rt_model,
            "pick_model": args.pick_model,
            "pick_touch_time": args.pick,
            "return_model": args.return_model,
            "bin_model": args.bin_model,
            "max_bins": args.max_bins,
            "grid_depth": args.grid_depth,
            "gen_seed": gen_seed,
            "cp_seeds": CP_SEEDS,
            "horizon": 10000,
            "alpha": 1.0,
            "beta": 1.0,
            "time_limit": TIME_LIMIT,
            "workers": args.workers,
        }

        print("This config:", config)

        if check_if_precalculated(config, with_removal=True) is not None:
            continue

        precalculate_config(config)


if __name__ == "__main__":
    main()
