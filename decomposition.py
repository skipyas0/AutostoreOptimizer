import argparse
import datetime
import json
import os
from collections import defaultdict
from contextlib import redirect_stdout

from cp_model import build_model
from datagen import generate_data
from heuristics import ExoState, HeuristicSolution, run_partitioned_rdi, run_rdi, validate_warmstart
from heuristics.partitioned_rdi import (  # noqa: F401  (re-exported for older imports)
    active_station_count,
    create_load_balanced_partitions,
    create_random_partitions,
    create_similarity_partitions,
    distribute_movecap,
)
from matheuristic import load_instance


def extract_exo_state(sol, handles, overlap_threshold=0):
    """
    Extracts locked interval coordinates from a solved submodel to pass
    into subsequent decompositions.

    Parameters:
    - sol: The Docplex solution object returned by submodel.solve()
    - handles: The handles dictionary returned by build_model()
    - overlap_threshold: For rolling horizons, only extract intervals that
                         end strictly after this absolute timestamp.
    """
    exo_lanes = defaultdict(list)
    exo_blocks = defaultdict(list)
    exo_moves = []

    if sol is None or not sol.is_solution():
        return dict(exo_lanes), dict(exo_blocks), exo_moves

    def get_bounds(iv):
        vs = sol.get_var_solution(iv)
        if vs is not None and vs.is_present():
            return vs.get_start(), vs.get_end()
        return None

    # 1. Extract Lane Usage
    for (o, s, ln), iv in handles.get("I_os_lane", {}).items():
        bounds = get_bounds(iv)
        if bounds and bounds[1] > overlap_threshold:
            exo_lanes[(s, ln)].append(bounds)

    # 2. Extract SKU Bin Blocks
    for (s, k, e), iv in handles.get("Block", {}).items():
        bounds = get_bounds(iv)
        if bounds and bounds[1] > overlap_threshold:
            exo_blocks[k].append(bounds)

    # 3. Extract Robot Fetch/Return Moves
    for move_dict in (handles.get("F", {}), handles.get("R", {})):
        for key, iv in move_dict.items():
            bounds = get_bounds(iv)
            if bounds and bounds[1] > overlap_threshold:
                exo_moves.append(bounds)

    return dict(exo_lanes), dict(exo_blocks), exo_moves


def validate_exo_constraints(heur_sol, exo_lanes, exo_blocks, exo_moves, handles):
    violations = []

    # 1. Lane overlaps
    for o, (s_a, ln_a, t_s, t_e) in heur_sol.order_assignments.items():
        if (s_a, ln_a) in exo_lanes:
            for st, en in exo_lanes[(s_a, ln_a)]:
                if max(t_s, st) < min(t_e, en):
                    violations.append(
                        f"Exo Lane overlap: Order {o} [{t_s},{t_e}) overlaps with exo block [{st},{en}) at S{s_a} L{ln_a}"
                    )

    # 2. Block concurrency
    N_map = handles.get("N", {})
    for k in handles.get("K", []):
        blocks = []
        if k in exo_blocks:
            for st, en in exo_blocks[k]:
                blocks.append((st, en, "exo"))
        for s in handles.get("S", []):
            for be in heur_sol.bin_events.get(s, []):
                if be.sku == k:
                    blocks.append((be.fetch_start, be.return_end, f"heur_S{s}"))

        sweep = []
        for bs, be_end, label in blocks:
            sweep.append((bs, +1, label))
            sweep.append((be_end, -1, label))
        sweep.sort(key=lambda x: (x[0], x[1]))
        concurrent = 0
        for t, delta, label in sweep:
            concurrent += delta
            if concurrent > N_map.get(k, 1):
                violations.append(
                    f"Exo Block concurrency SKU {k}: {concurrent} > N[{k}]={N_map.get(k, 1)} at t={t}"
                )
                break

    # 3. MoveCap
    mc = handles.get("move_cap")
    if mc is not None:
        sweep = []
        if exo_moves:
            for st, en in exo_moves:
                sweep.append((st, +1))
                sweep.append((en, -1))
        for s in handles.get("S", []):
            for be in heur_sol.bin_events.get(s, []):
                sweep.append((be.fetch_start, +1))
                sweep.append((be.fetch_end, -1))
                sweep.append((be.return_start, +1))
                sweep.append((be.return_end, -1))
        sweep.sort(key=lambda x: (x[0], x[1]))
        concurrent = 0
        for t, delta in sweep:
            concurrent += delta
            if concurrent > mc:
                violations.append(
                    f"Exo MoveCap violation: {concurrent} > {mc} at t={t}"
                )
                break

    return violations


def _set_warm_start(submodel, handles, heur_sol, exo_lanes, exo_blocks, exo_moves, label: str) -> None:
    """Warm-start `submodel` from the RDI solution only if it is fully valid (CP model + exogenous
    constraints); otherwise print why and let CP start cold. Never injects an infeasible warm start."""
    if heur_sol is None or not heur_sol.feasible:
        print(f"[{label}] heuristic solution infeasible -> CP cold start")
        return
    hard = [v for v in validate_warmstart(heur_sol, heur_sol.pick_events, handles) if not v.startswith("Symmetry")]
    hard += validate_exo_constraints(heur_sol, exo_lanes, exo_blocks, exo_moves, handles)
    if hard:
        print(f"--- WARMSTART VIOLATIONS in {label} -> CP cold start ---")
        for v in hard:
            print(v)
        print("---------------------------------------")
        return
    print(f"Warmstart for {label} is strictly feasible!")
    submodel.set_starting_point(heur_sol.to_warmstart(submodel, handles))


def solve_station_decomposition(
    instance,
    exo_blocks_in=None,
    exo_lanes_in=None,
    exo_moves_in=None,
    batch_start_time=0,
    partition_method="random",
):
    n_partitions = active_station_count(instance)  # stations without a robot (MoveCap < |S|) stay unused
    if partition_method == "load_balanced":
        partitions = create_load_balanced_partitions(instance, n_partitions)
    elif partition_method == "similarity":
        partitions = create_similarity_partitions(instance, n_partitions)
    else:
        partitions = create_random_partitions(instance, n_partitions)
    workloads = [
        sum(instance.p.get(k, 0) for o in p for k in instance.orders_req[o])
        for p in partitions
    ]
    station_movecaps = distribute_movecap(instance.movecap, n_partitions, workloads)

    subsolutions = []

    # We accumulate exo blocks so stations solved later respect bins used by earlier stations
    current_exo_blocks = defaultdict(list)
    if exo_blocks_in:
        for k, v in exo_blocks_in.items():
            current_exo_blocks[k].extend(v)

    for i, p in enumerate(partitions):
        subinstance = instance.subinstance(p, stations=[instance.S[i]], movecap=station_movecaps[i])

        # Build and solve CP model
        horizon = batch_start_time + 10000
        submodel, handles = build_model(
            subinstance,
            add_symmetry_breaking=True,
            horizon=horizon,
            exo_lanes=exo_lanes_in,
            exo_blocks=current_exo_blocks,
            exo_moves=exo_moves_in,
            batch_start_time=batch_start_time,
        )

        heur_sol = run_rdi(
            subinstance,
            horizon=horizon,
            exo=ExoState(
                lanes=exo_lanes_in,
                blocks=current_exo_blocks,
                moves=exo_moves_in,
                batch_start_time=batch_start_time,
            ),
        )
        _set_warm_start(
            submodel, handles, heur_sol, exo_lanes_in or {}, current_exo_blocks, exo_moves_in or [],
            f"station spatial batch {i}",
        )

        subsolution_cp = submodel.solve(
            TimeLimit=60, LogVerbosity="Quiet"
        )  # configurable time limit

        # Extract constraints for horizontal sharing (spatial)
        _, new_blocks, _ = extract_exo_state(
            subsolution_cp, handles, overlap_threshold=-1
        )
        for k, blocks in new_blocks.items():
            current_exo_blocks[k].extend(blocks)

        heuristic_sol = HeuristicSolution.from_cp(subsolution_cp, handles)
        subsolutions.append(heuristic_sol)

    # Stitch solutions across stations
    return HeuristicSolution.stitch(subsolutions)


def solve_load_balanced_station_decomposition(
    instance,
    exo_blocks_in=None,
    exo_lanes_in=None,
    exo_moves_in=None,
    batch_start_time=0,
):
    return solve_station_decomposition(
        instance,
        exo_blocks_in=exo_blocks_in,
        exo_lanes_in=exo_lanes_in,
        exo_moves_in=exo_moves_in,
        batch_start_time=batch_start_time,
        partition_method="load_balanced",
    )


def solve_time_decomposition(
    instance,
    exo_blocks_in=None,
    exo_lanes_in=None,
    exo_moves_in=None,
    n_time_batches=5,
    overlap_coeff=0.1,
):
    """
    order_batches: list of lists of order IDs
    Solves them chronologically and stitches the result.
    """
    order_batches = create_random_partitions(instance, n_time_batches)
    print()
    subsolutions = []

    current_exo_lanes = defaultdict(list)
    if exo_lanes_in:
        for k, v in exo_lanes_in.items():
            current_exo_lanes[k].extend(v)

    current_exo_blocks = defaultdict(list)
    if exo_blocks_in:
        for k, v in exo_blocks_in.items():
            current_exo_blocks[k].extend(v)

    current_exo_moves = []
    if exo_moves_in:
        current_exo_moves.extend(exo_moves_in)

    current_batch_start = 0

    for i, batch in enumerate(order_batches):
        print(f"Solving time decomposition batch {i}")
        subinstance = instance.subinstance(batch)

        horizon = current_batch_start + 10000
        submodel, handles = build_model(
            subinstance,
            add_symmetry_breaking=True,
            horizon=horizon,
            exo_lanes=current_exo_lanes,
            exo_blocks=current_exo_blocks,
            exo_moves=current_exo_moves,
            batch_start_time=current_batch_start,
        )

        heur_sol = run_rdi(
            subinstance,
            horizon=horizon,
            exo=ExoState(
                lanes=current_exo_lanes,
                blocks=current_exo_blocks,
                moves=current_exo_moves,
                batch_start_time=current_batch_start,
            ),
        )
        _set_warm_start(
            submodel, handles, heur_sol, current_exo_lanes, current_exo_blocks, current_exo_moves, f"batch {i}"
        )
        subsolution_cp = submodel.solve(LogVerbosity="Terse", TimeLimit=60)

        if subsolution_cp is not None and subsolution_cp.is_solution():
            heuristic_sol = HeuristicSolution.from_cp(subsolution_cp, handles)
            subsolutions.append(heuristic_sol)

            if heuristic_sol and heuristic_sol.order_assignments:
                max_end = max(b[3] for b in heuristic_sol.order_assignments.values())
                batch_span = max_end - current_batch_start
                next_batch_start = int(max_end - batch_span * overlap_coeff)
            else:
                next_batch_start = current_batch_start

            new_lanes, new_blocks, new_moves = extract_exo_state(
                subsolution_cp, handles, overlap_threshold=next_batch_start
            )

            for key, intervals in new_lanes.items():
                current_exo_lanes[key].extend(intervals)
            for key, intervals in current_exo_lanes.items():
                current_exo_lanes[key] = [
                    iv for iv in intervals if iv[1] > next_batch_start
                ]

            for key, intervals in new_blocks.items():
                current_exo_blocks[key].extend(intervals)
            for key, intervals in current_exo_blocks.items():
                current_exo_blocks[key] = [
                    iv for iv in intervals if iv[1] > next_batch_start
                ]

            current_exo_moves.extend(new_moves)
            current_exo_moves = [
                iv for iv in current_exo_moves if iv[1] > next_batch_start
            ]

            current_batch_start = next_batch_start
        else:
            print("Failed to solve time batch")
            break

    return HeuristicSolution.stitch(subsolutions)


def solve_2d_decomposition(instance, n_time_batches=5):
    """
    Combines spatial and temporal decomposition.
    time_batches: list of lists of order IDs
    """
    subsolutions = []
    time_batches = create_random_partitions(instance, n_time_batches)

    current_exo_lanes = defaultdict(list)
    current_exo_blocks = defaultdict(list)
    current_exo_moves = []

    current_batch_start = 0

    for batch in time_batches:
        subinstance = instance.subinstance(batch)

        # We run the spatial decomposition on this time slice
        batch_sol = solve_station_decomposition(
            subinstance,
            exo_blocks_in=current_exo_blocks,
            exo_lanes_in=current_exo_lanes,
            exo_moves_in=current_exo_moves,
            batch_start_time=current_batch_start,
        )

        if batch_sol:
            subsolutions.append(batch_sol)

            # We don't have a CP solution here to extract_exo_state easily,
            # we need to extract from heuristic solution!
            current_exo_lanes.clear()
            current_exo_blocks.clear()
            current_exo_moves.clear()

            for o, (s_sel, ln_sel, st, en) in batch_sol.order_assignments.items():
                current_exo_lanes[(s_sel, ln_sel)].append((st, en))

            for s, evts in batch_sol.bin_events.items():
                for ev in evts:
                    current_exo_blocks[ev.sku].append((ev.fetch_start, ev.return_end))
                    current_exo_moves.append((ev.fetch_start, ev.fetch_end))
                    current_exo_moves.append((ev.return_start, ev.return_end))

            # For a rolling horizon, we should ideally filter these to only intervals extending past current_batch_start.
            # But passing all is physically safe (just builds a bigger CP model over time).
            current_batch_start = batch_sol.makespan
        else:
            print("Failed to solve spatial decomposition for time batch")
            break

    return HeuristicSolution.stitch(subsolutions)


def solve_2d_decomposition_spatial_first(instance, n_time_batches=5):
    """
    Combines spatial and temporal decomposition by partitioning spatially first,
    then solving each station temporally.
    """
    n_partitions = active_station_count(instance)  # stations without a robot (MoveCap < |S|) stay unused
    spatial_partitions = create_random_partitions(instance, n_partitions)
    workloads = [len(p) for p in spatial_partitions]
    station_movecaps = distribute_movecap(instance.movecap, n_partitions, workloads)

    subsolutions = []
    current_exo_blocks = defaultdict(list)

    for i, p in enumerate(spatial_partitions):
        print(f"Solving station {i}")
        subinstance = instance.subinstance(p, stations=[instance.S[i]], movecap=station_movecaps[i])

        station_sol = solve_time_decomposition(
            subinstance, exo_blocks_in=current_exo_blocks, n_time_batches=n_time_batches
        )

        if station_sol:
            subsolutions.append(station_sol)

            for s, evts in station_sol.bin_events.items():
                for ev in evts:
                    current_exo_blocks[ev.sku].append((ev.fetch_start, ev.return_end))
        else:
            print(f"Failed to solve time decomposition for station {instance.S[i]}")
            break

    return HeuristicSolution.stitch(subsolutions)


def solve_rl_decomposition(instance, weights_path="rl_agent_weights.pth", device=None):
    import torch

    from rl_order_assignment import (
        OrderAssignmentAgent,
        OrderGraphGNN,
        StationOrderScorer,
        precompute_instance_similarities,
        rollout_station_order,
    )

    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    checkpoint = None
    try:
        checkpoint = torch.load(weights_path, map_location=device)
        print(f"Loaded RL weights from {weights_path} to {device}")
    except FileNotFoundError:
        print(f"Warning: {weights_path} not found. Using untrained RL agent.")

    if checkpoint is not None and any(k.startswith("net.") for k in checkpoint.keys()):
        feat_dim = checkpoint["net.0.weight"].shape[1]
        hid_dim = checkpoint["net.0.weight"].shape[0]
        model = StationOrderScorer(feature_dim=feat_dim, hidden_dim=hid_dim).to(device)
        model.load_state_dict(checkpoint)
        model.eval()

        sim_unw, sim_w = precompute_instance_similarities(instance)
        with torch.no_grad():
            actions, _, _, _ = rollout_station_order(
                instance, model, sim_unw, sim_w, device, deterministic=True
            )
    else:
        hidden_dim = 64
        mlp_hidden = 128
        max_stations = 20

        gnn = OrderGraphGNN(hidden_dim=hidden_dim, num_layers=2)
        agent = OrderAssignmentAgent(
            gnn, hidden_dim=hidden_dim, max_stations=max_stations, mlp_hidden=mlp_hidden
        ).to(device)
        if checkpoint is not None:
            agent.load_state_dict(checkpoint)
        agent.eval()

        hetero_data = instance.create_order_graph().to(device)
        instance_stats = {
            "num_stations": len(instance.S),
            "total_used_skus": instance.stats["used_skus"],
            "movecap": instance.movecap,
        }

        order_num_skus = torch.tensor(
            [len(instance.orders_req[o]) for o in instance.O],
            dtype=torch.float,
            device=device,
        )

        with torch.no_grad():
            actions, _, _ = agent(hetero_data, instance_stats, order_num_skus, deterministic=True)

    station_assignments = {s: [] for s in range(len(instance.S))}
    for o_idx, s_idx in enumerate(actions):
        s_idx = min(s_idx, len(instance.S) - 1)
        station_assignments[s_idx].append(instance.O[o_idx])

    return run_partitioned_rdi(instance, partition=station_assignments, horizon=50000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instance", type=str, default=None, help="Presolved instance folder name"
    )
    parser.add_argument(
        "--time-limit", type=int, default=3600, help="CP solver time limit in seconds"
    )
    parser.add_argument(
        "--gen-seed", type=int, default=42, help="Random seed for data generation"
    )
    parser.add_argument(
        "--cp-seed",
        type=int,
        default=42,
        help="Seed for CP Optimizer",
    )

    parser.add_argument(
        "--stations",
        type=int,
        default=10,
        help="Number of picker stations",
    )
    parser.add_argument(
        "--lanes",
        type=int,
        default=4,
        help="Number of max open order lanes",
    )
    parser.add_argument(
        "--skus",
        type=int,
        default=20000,
        help="Unique SKUs in generated warehouse",
    )
    parser.add_argument(
        "--orders",
        type=int,
        default=1000,
        help="Number of orders to fulfill",
    )
    parser.add_argument(
        "--movecap",
        type=int,
        default=40,
        help="Max simultaneous robot moves",
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

    parser.add_argument(
        "--decomposition-type",
        choices=["temporal", "spatial", "load_balanced", "2d_temporal", "2d_spatial", "rl"],
    )

    parser.add_argument("--pick", type=int, default=4, help="Pick touch time")
    parser.add_argument("--max-bins", type=int, default=8)

    parser.add_argument("--grid-depth", type=int, default=16)

    args = parser.parse_args()

    instance = args.instance
    if instance is None:
        config = {
            "stations": args.stations,
            "lanes": args.lanes,
            "orders": args.orders,
            "symmetry_breaking": True,
            "skus": args.skus,
            "movecap": args.movecap,
            "order_dist": args.order_dist,
            "skew": args.skew,
            "rt_model": args.rt_model,
            "pick_model": args.pick_model,
            "pick_touch_time": args.pick,
            "return_model": args.return_model,
            "bin_model": args.bin_model,
            "max_bins": args.max_bins,
            "grid_depth": args.grid_depth,
            "gen_seed": args.gen_seed,
            "cp_seeds": args.cp_seed,
            "horizon": 10000,
            "alpha": 1.0,
            "beta": 1.0,
            "time_limit": args.time_limit,
        }

        decomp = args.decomposition_type
        dt = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        path = f"decomposition_experiments/GEN{config['gen_seed']}-{config['stations']}-{config['lanes']}-{config['skus']}-{config['orders']}-{config['movecap']}-{decomp}-{dt}"

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
        instance_path = path
    else:
        if "precalculated_instances/" in instance:
            instance = instance.split("/")[1]
        (
            instance,
            heur_sol,
            # self.cp_records,
            instance_path,
            instance_config,
        ) = load_instance(None, instance_folder=f"precalculated_instances/{instance}")

    n_batches = max(args.orders // 50, 5)
    n_batches_2d = min(args.orders // (50 * args.stations), 5)
    if decomp == "temporal":
        stitched_sol = solve_time_decomposition(instance, n_time_batches=n_batches)

    elif decomp == "spatial":
        stitched_sol = solve_station_decomposition(instance)

    elif decomp == "load_balanced":
        stitched_sol = solve_load_balanced_station_decomposition(instance)

    elif decomp == "2d_spatial":
        stitched_sol = solve_2d_decomposition_spatial_first(
            instance, n_time_batches=n_batches_2d
        )

    elif decomp == "2d_temporal":
        stitched_sol = solve_2d_decomposition(instance, n_time_batches=n_batches_2d)

    elif decomp == "rl":
        stitched_sol = solve_rl_decomposition(instance)
    else:
        raise ValueError

    from schedule_visualizer import plot_schedule, write_html

    mock_sol, handles = stitched_sol.to_viz(instance)
    fig = plot_schedule(mock_sol, handles, show=True)
    html_file = f"./{instance_path}/autostore_{decomp}_decomposition_solution.html"
    write_html(fig, html_file)
    print(f"\nWrote visualisation to {html_file}")
