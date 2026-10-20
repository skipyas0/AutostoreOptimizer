#!/usr/bin/env python3
"""
Benchmark script to compare pretrained RL order-to-station assignment model checkpoints (MLP/GNN)
against:
  1. Random-partitioned RDI
  2. Similarity-partitioned RDI
  3. Load-balanced RDI
  4. Full RDI without decomposition
  5. Standard Full Greedy Heuristic (autostore_heuristic.py SGC)
  6. Load-balanced Greedy Heuristic (autostore_heuristic.py SGC with restricted candidate stations)

Instances are generated using the exact distribution and parameters from the RL training loop,
with disjoint random seeds. Metrics reported are the exact components and reward formulations
from the RL environment, plus average solution times.

Usage:
  .venv/bin/python benchmark_models.py
  .venv/bin/python benchmark_models.py --checkpoints rl_agent_weights_mlp_1200.pth --num-instances 10
"""

import argparse
import glob
import json
import os
import random
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# Suppress matplotlib font cache warnings
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_config")
import logging

logging.getLogger("matplotlib").setLevel(logging.ERROR)

import torch

# Ensure code root is in path
code_dir = os.path.dirname(os.path.abspath(__file__))
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from autostore_heuristic import run_sgc
from datagen import generate_data
from decomposition import (
    create_load_balanced_partitions,
    create_random_partitions,
    create_similarity_partitions,
    distribute_movecap,
)
from heuristic_rdi_sgc import run_rdi_sgc
from instance import Instance
from rl_order_assignment import (
    OrderAssignmentAgent,
    OrderGraphGNN,
    StationOrderScorer,
    precompute_instance_similarities,
    rollout_station_order,
)


def get_default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_agent(
    checkpoint_path: str,
    device: torch.device,
    hidden_dim: int = 64,
    max_stations: int = 20,
    mlp_hidden: int = 128,
):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if any(k.startswith("net.") for k in checkpoint.keys()):
        # StationOrderScorer model
        feat_dim = checkpoint["net.0.weight"].shape[1]
        hid_dim = checkpoint["net.0.weight"].shape[0]
        model = StationOrderScorer(feature_dim=feat_dim, hidden_dim=hid_dim).to(device)
        model.load_state_dict(checkpoint)
        model.eval()
        return model
    else:
        # Legacy GNN OrderAssignmentAgent
        gnn = OrderGraphGNN(hidden_dim=hidden_dim, num_layers=2)
        agent = OrderAssignmentAgent(
            gnn, hidden_dim=hidden_dim, max_stations=max_stations, mlp_hidden=mlp_hidden
        ).to(device)
        agent.load_state_dict(checkpoint)
        agent.eval()
        return agent


def generate_benchmark_instance(instance_idx: int, seed_base: int = 100000) -> Instance:
    """
    Generates an instance matching the training loop distribution in rl_order_assignment.py
    using a disjoint seed range.
    """
    rng = random.Random(seed_base + instance_idx)
    s = rng.randint(2, 12)
    l = rng.randint(3, 10)
    if s > 5:
        o = rng.randint(2, 8) * 50
        m = rng.randint(4, 8) * 5
        sku = rng.randint(8, 15) * 2000
    else:
        o = rng.randint(2, 8) * 10
        m = rng.randint(4, 8) * 2
        sku = rng.randint(8, 15) * 1000

    instance = generate_data(
        num_stations=s,
        lanes_per_station=l,
        num_orders=o,
        num_skus=sku,
        seed=seed_base + instance_idx,
        movecap=m,
        pick_touch_time=4,
        return_model="balancing",
        max_bins_per_sku=8,
        sku_popularity_skew=1.0 + 0.1 * (rng.random() - 0.5),
        retrieval_time_model="depth_based",
        pick_time_model="variable",
        bin_count_model="popularity_correlated",
        grid_depth=16 + rng.randint(-2, 2),
    )
    return instance


def compute_metrics_from_station_results(
    instance: Instance,
    M_k: List[float],
    W_k: List[float],
    Be_k: List[int],
    P_k: List[int],
    Rt_k: List[float],
    elapsed_time: float,
    partition_time: float = 0.0,
) -> Dict[str, float]:
    """
    Calculates the exact reward and summary metrics defined in the RL environment.
    """
    num_stations = len(instance.S)

    # 1. Upper bound: robot fetch + return limit
    upper_bound = sum(
        instance.rt.get(k, 0) + instance.rt_ret.get(k, 0)
        for o in instance.O
        for k in instance.orders_req[o]
    ) / max(instance.movecap, 1)

    # 2. Lower bound: picker saturation bound
    lower_bound = sum(
        instance.p.get(k, 0) for o in instance.O for k in instance.orders_req[o]
    ) / max(num_stations, 1)

    scale_estimate = (upper_bound + lower_bound) / 2
    scale_estimate = max(scale_estimate, 1e-6)

    M = max(M_k) if M_k else 0.0
    avg_W = sum(W_k) / len(W_k) if W_k else 0.0
    avg_M = sum(M_k) / len(M_k) if M_k else 0.0
    min_M = min(M_k) if M_k else 0.0

    total_rt = sum(Rt_k)
    total_be = sum(Be_k)
    total_p = sum(P_k)

    movecap_utilization = (
        total_rt / (M * instance.movecap) if (M > 0 and instance.movecap > 0) else 0.0
    )
    avg_pick_imbalance = (
        sum(abs(w - avg_W) for w in W_k) / num_stations if num_stations > 0 else 0.0
    )
    shift_length_coeff = (avg_M + num_stations * min_M) / M if M > 0 else 0.0
    picks_from_bin_presence = (total_p / total_be - 1.0) if total_be > 0 else 0.0
    normalized_M = M / scale_estimate

    # RL Reward formulations
    reward_neg_nm = -normalized_M
    reward_shift_nm = shift_length_coeff - 2.0 * normalized_M
    reward_multicomponent = (
        5.0 * movecap_utilization
        + 1000.0 * picks_from_bin_presence
        + shift_length_coeff
        - (3.0 * normalized_M + 0.1 * avg_pick_imbalance)
    )

    return {
        "makespan": float(M),
        "normalized_makespan": float(normalized_M),
        "avg_pick_imbalance": float(avg_pick_imbalance),
        "movecap_utilization": float(movecap_utilization),
        "picks_from_bin_presence": float(picks_from_bin_presence),
        "shift_length_coeff": float(shift_length_coeff),
        "reward_neg_nm": float(reward_neg_nm),
        "reward_shift_nm": float(reward_shift_nm),
        "reward_multicomponent": float(reward_multicomponent),
        "total_time": float(elapsed_time),
        "partition_time": float(partition_time),
        "solve_time": float(max(0.0, elapsed_time - partition_time)),
    }


def evaluate_station_assignments(
    instance: Instance,
    station_assignments: Dict[int, List[int]],
    horizon: int = 50000,
) -> Tuple[List[float], List[float], List[int], List[int], List[float]]:
    """
    Solves subinstances station by station using run_rdi_sgc and extracts station-level metrics.
    """
    M_k = []
    W_k = []
    Be_k = []
    P_k = []
    Rt_k = []

    n_stations = len(instance.S)
    workloads = [len(station_assignments.get(i, [])) for i in range(n_stations)]
    station_movecaps = distribute_movecap(instance.movecap, n_stations, workloads)

    for i, s_idx in enumerate(instance.S):
        assigned_orders = station_assignments.get(i, [])

        # Workload W_k
        w = sum(
            instance.p.get(k, 0)
            for o in assigned_orders
            for k in instance.orders_req[o]
        )
        W_k.append(float(w))

        if not assigned_orders:
            M_k.append(0.0)
            Be_k.append(0)
            P_k.append(0)
            Rt_k.append(0.0)
            continue

        subinstance = Instance(
            [s_idx],
            instance.L,
            instance.K,
            {o: instance.orders_req[o] for o in assigned_orders},
            instance.rt,
            instance.p,
            instance.N,
            station_movecaps[i],
            instance.seed,
            instance.rt_ret,
            instance.pickcap,
        )

        heur_sol = run_rdi_sgc(subinstance, horizon=horizon, batch_start_time=0)
        if heur_sol and heur_sol.feasible:
            M_k.append(float(heur_sol.makespan))
            events = heur_sol.bin_events.get(s_idx, [])
            Be_k.append(len(events))
            P_k.append(len(heur_sol.pick_events))
            rt_sum = sum(
                be.fetch_end - be.fetch_start + be.return_end - be.return_start
                for be in events
            )
            Rt_k.append(float(rt_sum))
        else:
            M_k.append(50000.0)
            Be_k.append(0)
            P_k.append(0)
            Rt_k.append(0.0)

    return M_k, W_k, Be_k, P_k, Rt_k


def run_random_partition_rdi(instance: Instance) -> Dict[str, float]:
    t0 = time.perf_counter()
    partitions = create_random_partitions(instance, len(instance.S))
    t_partition = time.perf_counter() - t0

    station_assignments = {i: list(p) for i, p in enumerate(partitions)}
    M_k, W_k, Be_k, P_k, Rt_k = evaluate_station_assignments(
        instance, station_assignments
    )
    t_total = time.perf_counter() - t0

    return compute_metrics_from_station_results(
        instance, M_k, W_k, Be_k, P_k, Rt_k, t_total, partition_time=t_partition
    )


def run_similarity_partition_rdi(instance: Instance) -> Dict[str, float]:
    t0 = time.perf_counter()
    partitions = create_similarity_partitions(instance, len(instance.S))
    t_partition = time.perf_counter() - t0

    station_assignments = {i: list(p) for i, p in enumerate(partitions)}
    M_k, W_k, Be_k, P_k, Rt_k = evaluate_station_assignments(
        instance, station_assignments
    )
    t_total = time.perf_counter() - t0

    return compute_metrics_from_station_results(
        instance, M_k, W_k, Be_k, P_k, Rt_k, t_total, partition_time=t_partition
    )


def run_load_balanced_partition_rdi(instance: Instance) -> Dict[str, float]:
    t0 = time.perf_counter()
    partitions = create_load_balanced_partitions(instance, len(instance.S))
    t_partition = time.perf_counter() - t0

    station_assignments = {i: list(p) for i, p in enumerate(partitions)}
    M_k, W_k, Be_k, P_k, Rt_k = evaluate_station_assignments(
        instance, station_assignments
    )
    t_total = time.perf_counter() - t0

    return compute_metrics_from_station_results(
        instance, M_k, W_k, Be_k, P_k, Rt_k, t_total, partition_time=t_partition
    )


def _extract_monolithic_solution_metrics(
    instance: Instance, heur_sol, t_total: float
) -> Dict[str, float]:
    M_k = []
    W_k = []
    Be_k = []
    P_k = []
    Rt_k = []

    if heur_sol and heur_sol.feasible:
        for s_idx in instance.S:
            assigned_orders = [
                o
                for o, (s, ln, st, en) in heur_sol.order_assignments.items()
                if s == s_idx
            ]
            w = sum(
                instance.p.get(k, 0)
                for o in assigned_orders
                for k in instance.orders_req[o]
            )
            W_k.append(float(w))

            stn_makespan = max(
                (
                    en
                    for o, (s, ln, st, en) in heur_sol.order_assignments.items()
                    if s == s_idx
                ),
                default=0.0,
            )
            M_k.append(float(stn_makespan))

            events = heur_sol.bin_events.get(s_idx, [])
            Be_k.append(len(events))

            # Number of pick events for orders at this station
            stn_picks = sum(
                1 for (o, stn, k) in heur_sol.pick_events.keys() if stn == s_idx
            )
            P_k.append(stn_picks)

            rt_sum = sum(
                be.fetch_end - be.fetch_start + be.return_end - be.return_start
                for be in events
            )
            Rt_k.append(float(rt_sum))
    else:
        for _ in instance.S:
            M_k.append(50000.0)
            W_k.append(0.0)
            Be_k.append(0)
            P_k.append(0)
            Rt_k.append(0.0)

    return compute_metrics_from_station_results(
        instance, M_k, W_k, Be_k, P_k, Rt_k, t_total, partition_time=0.0
    )


def run_full_rdi_no_decomposition(
    instance: Instance, horizon: int = 50000
) -> Dict[str, float]:
    t0 = time.perf_counter()
    heur_sol = run_rdi_sgc(instance, horizon=horizon, batch_start_time=0)
    t_total = time.perf_counter() - t0
    return _extract_monolithic_solution_metrics(instance, heur_sol, t_total)


def run_full_greedy_heuristic(
    instance: Instance,
    horizon: int = 50000,
    restrict_stations_by_workload: bool = False,
) -> Dict[str, float]:
    t0 = time.perf_counter()
    heur_sol = run_sgc(
        instance,
        horizon=horizon,
        restrict_stations_by_workload=restrict_stations_by_workload,
    )
    t_total = time.perf_counter() - t0
    return _extract_monolithic_solution_metrics(instance, heur_sol, t_total)


def run_rl_checkpoint_rdi(
    instance: Instance,
    agent,
    device: torch.device,
    deterministic: bool = True,
) -> Dict[str, float]:
    t0 = time.perf_counter()

    if isinstance(agent, StationOrderScorer):
        sim_unw, sim_w = precompute_instance_similarities(instance)
        with torch.no_grad():
            actions, _, _, _ = rollout_station_order(
                instance, agent, sim_unw, sim_w, device, deterministic=deterministic
            )
    else:
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
            actions, _, _ = agent(
                hetero_data, instance_stats, order_num_skus, deterministic=deterministic
            )

    t_partition = time.perf_counter() - t0

    station_assignments = {s: [] for s in range(len(instance.S))}
    for o_idx, s_idx in enumerate(actions):
        s_idx = min(s_idx, len(instance.S) - 1)
        station_assignments[s_idx].append(instance.O[o_idx])

    M_k, W_k, Be_k, P_k, Rt_k = evaluate_station_assignments(
        instance, station_assignments
    )
    t_total = time.perf_counter() - t0

    return compute_metrics_from_station_results(
        instance, M_k, W_k, Be_k, P_k, Rt_k, t_total, partition_time=t_partition
    )


def print_comparison_table(results_by_method: Dict[str, List[Dict[str, float]]]):
    """
    Prints a formatted summary table of average metrics across all benchmark instances.
    """
    print("\n" + "=" * 128)
    print(f"{'BENCHMARK SUMMARY (Averages across test instances)':^128}")
    print("=" * 128)

    header = (
        f"{'Method':<32} | {'Makespan (M)':<13} | {'Norm M (NM)':<12} | "
        f"{'Pick Imb (PI)':<13} | {'Movecap Util':<12} | {'Picks/Bin (PBP)':<15} | "
        f"{'Reward (-NM)':<12} | {'Avg Time (s)':<12}"
    )
    print(header)
    print("-" * 128)

    for method_name, results in results_by_method.items():
        if not results:
            continue
        n = len(results)
        avg_m = sum(r["makespan"] for r in results) / n
        avg_nm = sum(r["normalized_makespan"] for r in results) / n
        avg_pi = sum(r["avg_pick_imbalance"] for r in results) / n
        avg_mu = sum(r["movecap_utilization"] for r in results) / n
        avg_pbp = sum(r["picks_from_bin_presence"] for r in results) / n
        avg_rew = sum(r["reward_neg_nm"] for r in results) / n
        avg_time = sum(r["total_time"] for r in results) / n

        row = (
            f"{method_name:<32} | {avg_m:13.1f} | {avg_nm:12.4f} | "
            f"{avg_pi:13.2f} | {avg_mu:12.4f} | {avg_pbp:15.4f} | "
            f"{avg_rew:12.4f} | {avg_time:12.3f}"
        )
        print(row)

    print("=" * 128 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark pretrained RL checkpoints against heuristic partitioning baselines and full RDI."
    )
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        default=None,
        help="List of model checkpoint paths (.pth). If not specified, auto-discovers rl_agent_*.pth in current dir.",
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=10,
        help="Number of randomized instances to benchmark (default: 10).",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=100000,
        help="Base seed for instance generation (disjoint from training seeds, default: 100000).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for PyTorch models ('mps', 'cuda', 'cpu'). Auto-detected by default.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions stochastically instead of taking argmax actions during RL inference.",
    )
    parser.add_argument(
        "--skip-full-rdi",
        action="store_true",
        help="Skip Full RDI (without decomposition) if speed is a concern on large instances.",
    )
    parser.add_argument(
        "--skip-greedy",
        action="store_true",
        help="Skip Standard Greedy Heuristic (autostore_heuristic.py SGC) without decomposition.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="benchmark_results.json",
        help="Path to save detailed benchmark results as JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_default_device()
    print(f"[*] Running benchmark on PyTorch device: {device}")

    # Discover checkpoints
    checkpoints = args.checkpoints
    if not checkpoints:
        all_pths = sorted(glob.glob("rl_agent_weights*.pth"))
        checkpoints = all_pths
        if not checkpoints:
            print("[!] Warning: No .pth files found in current directory.")
        else:
            print(
                f"[*] Auto-discovered {len(checkpoints)} checkpoint(s): {checkpoints}"
            )

    # Load RL models
    loaded_models: Dict[str, OrderAssignmentAgent] = {}
    for cp_path in checkpoints:
        if not os.path.exists(cp_path):
            print(f"[!] Warning: Checkpoint file '{cp_path}' does not exist. Skipping.")
            continue
        short_name = os.path.basename(cp_path).replace(".pth", "")
        try:
            agent = load_agent(cp_path, device=device)
            loaded_models[f"RL ({short_name})"] = agent
            print(f"[+] Successfully loaded checkpoint: {cp_path}")
        except Exception as e:
            print(f"[!] Failed to load checkpoint {cp_path}: {e}")

    # Setup methods list
    methods = []
    for model_name in loaded_models.keys():
        methods.append(model_name)
    methods.append("1. Random-partitioned RDI")
    methods.append("2. Similarity-partitioned RDI")
    methods.append("3. Load-balanced RDI")
    if not args.skip_full_rdi:
        methods.append("4. Full RDI (No Decomposition)")
    if not args.skip_greedy:
        methods.append("5. Standard Full Greedy (SGC)")
        methods.append("6. Load-balanced Greedy (SGC)")

    print(
        f"\n[*] Evaluating {len(methods)} method(s) across {args.num_instances} instance(s)..."
    )

    results_by_method: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    detailed_history = []

    for inst_idx in range(args.num_instances):
        print(
            f"\n>>> Instance {inst_idx + 1:02d}/{args.num_instances} (Seed: {args.seed_base + inst_idx})"
        )
        instance = generate_benchmark_instance(inst_idx, seed_base=args.seed_base)
        print(
            f"    Specs: Stations={len(instance.S)}, Lanes={len(instance.L)}, "
            f"Orders={len(instance.O)}, SKUs={len(instance.K)}, Movecap={instance.movecap}"
        )

        inst_record = {
            "instance_idx": inst_idx,
            "seed": args.seed_base + inst_idx,
            "stations": len(instance.S),
            "orders": len(instance.O),
            "skus": len(instance.K),
            "movecap": instance.movecap,
            "methods": {},
        }

        # 1. Pretrained RL Models
        for model_label, agent in loaded_models.items():
            try:
                res = run_rl_checkpoint_rdi(
                    instance, agent, device=device, deterministic=not args.stochastic
                )
                results_by_method[model_label].append(res)
                inst_record["methods"][model_label] = res
                print(
                    f"    - {model_label:<32} | Makespan: {res['makespan']:6.0f} | "
                    f"Norm M: {res['normalized_makespan']:7.4f} | Imbalance: {res['avg_pick_imbalance']:7.2f} | "
                    f"Time: {res['total_time']:6.2f}s"
                )
            except Exception as e:
                print(f"    - {model_label:<32} | FAILED: {e}")

        # 2. Random-partitioned RDI
        try:
            res_rand = run_random_partition_rdi(instance)
            results_by_method["1. Random-partitioned RDI"].append(res_rand)
            inst_record["methods"]["1. Random-partitioned RDI"] = res_rand
            print(
                f"    - {'1. Random-partitioned RDI':<32} | Makespan: {res_rand['makespan']:6.0f} | "
                f"Norm M: {res_rand['normalized_makespan']:7.4f} | Imbalance: {res_rand['avg_pick_imbalance']:7.2f} | "
                f"Time: {res_rand['total_time']:6.2f}s"
            )
        except Exception as e:
            print(f"    - {'1. Random-partitioned RDI':<32} | FAILED: {e}")

        # 3. Similarity-partitioned RDI
        try:
            res_sim = run_similarity_partition_rdi(instance)
            results_by_method["2. Similarity-partitioned RDI"].append(res_sim)
            inst_record["methods"]["2. Similarity-partitioned RDI"] = res_sim
            print(
                f"    - {'2. Similarity-partitioned RDI':<32} | Makespan: {res_sim['makespan']:6.0f} | "
                f"Norm M: {res_sim['normalized_makespan']:7.4f} | Imbalance: {res_sim['avg_pick_imbalance']:7.2f} | "
                f"Time: {res_sim['total_time']:6.2f}s"
            )
        except Exception as e:
            print(f"    - {'2. Similarity-partitioned RDI':<32} | FAILED: {e}")

        # 4. Load-balanced RDI
        try:
            res_lb = run_load_balanced_partition_rdi(instance)
            results_by_method["3. Load-balanced RDI"].append(res_lb)
            inst_record["methods"]["3. Load-balanced RDI"] = res_lb
            print(
                f"    - {'3. Load-balanced RDI':<32} | Makespan: {res_lb['makespan']:6.0f} | "
                f"Norm M: {res_lb['normalized_makespan']:7.4f} | Imbalance: {res_lb['avg_pick_imbalance']:7.2f} | "
                f"Time: {res_lb['total_time']:6.2f}s"
            )
        except Exception as e:
            print(f"    - {'3. Load-balanced RDI':<32} | FAILED: {e}")

        # 5. Full RDI without decomposition
        if not args.skip_full_rdi:
            try:
                res_full = run_full_rdi_no_decomposition(instance)
                results_by_method["4. Full RDI (No Decomposition)"].append(res_full)
                inst_record["methods"]["4. Full RDI (No Decomposition)"] = res_full
                print(
                    f"    - {'4. Full RDI (No Decomposition)':<32} | Makespan: {res_full['makespan']:6.0f} | "
                    f"Norm M: {res_full['normalized_makespan']:7.4f} | Imbalance: {res_full['avg_pick_imbalance']:7.2f} | "
                    f"Time: {res_full['total_time']:6.2f}s"
                )
            except Exception as e:
                print(f"    - {'4. Full RDI (No Decomposition)':<32} | FAILED: {e}")

        # 6. Standard Full Greedy Heuristic (SGC) without decomposition
        if not args.skip_greedy:
            try:
                res_greedy = run_full_greedy_heuristic(
                    instance, restrict_stations_by_workload=False
                )
                results_by_method["5. Standard Full Greedy (SGC)"].append(res_greedy)
                inst_record["methods"]["5. Standard Full Greedy (SGC)"] = res_greedy
                print(
                    f"    - {'5. Standard Full Greedy (SGC)':<32} | Makespan: {res_greedy['makespan']:6.0f} | "
                    f"Norm M: {res_greedy['normalized_makespan']:7.4f} | Imbalance: {res_greedy['avg_pick_imbalance']:7.2f} | "
                    f"Time: {res_greedy['total_time']:6.2f}s"
                )
            except Exception as e:
                print(f"    - {'5. Standard Full Greedy (SGC)':<32} | FAILED: {e}")

            try:
                res_lb_greedy = run_full_greedy_heuristic(
                    instance, restrict_stations_by_workload=True
                )
                results_by_method["6. Load-balanced Greedy (SGC)"].append(res_lb_greedy)
                inst_record["methods"]["6. Load-balanced Greedy (SGC)"] = res_lb_greedy
                print(
                    f"    - {'6. Load-balanced Greedy (SGC)':<32} | Makespan: {res_lb_greedy['makespan']:6.0f} | "
                    f"Norm M: {res_lb_greedy['normalized_makespan']:7.4f} | Imbalance: {res_lb_greedy['avg_pick_imbalance']:7.2f} | "
                    f"Time: {res_lb_greedy['total_time']:6.2f}s"
                )
            except Exception as e:
                print(f"    - {'6. Load-balanced Greedy (SGC)':<32} | FAILED: {e}")

        detailed_history.append(inst_record)

    # Print summary table
    print_comparison_table(results_by_method)

    # Save to JSON
    if args.output_json:
        output_payload = {
            "meta": {
                "num_instances": args.num_instances,
                "seed_base": args.seed_base,
                "device": str(device),
                "deterministic": not args.stochastic,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "summary": {
                method: {
                    metric: sum(r[metric] for r in res_list) / len(res_list)
                    for metric in res_list[0].keys()
                }
                for method, res_list in results_by_method.items()
                if res_list
            },
            "detailed_history": detailed_history,
        }
        with open(args.output_json, "w") as f:
            json.dump(output_payload, f, indent=2)
        print(f"[+] Saved detailed benchmark results to '{args.output_json}'")


if __name__ == "__main__":
    main()
