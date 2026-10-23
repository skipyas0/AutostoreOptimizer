#!/usr/bin/env python3
"""
Benchmark heuristics (and RL order-to-station assignment checkpoints) on instances
drawn from the RL training distribution, with disjoint seeds.

Every method is a ``Callable[[Instance], HeuristicSolution]``; metrics are computed
from the returned solution by ``solution_metrics`` (RL-environment reward components
plus timings). Default methods (same labels as before the heuristics refactor):

  RL (<checkpoint>)                  RL assignment + per-station RDI
  1. Random-partitioned RDI          heuristics: partitioned_rdi_random
  2. Similarity-partitioned RDI      heuristics: partitioned_rdi_similarity
  3. Load-balanced RDI               heuristics: partitioned_rdi_load_balanced
  4. Full RDI (No Decomposition)     heuristics: rdi
  5. Standard Full Greedy (SGC)      heuristics: sgc
  6. Load-balanced Greedy (SGC)      heuristics: lb_sgc
  7. Standard Piggyback SGC          heuristics: sgc_singles
  8. Load-balanced Piggyback SGC     heuristics: lb_sgc_singles

``--methods`` selects a subset by label number/registry name, or ``all`` to add
EXTRA_METHODS (RDI tiebreakers, AMA, CFSS). All heuristics use horizon 50000.

Usage:
  .venv/bin/python benchmark_models.py
  .venv/bin/python benchmark_models.py --checkpoints rl_agent_weights_mlp_1200.pth --num-instances 10
  .venv/bin/python benchmark_models.py --methods sgc rdi cfss ama_two_phase --num-instances 5
  .venv/bin/python benchmark_models.py --methods all --skip-rl
"""

import argparse
import glob
import json
import os
import random
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from functools import partial

# Suppress matplotlib font cache warnings
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_config")
import logging

logging.getLogger("matplotlib").setLevel(logging.ERROR)

import torch

# Ensure code root is in path
code_dir = os.path.dirname(os.path.abspath(__file__))
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from datagen import generate_data
from heuristics import (
    HeuristicSolution,
    resolve,
    run_heuristic,
    run_partitioned_rdi,
    validate_solution,
)
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


def generate_benchmark_instance(
    instance_idx: int, scale, seed_base: int = 100000
) -> Instance:
    """
    Generates an instance matching the training loop distribution in rl_order_assignment.py
    using a disjoint seed range.
    """
    rng = random.Random(seed_base + instance_idx)

    preset = {
        "small": (4, 3, 10, 1, 500),
        "medium": (6, 5, 35, 3, 1000),
        "large": (15, 8, 75, 8, 2500),
    }
    max_s, max_l, o_mul, m_mul, sku_mul = preset[scale]

    s = rng.randint(2, max_s)
    l = rng.randint(2, max_l)
    o = rng.randint(2, 8) * o_mul
    m = rng.randint(4, 8) * m_mul
    sku = rng.randint(8, 15) * sku_mul

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
    M_k: list[float],
    W_k: list[float],
    Be_k: list[int],
    P_k: list[int],
    Rt_k: list[float],
    elapsed_time: float,
    partition_time: float = 0.0,
) -> dict[str, float]:
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


# ======================================================================
# Metrics from a HeuristicSolution
# ======================================================================

PENALTY_MAKESPAN = 50000.0


def solution_metrics(
    instance: Instance,
    sol: HeuristicSolution | None,
    elapsed: float,
    partition_time: float = 0.0,
) -> dict[str, float]:
    """Per-station components (M_k, W_k, Be_k, P_k, Rt_k) -> RL-environment metrics.

    Monolithic solutions: if infeasible, every station gets M_k = 50000 and W_k = 0.
    Partitioned solutions (meta["station_feasible"]): only infeasible stations are
    penalised and W_k is always the assigned workload (meta["assignment"]).
    """
    M_k, W_k, Be_k, P_k, Rt_k = [], [], [], [], []
    station_feasible = sol.meta.get("station_feasible") if sol is not None else None
    assignment = sol.meta.get("assignment") if sol is not None else None

    if station_feasible is None and (sol is None or not sol.feasible):
        n = len(instance.S)
        return compute_metrics_from_station_results(
            instance,
            [PENALTY_MAKESPAN] * n,
            [0.0] * n,
            [0] * n,
            [0] * n,
            [0.0] * n,
            elapsed,
            partition_time,
        )

    for s in instance.S:
        if assignment is not None:
            orders = assignment.get(s, [])
        else:
            orders = [
                o for o, (st, _, _, _) in sol.order_assignments.items() if st == s
            ]
        W_k.append(
            float(
                sum(
                    instance.p.get(k, 0) for o in orders for k in instance.orders_req[o]
                )
            )
        )

        if station_feasible is not None and not station_feasible.get(s, True):
            M_k.append(PENALTY_MAKESPAN)
            Be_k.append(0)
            P_k.append(0)
            Rt_k.append(0.0)
            continue

        M_k.append(
            float(
                max(
                    (
                        en
                        for (st, _, _, en) in sol.order_assignments.values()
                        if st == s
                    ),
                    default=0.0,
                )
            )
        )
        events = sol.bin_events.get(s, [])
        Be_k.append(len(events))
        P_k.append(sum(1 for (_, st, _) in sol.pick_events if st == s))
        Rt_k.append(
            float(
                sum(
                    be.fetch_end - be.fetch_start + be.return_end - be.return_start
                    for be in events
                )
            )
        )

    return compute_metrics_from_station_results(
        instance, M_k, W_k, Be_k, P_k, Rt_k, elapsed, partition_time
    )


# ======================================================================
# Methods: Instance -> HeuristicSolution
# ======================================================================

BENCH_HORIZON = 50000

DEFAULT_METHODS: dict[str, Callable[[Instance], HeuristicSolution]] = {
    "1. Random-partitioned RDI": partial(
        run_heuristic, "partitioned_rdi_random", horizon=BENCH_HORIZON
    ),
    "2. Similarity-partitioned RDI": partial(
        run_heuristic, "partitioned_rdi_similarity", horizon=BENCH_HORIZON
    ),
    "3. Load-balanced RDI": partial(
        run_heuristic, "partitioned_rdi_load_balanced", horizon=BENCH_HORIZON
    ),
    "4. Full RDI (No Decomposition)": partial(
        run_heuristic, "rdi", horizon=BENCH_HORIZON
    ),
    "5. Standard Full Greedy (SGC)": partial(
        run_heuristic, "sgc", horizon=BENCH_HORIZON
    ),
    "6. Load-balanced Greedy (SGC)": partial(
        run_heuristic, "lb_sgc", horizon=BENCH_HORIZON
    ),
    "7. Standard Piggyback SGC": partial(
        run_heuristic, "sgc_singles", horizon=BENCH_HORIZON
    ),
    "8. Load-balanced Piggyback SGC": partial(
        run_heuristic, "lb_sgc_singles", horizon=BENCH_HORIZON
    ),
}

# Extra registry heuristics added by `--methods all` (or by name).
EXTRA_METHODS = [
    "rdi_sum_rt",
    "rdi_best_score",
    "ama_two_phase",
    "ama_two_phase_singles",
    "ama",
    "cfss",
    "cfss_normalized",
]


def registry_method(name: str) -> Callable[[Instance], HeuristicSolution]:
    key = resolve(name)
    horizon = {"horizon": BENCH_HORIZON}
    return partial(run_heuristic, key, **horizon)


def rl_method(
    agent, device: torch.device, deterministic: bool = True
) -> Callable[[Instance], HeuristicSolution]:
    """RL order->station assignment followed by per-station RDI (partitioned RDI)."""

    def run(instance: Instance) -> HeuristicSolution:
        t0 = time.perf_counter()
        assignment = rl_assignment(instance, agent, device, deterministic)
        t_partition = time.perf_counter() - t0
        sol = run_partitioned_rdi(instance, partition=assignment, horizon=BENCH_HORIZON)
        sol.meta["partition_time"] = t_partition
        return sol

    return run


def run_method(
    method: Callable[[Instance], HeuristicSolution], instance: Instance
) -> dict[str, float]:
    t0 = time.perf_counter()
    sol = method(instance)
    elapsed = time.perf_counter() - t0
    partition_time = sol.meta.get("partition_time", 0.0) if sol is not None else 0.0
    return solution_metrics(instance, sol, elapsed, partition_time)


def rl_assignment(
    instance: Instance, agent, device: torch.device, deterministic: bool = True
) -> dict[int, list[int]]:
    """{station_index: [orders]} chosen by an RL checkpoint."""
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

    station_assignments = {s: [] for s in range(len(instance.S))}
    for o_idx, s_idx in enumerate(actions):
        station_assignments[min(s_idx, len(instance.S) - 1)].append(instance.O[o_idx])
    return station_assignments


def print_comparison_table(results_by_method: dict[str, list[dict[str, float]]]):
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
        help="Skip the Standard, Load-balanced and Piggyback SGC methods (5-8).",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Subset of methods: default labels' leading numbers (1-6), registry names "
        "(e.g. sgc rdi cfss ama_two_phase), or 'all' (defaults + EXTRA_METHODS). Default: 1-8.",
    )
    parser.add_argument(
        "--scale",
        type=str,
        default="small",
        help="Scale for the benchmark instances (small, medium, large).",
    )
    parser.add_argument(
        "--skip-rl", action="store_true", help="Do not evaluate RL checkpoints."
    )
    parser.add_argument(
        "--skip-rdi",
        action="store_true",
        help="Skip every RDI-based method: partitioned RDI (1-3), Full RDI (4), "
        "RDI registry variants (e.g. from --methods/all), and RL checkpoints "
        "(they decompose with partitioned RDI).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="benchmark_results.json",
        help="Path to save detailed benchmark results as JSON.",
    )
    return parser.parse_args()


def is_rdi_method(name: str) -> bool:
    """True for methods built on RDI: labels 1-4 ("... RDI ...") and registry rdi* / partitioned_rdi* names."""
    return "rdi" in name.lower()


def select_methods(args) -> dict[str, Callable[[Instance], HeuristicSolution]]:
    methods: dict[str, Callable[[Instance], HeuristicSolution]] = {}
    wanted = args.methods
    if not wanted:
        methods = dict(DEFAULT_METHODS)
    else:
        for w in wanted:
            if w == "all":
                methods.update(DEFAULT_METHODS)
                for name in EXTRA_METHODS:
                    methods[name] = registry_method(name)
                continue
            label = next(
                (lab for lab in DEFAULT_METHODS if lab.split(".")[0] == w), None
            )
            if label:
                methods[label] = DEFAULT_METHODS[label]
            else:
                methods[resolve(w)] = registry_method(w)
    if args.skip_full_rdi:
        methods.pop("4. Full RDI (No Decomposition)", None)
    if args.skip_rdi:
        methods = {k: v for k, v in methods.items() if not is_rdi_method(k)}
    if args.skip_greedy:
        for label in (
            "5. Standard Full Greedy (SGC)",
            "6. Load-balanced Greedy (SGC)",
            "7. Standard Piggyback SGC",
            "8. Load-balanced Piggyback SGC",
        ):
            methods.pop(label, None)
    return methods


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else get_default_device()
    print(f"[*] Running benchmark on PyTorch device: {device}")

    methods: dict[str, Callable[[Instance], HeuristicSolution]] = {}
    if not args.skip_rl and not args.skip_rdi:
        checkpoints = args.checkpoints or sorted(glob.glob("rl_agent_weights*.pth"))
        if not checkpoints:
            print("[!] Warning: No .pth files found in current directory.")
        for cp_path in checkpoints:
            if not os.path.exists(cp_path):
                print(
                    f"[!] Warning: Checkpoint file '{cp_path}' does not exist. Skipping."
                )
                continue
            short_name = os.path.basename(cp_path).replace(".pth", "")
            try:
                agent = load_agent(cp_path, device=device)
                methods[f"RL ({short_name})"] = rl_method(
                    agent, device, deterministic=not args.stochastic
                )
                print(f"[+] Successfully loaded checkpoint: {cp_path}")
            except Exception as e:
                print(f"[!] Failed to load checkpoint {cp_path}: {e}")
    methods.update(select_methods(args))

    print(
        f"\n[*] Evaluating {len(methods)} method(s) across {args.num_instances} instance(s)..."
    )
    results_by_method: dict[str, list[dict[str, float]]] = defaultdict(list)
    detailed_history = []

    for inst_idx in range(args.num_instances):
        print(
            f"\n>>> Instance {inst_idx + 1:02d}/{args.num_instances} (Seed: {args.seed_base + inst_idx})"
        )
        instance = generate_benchmark_instance(
            inst_idx, args.scale, seed_base=args.seed_base
        )
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

        for label, method in methods.items():
            try:
                t0 = time.perf_counter()
                sol = method(instance)
                elapsed = time.perf_counter() - t0
                part = sol.meta.get("partition_time", 0.0) if sol is not None else 0.0
                res = solution_metrics(instance, sol, elapsed, part)
                res["n_violations"] = (
                    len(validate_solution(sol, instance, horizon=BENCH_HORIZON))
                    if sol is not None
                    else -1
                )
                results_by_method[label].append(res)
                inst_record["methods"][label] = res
                print(
                    f"    - {label:<32} | Makespan: {res['makespan']:6.0f} | "
                    f"Norm M: {res['normalized_makespan']:7.4f} | Imbalance: {res['avg_pick_imbalance']:7.2f} | "
                    f"Violations: {res['n_violations']:3d} | Time: {res['total_time']:6.2f}s"
                )
            except Exception as e:
                print(f"    - {label:<32} | FAILED: {e}")

        detailed_history.append(inst_record)

    print_comparison_table(results_by_method)

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
