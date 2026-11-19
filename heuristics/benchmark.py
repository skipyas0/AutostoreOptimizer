"""Benchmark registered heuristics (and, optionally, RL order->station checkpoints).

    python -m heuristics benchmark                                   # default set, preset "small"
    python -m heuristics benchmark --methods all --skip '*rdi*'
    python -m heuristics benchmark --methods '*sgc*' --skip '*singles*' --preset medium -n 20
    python -m heuristics benchmark --methods 'gbs_*' lb_sgc_share precalculated_instances/
    python -m heuristics benchmark --preset large --stations 10 --orders-per-station 30-50
    python -m heuristics benchmark --checkpoints rl_agent_weights_mlp_1200.pth --preset rl-small
    python -m heuristics benchmark --methods all --skip 'ama*' 'cfss*' --list      # dry run

Method selection
----------------
Methods are named by their registry key (``python -m heuristics --list``); RL checkpoints
are named ``rl:<checkpoint stem>``. ``--methods`` and ``--skip`` take any number of tokens
(also comma-separated):

  default     the standard comparison set (DEFAULT_METHODS) + loaded RL checkpoints
  all         every registered heuristic + loaded RL checkpoints
  <pattern>   shell-style glob over method names: '*rdi*', 'gbs_*', 'rl:*', 'lb_?gc*'
  <name>      exact registry name or alias (e.g. full_rdi -> rdi)

The selection is the union of ``--methods`` minus everything matching ``--skip``, in
registry order. Quote glob patterns so the shell does not expand them.

Instances
---------
* Positional paths: pregenerated instances -- a precalculated folder (with instance.pkl),
  an instance.pkl, or a directory of such folders (e.g. ``precalculated_instances/``).
  The horizon comes from each folder's config.json when present.
* Otherwise ``-n`` instances are generated from ``--preset`` (see PRESETS) with seeds
  ``seed_base + i``. Any preset parameter can be overridden from the CLI with a fixed
  value (``5``), an integer range (``4-8``) or a choice list (``5,10,15``).
  ``--save-instances DIR`` writes them as precalculated-style folders for later reuse.

Metrics: the RL-environment reward components (``solution_metrics``), robot work / fetches,
robot and picker utilisation, station-makespan balance and picker-schedule gaps
(``schedule_metrics``; ``--gap-threshold`` sets what counts as a long pause), gaps to the
best method per instance, timings and the validator's violation count. ``--output-json``
stores the per-instance details.
"""

from __future__ import annotations

import argparse
import fnmatch
import itertools
import math
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from heuristics import (
    ALIASES,
    HEURISTICS,
    HeuristicSolution,
    resolve,
    run_heuristic,
    run_partitioned_rdi,
    validate_solution,
)
from instance import Instance

DEFAULT_HORIZON = 10000

Method = Callable[[Instance, int], HeuristicSolution]  # (instance, horizon) -> solution

# The standard comparison set (former benchmark_models.py methods 1-14, same order).
DEFAULT_METHODS = [
    "partitioned_rdi_random",
    "partitioned_rdi_similarity",
    "partitioned_rdi_load_balanced",
    "rdi",
    "sgc",
    "lb_sgc",
    "sgc_singles",
    "lb_sgc_singles",
    "sgc_share",
    "lb_sgc_share",
    "rdi_share",
    "partitioned_rdi_load_balanced_share",
    "rdi_lb",
    "rdi_lb_share",
]


# ======================================================================
# Instance generation
# ======================================================================

# A parameter spec is a fixed value, a (lo, hi) inclusive range (ints -> randint, floats ->
# uniform) or a list of choices.
Spec = int | float | tuple | list

# Fixed datagen settings shared by all generated benchmark instances (same as the RL
# training distribution and the precalculated LNS instances).
DATAGEN_FIXED = dict(
    pick_touch_time=4,
    return_model="balancing",
    max_bins_per_sku=8,
    retrieval_time_model="depth_based",
    pick_time_model="variable",
    bin_count_model="popularity_correlated",
)


@dataclass
class Preset:
    """Sampling distribution of benchmark instances.

    Workload and robot fleet scale with the number of stations: unless ``orders`` /
    ``movecap`` are given explicitly, |O| = stations * orders_per_station and
    MoveCap = round(stations * robots_per_station).
    """

    description: str
    stations: Spec
    lanes: Spec
    orders_per_station: Spec = (20, 40)
    robots_per_station: Spec = (3.0, 6.0)
    skus: Spec = (5000, 20000)
    skew: Spec = (0.95, 1.05)
    grid_depth: Spec = (14, 18)
    orders: Spec | None = None
    movecap: Spec | None = None
    # cycle through these fixed combinations (instance i uses cycle[i % len], seed_base + i // len)
    cycle: list[dict] = field(default_factory=list)


PRESETS: dict[str, Preset] = {
    "tiny": Preset(
        "2-3 stations, ~10-20 orders/station; quick smoke runs",
        stations=(2, 3), lanes=(2, 3), orders_per_station=(10, 20), robots_per_station=(3.0, 5.0),
        skus=(1000, 4000),
    ),
    "small": Preset(
        "2-4 stations, 20-40 orders/station, 3-6 robots/station",
        stations=(2, 4), lanes=(2, 3), orders_per_station=(20, 40), skus=(4000, 10000),
    ),
    "medium": Preset(
        "4-8 stations, 30-50 orders/station, 3-6 robots/station",
        stations=(4, 8), lanes=(3, 4), orders_per_station=(30, 50), skus=(10000, 25000),
    ),
    "large": Preset(
        "8-15 stations, 40-60 orders/station, 2.5-5 robots/station",
        stations=(8, 15), lanes=(3, 4), orders_per_station=(40, 60), robots_per_station=(2.5, 5.0),
        skus=(20000, 35000),
    ),
    "lns": Preset(
        "the precalculated LNS instances GEN<seed>-{5,10,15}-3-28750-{300,400,600}-40 "
        "(use --seed-base 1 to reproduce GEN1..)",
        stations=None, lanes=3, skus=28750, skew=1.0, grid_depth=16, movecap=40,
        cycle=[{"stations": 5, "orders": 300}, {"stations": 10, "orders": 400}, {"stations": 15, "orders": 600}],
    ),
}

# Legacy presets: exactly the old benchmark_models.py generator (RL training distribution:
# 400 orders and MoveCap 40 regardless of the number of stations). Kept for RL checkpoints
# and for regression tests that pin specific instances.
LEGACY_PRESETS = {
    "rl-small": (4, 3, 500),
    "rl-medium": (6, 5, 1000),
    "rl-large": (15, 8, 2500),
}

PRESET_PARAMS = ("stations", "lanes", "orders", "orders_per_station", "movecap",
                 "robots_per_station", "skus", "skew", "grid_depth")


def parse_spec(text: str) -> Spec:
    """'5' -> 5, '4-8' -> (4, 8), '2.5-5' -> (2.5, 5.0), '5,10,15' -> [5, 10, 15]."""
    num = lambda s: float(s) if "." in s else int(s)  # noqa: E731
    text = text.strip()
    if "," in text:
        return [num(t) for t in text.split(",") if t]
    if "-" in text:
        lo, hi = text.split("-", 1)
        return (num(lo), num(hi))
    return num(text)


def sample(spec: Spec, rng: random.Random):
    if isinstance(spec, list):
        return rng.choice(spec)
    if isinstance(spec, tuple):
        lo, hi = spec
        if isinstance(lo, int) and isinstance(hi, int):
            return rng.randint(lo, hi)
        return rng.uniform(lo, hi)
    return spec


def _legacy_instance(name: str, idx: int, seed_base: int) -> tuple[Instance, dict]:
    max_s, max_l, sku_mul = LEGACY_PRESETS[name]
    seed = seed_base + idx
    rng = random.Random(seed)
    s = rng.randint(2, max_s)
    l = rng.randint(2, max_l)  # noqa: E741
    sku = rng.randint(8, 15) * sku_mul
    cfg = dict(stations=s, lanes=l, orders=400, skus=sku, movecap=40,
               skew=1.0 + 0.1 * (rng.random() - 0.5), grid_depth=16 + rng.randint(-2, 2), gen_seed=seed)
    return _generate(cfg), cfg


def _generate(cfg: dict) -> Instance:
    from datagen import generate_data

    return generate_data(
        num_stations=cfg["stations"],
        lanes_per_station=cfg["lanes"],
        num_orders=cfg["orders"],
        num_skus=cfg["skus"],
        seed=cfg["gen_seed"],
        movecap=cfg["movecap"],
        sku_popularity_skew=cfg["skew"],
        grid_depth=cfg["grid_depth"],
        **DATAGEN_FIXED,
    )


def generate_instance(preset: str, idx: int, seed_base: int = 100000,
                      overrides: dict[str, Spec] | None = None) -> tuple[Instance, dict]:
    """Instance number ``idx`` of a preset -> (instance, generation config)."""
    if preset in LEGACY_PRESETS:
        if overrides:
            raise ValueError(f"preset {preset!r} reproduces the legacy generator and takes no overrides")
        return _legacy_instance(preset, idx, seed_base)
    p = PRESETS[preset]
    spec = {k: getattr(p, k) for k in PRESET_PARAMS}
    seed = seed_base + idx
    if p.cycle:
        spec.update(p.cycle[idx % len(p.cycle)])
        seed = seed_base + idx // len(p.cycle)
    spec.update(overrides or {})

    rng = random.Random(seed)
    cfg = {k: sample(spec[k], rng) for k in ("stations", "lanes", "skus", "skew", "grid_depth")}
    s = cfg["stations"]
    cfg["orders"] = (sample(spec["orders"], rng) if spec["orders"] is not None
                     else s * sample(spec["orders_per_station"], rng))
    cfg["movecap"] = (sample(spec["movecap"], rng) if spec["movecap"] is not None
                      else max(1, round(s * sample(spec["robots_per_station"], rng))))
    cfg["gen_seed"] = seed
    return _generate(cfg), cfg


@dataclass
class BenchInstance:
    name: str
    instance: Instance
    horizon: int
    config: dict


def generated_instances(args, overrides) -> list[BenchInstance]:
    out = []
    for i in range(args.num_instances):
        inst, cfg = generate_instance(args.preset, i, args.seed_base, overrides)
        name = (f"{args.preset}-{i:02d}-S{cfg['stations']}-L{cfg['lanes']}-O{cfg['orders']}"
                f"-M{cfg['movecap']}-seed{cfg['gen_seed']}")
        out.append(BenchInstance(name, inst, args.horizon or DEFAULT_HORIZON, cfg))
    return out


def _instance_pkls(paths: list[str]) -> list[Path]:
    """Folders with instance.pkl, .pkl files, or directories of such folders (one level)."""
    found = []
    for p in map(Path, paths):
        if p.is_file() and p.suffix == ".pkl":
            found.append(p)
        elif (p / "instance.pkl").is_file():
            found.append(p / "instance.pkl")
        elif p.is_dir():
            sub = sorted(c / "instance.pkl" for c in p.iterdir() if (c / "instance.pkl").is_file())
            if not sub:
                print(f"[!] No instance.pkl in {p} or its subfolders, skipping")
            found.extend(sub)
        else:
            print(f"[!] Not an instance: {p}, skipping")
    return found


def loaded_instances(paths: list[str], horizon: int | None) -> list[BenchInstance]:
    out = []
    for pkl in _instance_pkls(paths):
        cfg_file = pkl.parent / "config.json"
        cfg = json.loads(cfg_file.read_text()) if cfg_file.is_file() else {}
        name = pkl.parent.name if pkl.name == "instance.pkl" else pkl.stem
        h = horizon or cfg.get("horizon", DEFAULT_HORIZON)
        out.append(BenchInstance(name, Instance.from_pickle(str(pkl)), h, cfg))
    return out


def save_instances(instances: list[BenchInstance], out_dir: str) -> None:
    for bi in instances:
        d = Path(out_dir) / bi.name
        d.mkdir(parents=True, exist_ok=True)
        bi.instance.to_pickle(str(d / "instance.pkl"))
        cfg = {**bi.config, "horizon": bi.horizon, **DATAGEN_FIXED}
        (d / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"[+] Saved {len(instances)} instance(s) to {out_dir}")


# ======================================================================
# Metrics from a HeuristicSolution
# ======================================================================

PENALTY_MAKESPAN = 50000.0


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
    """Reward and summary metrics as defined in the RL environment."""
    num_stations = len(instance.S)

    # upper bound: robot fetch + return limit
    upper_bound = sum(
        instance.rt.get(k, 0) + instance.rt_ret.get(k, 0) for o in instance.O for k in instance.orders_req[o]
    ) / max(instance.movecap, 1)
    # lower bound: picker saturation bound
    lower_bound = sum(instance.p.get(k, 0) for o in instance.O for k in instance.orders_req[o]) / max(num_stations, 1)
    scale_estimate = max((upper_bound + lower_bound) / 2, 1e-6)

    M = max(M_k) if M_k else 0.0
    avg_W = sum(W_k) / len(W_k) if W_k else 0.0
    avg_M = sum(M_k) / len(M_k) if M_k else 0.0
    min_M = min(M_k) if M_k else 0.0

    total_rt = sum(Rt_k)
    total_be = sum(Be_k)
    total_p = sum(P_k)

    movecap_utilization = total_rt / (M * instance.movecap) if (M > 0 and instance.movecap > 0) else 0.0
    avg_pick_imbalance = sum(abs(w - avg_W) for w in W_k) / num_stations if num_stations > 0 else 0.0
    shift_length_coeff = (avg_M + num_stations * min_M) / M if M > 0 else 0.0
    picks_from_bin_presence = (total_p / total_be - 1.0) if total_be > 0 else 0.0
    normalized_M = M / scale_estimate

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


def solution_metrics(
    instance: Instance,
    sol: HeuristicSolution | None,
    elapsed: float,
    partition_time: float = 0.0,
) -> dict[str, float]:
    """Per-station components (M_k, W_k, Be_k, P_k, Rt_k) -> RL-environment metrics.

    W_k is always the picking workload of the orders assigned to station k (from
    meta["assignment"] for partitioned solutions, else from the schedule). An
    infeasible station -- every station of an infeasible monolithic solution, or the
    infeasible ones of a partitioned solution (meta["station_feasible"]) -- gets
    M_k = PENALTY_MAKESPAN and Be_k = P_k = Rt_k = 0.
    """
    M_k, W_k, Be_k, P_k, Rt_k = [], [], [], [], []
    station_feasible = sol.meta.get("station_feasible") if sol is not None else None
    assignment = sol.meta.get("assignment") if sol is not None else None

    for s in instance.S:
        if assignment is not None:
            orders = assignment.get(s, [])
        elif sol is not None:
            orders = [o for o, (st, _, _, _) in sol.order_assignments.items() if st == s]
        else:
            orders = []
        W_k.append(float(sum(instance.p.get(k, 0) for o in orders for k in instance.orders_req[o])))

        ok = station_feasible.get(s, True) if station_feasible is not None else (sol is not None and sol.feasible)
        if not ok:
            M_k.append(PENALTY_MAKESPAN)
            Be_k.append(0)
            P_k.append(0)
            Rt_k.append(0.0)
            continue

        M_k.append(float(max((en for (st, _, _, en) in sol.order_assignments.values() if st == s), default=0.0)))
        events = sol.bin_events.get(s, [])
        Be_k.append(len(events))
        P_k.append(sum(1 for (_, st, _) in sol.pick_events if st == s))
        Rt_k.append(float(sum(be.fetch_end - be.fetch_start + be.return_end - be.return_start for be in events)))

    return compute_metrics_from_station_results(instance, M_k, W_k, Be_k, P_k, Rt_k, elapsed, partition_time)


DEFAULT_GAP_THRESHOLD = 60  # s; a picker pause longer than this counts as a "long" gap


def _union(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


SCHEDULE_METRIC_KEYS = (
    "robot_work", "n_fetches", "robot_work_rel", "picker_utilization", "station_makespan_spread",
    "station_makespan_cv", "picker_idle_frac", "picker_max_gap", "picker_long_gap_frac", "picker_long_gaps",
)


def schedule_metrics(instance: Instance, sol: HeuristicSolution | None,
                     gap_threshold: float = DEFAULT_GAP_THRESHOLD) -> dict[str, float]:
    """Robot work, utilisation, station balance and picker-schedule "holeyness".

    Computed over the feasible stations only (see ``solution_metrics`` for the penalised
    view); all NaN if no station is feasible. A station's picker is busy while at least one
    pick runs there (lanes pick in parallel); its *span* runs from the first pick start to
    the last pick end, and a *gap* is a maximal idle interval inside the span.

      robot_work              sum of fetch + return durations (robot-seconds)
      n_fetches               number of bin visits
      robot_work_rel          robot_work / robot work without any bin sharing (1 fetch per order line)
      picker_utilization      mean over stations of busy time / makespan C
      station_makespan_spread (C - min_s C_s) / C, C_s = completion of station s's last order
      station_makespan_cv     coefficient of variation of the C_s
      picker_idle_frac        idle time inside spans / total span
      picker_max_gap          longest single picker pause (s)
      picker_long_gap_frac    idle time in gaps > gap_threshold / total span
      picker_long_gaps        number of gaps > gap_threshold per station
    """
    nan = {k: float("nan") for k in SCHEDULE_METRIC_KEYS}
    if sol is None:
        return nan
    station_feasible = sol.meta.get("station_feasible")
    stations = [s for s in instance.S
                if (station_feasible.get(s, True) if station_feasible is not None else sol.feasible)]
    if not stations:
        return nan

    picks_by_station: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for (_, s, _), iv in sol.pick_events.items():
        picks_by_station[s].append(iv)
    C_s = [max((en for (st, _, _, en) in sol.order_assignments.values() if st == s), default=0) for s in stations]
    C = max(C_s)

    robot_work, n_fetches = 0, 0
    busy_frac, span_total, idle_total, long_idle, n_long, max_gap = [], 0, 0, 0, 0, 0
    for s in stations:
        events = sol.bin_events.get(s, [])
        n_fetches += len(events)
        robot_work += sum(be.fetch_end - be.fetch_start + be.return_end - be.return_start for be in events)
        busy = _union(picks_by_station.get(s, []))
        if not busy:
            busy_frac.append(0.0)
            continue
        busy_frac.append(sum(b - a for a, b in busy) / C if C > 0 else 0.0)
        span_total += busy[-1][1] - busy[0][0]
        for (_, b), (a, _) in itertools.pairwise(busy):
            gap = a - b
            idle_total += gap
            max_gap = max(max_gap, gap)
            if gap > gap_threshold:
                long_idle += gap
                n_long += 1

    no_share = sum(instance.rt.get(k, 0) + instance.rt_ret.get(k, 0) for o in instance.O for k in instance.orders_req[o])
    mean_c = sum(C_s) / len(C_s)
    std_c = (sum((c - mean_c) ** 2 for c in C_s) / len(C_s)) ** 0.5
    return {
        "robot_work": float(robot_work),
        "n_fetches": float(n_fetches),
        "robot_work_rel": robot_work / no_share if no_share else float("nan"),
        "picker_utilization": sum(busy_frac) / len(busy_frac),
        "station_makespan_spread": (C - min(C_s)) / C if C > 0 else 0.0,
        "station_makespan_cv": std_c / mean_c if mean_c > 0 else 0.0,
        "picker_idle_frac": idle_total / span_total if span_total else 0.0,
        "picker_max_gap": float(max_gap),
        "picker_long_gap_frac": long_idle / span_total if span_total else 0.0,
        "picker_long_gaps": n_long / len(stations),
    }


# ======================================================================
# Methods
# ======================================================================


def heuristic_method(name: str) -> Method:
    key = resolve(name)
    return lambda inst, horizon: run_heuristic(key, inst, horizon=horizon)


def run_method(method: Method, instance: Instance, horizon: int = DEFAULT_HORIZON) -> dict[str, float]:
    t0 = time.perf_counter()
    sol = method(instance, horizon)
    elapsed = time.perf_counter() - t0
    partition_time = sol.meta.get("partition_time", 0.0) if sol is not None else 0.0
    return solution_metrics(instance, sol, elapsed, partition_time)


# ---- RL checkpoints (torch is imported only when checkpoints are requested) ----


def _default_device():
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_agent(checkpoint_path: str, device, hidden_dim: int = 64, max_stations: int = 20, mlp_hidden: int = 128):
    import torch
    from rl_order_assignment import OrderAssignmentAgent, OrderGraphGNN, StationOrderScorer

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if any(k.startswith("net.") for k in checkpoint.keys()):
        feat_dim = checkpoint["net.0.weight"].shape[1]
        hid_dim = checkpoint["net.0.weight"].shape[0]
        model = StationOrderScorer(feature_dim=feat_dim, hidden_dim=hid_dim).to(device)
    else:  # legacy GNN OrderAssignmentAgent
        gnn = OrderGraphGNN(hidden_dim=hidden_dim, num_layers=2)
        model = OrderAssignmentAgent(gnn, hidden_dim=hidden_dim, max_stations=max_stations,
                                     mlp_hidden=mlp_hidden).to(device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def rl_assignment(instance: Instance, agent, device, deterministic: bool = True) -> dict[int, list[int]]:
    """{station_index: [orders]} chosen by an RL checkpoint."""
    import torch
    from rl_order_assignment import StationOrderScorer, precompute_instance_similarities, rollout_station_order

    if isinstance(agent, StationOrderScorer):
        sim_unw, sim_w = precompute_instance_similarities(instance)
        with torch.no_grad():
            actions, _, _, _ = rollout_station_order(instance, agent, sim_unw, sim_w, device,
                                                     deterministic=deterministic)
    else:
        hetero_data = instance.create_order_graph().to(device)
        instance_stats = {
            "num_stations": len(instance.S),
            "total_used_skus": instance.stats["used_skus"],
            "movecap": instance.movecap,
        }
        order_num_skus = torch.tensor([len(instance.orders_req[o]) for o in instance.O],
                                      dtype=torch.float, device=device)
        with torch.no_grad():
            actions, _, _ = agent(hetero_data, instance_stats, order_num_skus, deterministic=deterministic)

    station_assignments = {s: [] for s in range(len(instance.S))}
    for o_idx, s_idx in enumerate(actions):
        station_assignments[min(s_idx, len(instance.S) - 1)].append(instance.O[o_idx])
    return station_assignments


def rl_method(agent, device, deterministic: bool = True) -> Method:
    """RL order->station assignment followed by per-station RDI (partitioned RDI)."""

    def run(instance: Instance, horizon: int) -> HeuristicSolution:
        t0 = time.perf_counter()
        assignment = rl_assignment(instance, agent, device, deterministic)
        t_partition = time.perf_counter() - t0
        sol = run_partitioned_rdi(instance, partition=assignment, horizon=horizon)
        sol.meta["partition_time"] = t_partition
        return sol

    return run


def checkpoint_paths(checkpoints: list[str]) -> list[str]:
    """Given paths, or rl_agent_weights*.pth in the current directory when none are given."""
    paths = checkpoints or sorted(str(p) for p in Path().glob("rl_agent_weights*.pth"))
    if not paths:
        print("[!] --checkpoints given but no rl_agent_weights*.pth found in the current directory")
    return paths


def rl_methods(checkpoints: list[str], device_name: str | None, deterministic: bool) -> dict[str, Method]:
    import torch

    device = torch.device(device_name) if device_name else _default_device()
    out = {}
    for cp in checkpoint_paths(checkpoints):
        try:
            out[f"rl:{Path(cp).stem}"] = rl_method(load_agent(cp, device), device, deterministic)
            print(f"[+] Loaded checkpoint {cp} on {device}")
        except Exception as e:
            print(f"[!] Failed to load checkpoint {cp}: {type(e).__name__}: {e}")
    return out


# ======================================================================
# Method selection
# ======================================================================


def _tokens(values: list[str] | None) -> list[str]:
    return [t.strip() for v in values or [] for t in v.split(",") if t.strip()]


def _match(token: str, universe: list[str], groups: dict[str, list[str]]) -> list[str]:
    if token in groups:
        return groups[token]
    if any(c in token for c in "*?["):
        return [n for n in universe if fnmatch.fnmatchcase(n, token)]
    if token in universe:
        return [token]
    key = token.lower().replace("-", "_")
    key = ALIASES.get(key, key)
    return [key] if key in universe else []


def select_methods(include: list[str] | None, skip: list[str] | None, rl_names: list[str] = ()) -> list[str]:
    """Names selected by --methods (default: 'default') minus those matching --skip.

    Raises ValueError for a --methods token that matches nothing (likely a typo).
    """
    rl_names = list(rl_names)
    universe = rl_names + list(HEURISTICS)
    groups = {"default": rl_names + DEFAULT_METHODS, "all": universe}

    chosen: set[str] = set()
    for tok in _tokens(include) or ["default"]:
        hits = _match(tok, universe, groups)
        if not hits:
            raise ValueError(f"--methods {tok!r} matches no method. Available: {', '.join(universe)}")
        chosen.update(hits)
    for tok in _tokens(skip):
        hits = _match(tok, universe, groups)
        if not hits:
            print(f"[!] --skip {tok!r} matches no method")
        chosen.difference_update(hits)
    return [n for n in universe if n in chosen]


# ======================================================================
# Reporting
# ======================================================================

# (header, metric key, format); one summary table per group
TABLES = {
    "Quality": [
        ("Makespan", "makespan", "{:.1f}"),
        ("M vs best", "makespan_vs_best", "{:+.1%}"),
        ("Norm M", "normalized_makespan", "{:.4f}"),
        ("Robot work", "robot_work", "{:.0f}"),
        ("W vs best", "robot_work_vs_best", "{:+.1%}"),
        ("Fetches", "n_fetches", "{:.1f}"),
        ("W/no-share", "robot_work_rel", "{:.3f}"),
        ("Viol", "n_violations", "{:.1f}"),
        ("Time (s)", "total_time", "{:.3f}"),
    ],
    "Utilisation, balance and picker gaps": [
        ("Robot util", "movecap_utilization", "{:.1%}"),
        ("Picker util", "picker_utilization", "{:.1%}"),
        ("C spread", "station_makespan_spread", "{:.1%}"),
        ("C CV", "station_makespan_cv", "{:.3f}"),
        ("Pick imb", "avg_pick_imbalance", "{:.1f}"),
        ("Picks/bin", "picks_from_bin_presence", "{:.3f}"),
        ("Idle", "picker_idle_frac", "{:.1%}"),
        ("Long gaps", "picker_long_gap_frac", "{:.1%}"),
        ("#Long/st", "picker_long_gaps", "{:.1f}"),
        ("Max gap", "picker_max_gap", "{:.0f}"),
    ],
}

TABLE_LEGEND = (
    "M/W vs best: mean relative gap to the best method on each instance (makespan / robot work)\n"
    "W/no-share: robot work relative to one fetch per order line; Robot util: robot work / (MoveCap * C)\n"
    "Picker util: time with >= 1 pick running / C; C spread: (C - earliest station finish) / C; "
    "C CV: coeff. of variation of station finishes\n"
    "Idle / Long gaps: picker idle time inside [first pick, last pick] (all gaps / gaps > {thr} s); "
    "#Long/st: long gaps per station; Max gap in s"
)


def add_relative_metrics(record: dict) -> None:
    """Per-instance gaps to the best method: makespan_vs_best, robot_work_vs_best."""
    res = [r for r in record["methods"].values() if "error" not in r]
    for key, rel in (("makespan", "makespan_vs_best"), ("robot_work", "robot_work_vs_best")):
        vals = [r[key] for r in res if not math.isnan(r[key]) and r.get("n_violations", 0) == 0]
        best = min(vals, default=float("nan"))
        for r in res:
            r[rel] = r[key] / best - 1 if best and not math.isnan(best) and not math.isnan(r[key]) else float("nan")


def mean_metrics(results: list[dict]) -> dict[str, float]:
    """Means per metric, ignoring NaN (metrics undefined for infeasible solutions)."""
    out = {}
    for k in results[0]:
        vals = [r[k] for r in results if isinstance(r.get(k), (int, float)) and not math.isnan(r[k])]
        out[k] = sum(vals) / len(vals) if vals else float("nan")
    return out


def print_comparison_table(results_by_method: dict[str, list[dict]], n_instances: int,
                           gap_threshold: float = DEFAULT_GAP_THRESHOLD) -> None:
    rows = {m: mean_metrics(r) for m, r in results_by_method.items() if r}
    if not rows:
        print("\nNo results.")
        return
    w = max(len("Method"), *(len(m) for m in rows))
    for title, cols in TABLES.items():
        widths = [max(len(h), 8) for h, _, _ in cols]
        header = f"{'Method':<{w}} | {'Runs':>4} | " + " | ".join(f"{h:>{cw}}" for (h, _, _), cw in zip(cols, widths))
        line = "=" * len(header)
        print(f"\n{line}\n{f'{title} (means over {n_instances} instance(s))':^{len(header)}}\n{line}")
        print(header)
        print("-" * len(header))
        for m, r in rows.items():
            cells = " | ".join(
                f"{'-' if math.isnan(r.get(k, float('nan'))) else fmt.format(r[k]):>{cw}}"
                for (_, k, fmt), cw in zip(cols, widths)
            )
            print(f"{m:<{w}} | {len(results_by_method[m]):>4} | {cells}")
        print(line)
    print(TABLE_LEGEND.format(thr=gap_threshold))
    print("Failed runs are excluded; schedule metrics cover feasible stations only.\n")


# ======================================================================
# CLI
# ======================================================================


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m heuristics benchmark",
        description="Benchmark registered heuristics (and RL checkpoints) on generated or pregenerated instances.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Presets:\n" + "\n".join(f"  {k:<10} {p.description}" for k, p in PRESETS.items())
        + "\n" + "\n".join(f"  {k:<10} legacy RL-training generator (400 orders, MoveCap 40)" for k in LEGACY_PRESETS)
        + "\n\nQuote glob patterns: --methods '*sgc*' --skip '*singles*'",
    )
    ap.add_argument("instances", nargs="*",
                    help="pregenerated instances: precalculated folders, instance.pkl files or a directory of "
                         "folders; if omitted, instances are generated from --preset")
    sel = ap.add_argument_group("method selection")
    sel.add_argument("-m", "--methods", nargs="+", metavar="PAT",
                     help="'default', 'all', registry names/aliases or globs (e.g. '*rdi*'); default: default")
    sel.add_argument("-x", "--skip", nargs="+", metavar="PAT", help="names/globs to drop from the selection")
    sel.add_argument("--list", action="store_true", help="print the selected methods and instances, then exit")

    gen = ap.add_argument_group("instance generation (no positional instances)")
    gen.add_argument("--preset", default="small", choices=[*PRESETS, *LEGACY_PRESETS])
    gen.add_argument("-n", "--num-instances", type=int, default=10)
    gen.add_argument("--seed-base", type=int, default=100000,
                     help="instance i uses seed seed_base + i (disjoint from RL training seeds by default)")
    for p in PRESET_PARAMS:
        gen.add_argument(f"--{p.replace('_', '-')}", type=parse_spec, default=None, metavar="SPEC",
                         help=f"override the preset's {p}: N, LO-HI or A,B,C")
    gen.add_argument("--save-instances", metavar="DIR",
                     help="also write the generated instances as <DIR>/<name>/{instance.pkl,config.json}")

    run = ap.add_argument_group("running")
    run.add_argument("--horizon", type=int, default=None,
                     help=f"scheduling horizon (default: config.json's for pregenerated, else {DEFAULT_HORIZON})")
    run.add_argument("--gap-threshold", type=float, default=DEFAULT_GAP_THRESHOLD, metavar="SEC",
                     help=f"picker pauses longer than this count as long gaps (default {DEFAULT_GAP_THRESHOLD})")
    run.add_argument("--output-json", default="benchmark_results.json", help="'' to disable")

    rl = ap.add_argument_group("RL checkpoints (named rl:<stem>; need torch)")
    rl.add_argument("--checkpoints", nargs="*", default=None,
                    help="checkpoint .pth files; with no paths, auto-discovers rl_agent_weights*.pth")
    rl.add_argument("--device", default=None, help="mps / cuda / cpu (auto-detected)")
    rl.add_argument("--stochastic", action="store_true", help="sample RL actions instead of argmax")
    return ap


def main(argv: list[str] | None = None) -> None:
    ap = build_parser()
    args = ap.parse_args(argv)
    # nargs="+" swallows trailing positionals (`-m sgc precalculated_instances/`): existing
    # paths are instances, never method patterns
    for opt in ("methods", "skip"):
        toks = getattr(args, opt) or []
        args.instances += [t for t in toks if os.path.exists(t)]
        setattr(args, opt, [t for t in toks if not os.path.exists(t)] or None)

    overrides = {p: getattr(args, p) for p in PRESET_PARAMS if getattr(args, p) is not None}
    if args.instances and (overrides or args.save_instances):
        ap.error("generation flags (--stations, ..., --save-instances) do not apply to pregenerated instances")
    if args.preset in LEGACY_PRESETS and overrides:
        ap.error(f"preset {args.preset} reproduces the legacy generator and takes no overrides")

    rl = {}
    if args.checkpoints is not None and not args.list:
        rl = rl_methods(args.checkpoints, args.device, deterministic=not args.stochastic)
    elif args.checkpoints is not None:
        rl = {f"rl:{Path(c).stem}": None for c in checkpoint_paths(args.checkpoints)}
    try:
        names = select_methods(args.methods, args.skip, list(rl))
    except ValueError as e:
        ap.error(str(e))
    methods: dict[str, Method] = {n: rl[n] if n in rl else heuristic_method(n) for n in names}
    if not methods:
        ap.error("no methods left after --skip")

    if args.list:
        print(f"{len(names)} method(s): {', '.join(names)}")
        if args.instances:
            for pkl in _instance_pkls(args.instances):
                print(f"  instance {pkl}")
        else:
            print(f"{args.num_instances} generated instance(s) from preset {args.preset!r}, "
                  f"seeds {args.seed_base}..{args.seed_base + args.num_instances - 1}, overrides {overrides or '-'}")
        return

    if args.instances:
        instances = loaded_instances(args.instances, args.horizon)
        source = {"instances": args.instances}
    else:
        instances = generated_instances(args, overrides)
        source = {"preset": args.preset, "seed_base": args.seed_base, "overrides": {k: str(v) for k, v in overrides.items()}}
        if args.save_instances:
            save_instances(instances, args.save_instances)
    if not instances:
        ap.error("no instances to run on")

    print(f"\n[*] {len(methods)} method(s) x {len(instances)} instance(s): {', '.join(methods)}")
    results_by_method: dict[str, list[dict]] = defaultdict(list)
    history = []
    w = max(len(n) for n in methods)

    for i, bi in enumerate(instances, 1):
        inst = bi.instance
        print(f"\n>>> Instance {i:02d}/{len(instances)} {bi.name}")
        print(f"    Stations={len(inst.S)} Lanes={len(inst.L)} Orders={len(inst.O)} SKUs={len(inst.K)} "
              f"MoveCap={inst.movecap} Horizon={bi.horizon}")
        record = {"name": bi.name, "config": bi.config, "horizon": bi.horizon, "stations": len(inst.S),
                  "orders": len(inst.O), "skus": len(inst.K), "movecap": inst.movecap, "methods": {}}
        for label, method in methods.items():
            try:
                t0 = time.perf_counter()
                sol = method(inst, bi.horizon)
                elapsed = time.perf_counter() - t0
                part = sol.meta.get("partition_time", 0.0) if sol is not None else 0.0
                res = solution_metrics(inst, sol, elapsed, part)
                res["n_violations"] = len(validate_solution(sol, inst, horizon=bi.horizon)) if sol is not None else -1
                res.update(schedule_metrics(inst, sol, args.gap_threshold))
            except Exception as e:  # keep going over the remaining methods
                print(f"    - {label:<{w}} | FAILED: {type(e).__name__}: {e}")
                record["methods"][label] = {"error": f"{type(e).__name__}: {e}"}
                continue
            results_by_method[label].append(res)
            record["methods"][label] = res
            print(f"    - {label:<{w}} | Makespan: {res['makespan']:7.0f} | Norm M: {res['normalized_makespan']:7.4f} | "
                  f"W: {res['robot_work']:7.0f} | Robot {res['movecap_utilization']:5.1%} | "
                  f"Picker {res['picker_utilization']:5.1%} | Idle {res['picker_idle_frac']:5.1%} | "
                  f"Viol: {res['n_violations']:3d} | {res['total_time']:6.2f}s")
        add_relative_metrics(record)
        history.append(record)

    print_comparison_table(results_by_method, len(instances), args.gap_threshold)

    if args.output_json:
        payload = {
            "meta": {
                "num_instances": len(instances),
                **source,
                "methods": list(methods),
                "deterministic": not args.stochastic,
                "gap_threshold": args.gap_threshold,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "summary": {m: mean_metrics(r) for m, r in results_by_method.items() if r},
            "detailed_history": history,
        }
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"[+] Saved detailed results to {args.output_json}")


if __name__ == "__main__":
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_config")
    main(sys.argv[1:])
