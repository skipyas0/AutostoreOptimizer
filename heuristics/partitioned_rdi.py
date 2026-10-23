"""Partitioned ("decomposition") RDI: assign orders to stations first, then run
RDI independently on each single-station subinstance and stitch the results.

Partition methods
    "random"         uniform random split (global ``random`` RNG; pass ``seed`` to fix it)
    "similarity"     greedy groups of weighted-Jaccard-similar orders
    "load_balanced"  LPT: orders by picking workload desc -> least-loaded station
    dict             explicit assignment {station_index: [orders]} (index into inst.S),
                     e.g. produced by the RL order-assignment policy

MoveCap is split across stations by ``distribute_movecap`` (remainder to the
stations with the most orders). Stations are solved *independently*: bin
copies (N[k]) are NOT coordinated across stations, so the stitched solution
can violate copy concurrency (see docs/KNOWN_ISSUES.md).

``meta["station_feasible"]`` records per-station feasibility, and
``meta["partition_time"]`` the time spent partitioning.
"""

from __future__ import annotations

import random
import time

from heuristics.rdi import run_rdi
from heuristics.solution import HeuristicSolution
from instance import Instance

PARTITION_METHODS = ("random", "similarity", "load_balanced")


# ======================================================================
# Partitioning (also used by decomposition.py)
# ======================================================================


def create_random_partitions(instance: Instance, n_partitions: int = 5) -> list[set[int]]:
    from neighborhood_selection import strategy_random_orders

    order_set = set(instance.O)
    partitions = []
    for i in range(n_partitions - 1):
        sel = strategy_random_orders({"O": list(order_set)}, p=1 / (n_partitions - i))
        order_set -= sel.seed_orders
        partitions.append(sel.seed_orders)
    partitions.append(order_set)
    return partitions


def create_similarity_partitions(instance: Instance, n_partitions: int = 5) -> list[set[int]]:
    from jaccard_similarity import build_similarity_matrix
    from neighborhood_selection import strategy_similar_orders

    sim, _ = build_similarity_matrix(instance.O, instance.orders_req, instance.rt, instance.rt_ret, normalize=False)
    order_set = set(instance.O)
    partitions = []
    for i in range(n_partitions - 1):
        sel = strategy_similar_orders({"O": list(order_set)}, 1 / (n_partitions - i), sim)
        order_set -= sel.seed_orders
        partitions.append(sel.seed_orders)
    partitions.append(order_set)
    return partitions


def create_load_balanced_partitions(
    instance: Instance, n_partitions: int = 5, sort_descending: bool = True
) -> list[set[int]]:
    """Greedy LPT: each order (by picking workload desc) goes to the least-loaded partition."""
    if n_partitions <= 0:
        return []
    if n_partitions == 1:
        return [set(instance.O)]
    loads = [0.0] * n_partitions
    partitions: list[set[int]] = [set() for _ in range(n_partitions)]
    work = {o: sum(instance.p.get(k, 0) for k in instance.orders_req[o]) for o in instance.O}
    orders = sorted(instance.O, key=lambda o: work[o], reverse=True) if sort_descending else list(instance.O)
    for o in orders:
        idx = min(range(n_partitions), key=lambda i: loads[i])
        partitions[idx].add(o)
        loads[idx] += work[o]
    return partitions


def distribute_movecap(total_movecap: int, n_stations: int, workloads: list | None = None) -> list[int]:
    """Split MoveCap over stations without waste; the remainder goes to the highest workloads
    (or the first stations if no workloads are given)."""
    if n_stations <= 0:
        return []
    base, rem = divmod(total_movecap, n_stations)
    if workloads is not None and len(workloads) == n_stations and rem > 0:
        bonus = set(sorted(range(n_stations), key=lambda i: workloads[i], reverse=True)[:rem])
        return [base + (1 if i in bonus else 0) for i in range(n_stations)]
    return [base + (1 if i < rem else 0) for i in range(n_stations)]


def partition_orders(
    inst: Instance, method: str | dict[int, list[int]], seed: int | None = None
) -> dict[int, list[int]]:
    """{station_index: [orders]} for a partition method name or an explicit assignment."""
    if isinstance(method, dict):
        return {i: list(method.get(i, [])) for i in range(len(inst.S))}
    # Import before seeding: importing neighborhood_selection (-> matheuristic_plots ...)
    # draws from the global RNG, which would otherwise shift the partition.
    import neighborhood_selection  # noqa: F401

    if seed is not None:
        random.seed(seed)
    fn = {
        "random": create_random_partitions,
        "similarity": create_similarity_partitions,
        "load_balanced": create_load_balanced_partitions,
    }[method]
    return {i: list(p) for i, p in enumerate(fn(inst, len(inst.S)))}


# ======================================================================
# Heuristic
# ======================================================================


def run_partitioned_rdi(
    inst: Instance,
    *,
    partition: str | dict[int, list[int]] = "load_balanced",
    horizon: int = 50000,
    seed: int | None = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    regret_k: int = 2,
    use_lazy: bool = True,
    tiebreaker: str = "sharing_degree",
) -> HeuristicSolution:
    t0 = time.perf_counter()
    assignment = partition_orders(inst, partition, seed)
    partition_time = time.perf_counter() - t0

    n = len(inst.S)
    caps = distribute_movecap(inst.movecap, n, [len(assignment.get(i, [])) for i in range(n)])

    subs, station_feasible = [], {}
    for i, s in enumerate(inst.S):
        orders = assignment.get(i, [])
        if not orders:
            station_feasible[s] = True
            continue
        sub = run_rdi(
            inst.subinstance(orders, stations=[s], movecap=caps[i]),
            horizon=horizon,
            alpha=alpha,
            beta=beta,
            regret_k=regret_k,
            use_lazy=use_lazy,
            tiebreaker=tiebreaker,
        )
        station_feasible[s] = sub.feasible
        subs.append(sub)

    sol = HeuristicSolution.stitch(subs) or HeuristicSolution({}, {}, 0, 0, True, {})
    sol.meta.update(
        {
            "heuristic": "partitioned_rdi",
            "partition": partition if isinstance(partition, str) else "explicit",
            "station_feasible": station_feasible,
            "assignment": {s: list(assignment.get(i, [])) for i, s in enumerate(inst.S)},
            "partition_time": partition_time,
        }
    )
    return sol
