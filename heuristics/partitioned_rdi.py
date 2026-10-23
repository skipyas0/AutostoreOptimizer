"""Partitioned ("decomposition") RDI: assign orders to stations first, then run
RDI independently on each single-station subinstance and stitch the results.

Partition methods
    "random"         uniform random split (local RNG seeded by ``seed``, default 0)
    "similarity"     greedy groups of weighted-Jaccard-similar orders (seeded likewise)
    "load_balanced"  LPT: orders by picking workload desc -> least-loaded station
    dict             explicit assignment {station_index: [orders]} (index into inst.S),
                     e.g. produced by the RL order-assignment policy

MoveCap is split across stations by ``distribute_movecap`` (remainder to the
stations with the most orders). If MoveCap < |S|, named methods only use the
first ``movecap`` stations (``active_station_count``) so every used station has a
robot; with an explicit assignment, a station that ends up with MoveCap 0 is not
solved — its orders stay unscheduled (warning, station marked infeasible).

Stations are solved *independently*: bin copies (N[k]) are NOT coordinated
across stations. If the stitched solution exceeds a copy limit, a warning is
issued and the solution is marked infeasible (so it is never used as a warm start).

meta: ``station_feasible`` (per station), ``assignment`` (station -> orders),
``partition_time``, ``copy_conflicts`` (number of SKUs over their copy limit).
"""

from __future__ import annotations

import random
import time
import warnings

from heuristics.rdi import run_rdi
from heuristics.solution import HeuristicSolution, validate_solution
from instance import Instance

PARTITION_METHODS = ("random", "similarity", "load_balanced")


# ======================================================================
# Partitioning (also used by decomposition.py)
# ======================================================================


def active_station_count(inst: Instance) -> int:
    """Stations that can get at least one robot when MoveCap is split: min(|S|, MoveCap)."""
    return min(len(inst.S), inst.movecap)


def create_random_partitions(instance: Instance, n_partitions: int = 5, seed: int = 0) -> list[set[int]]:
    """Successively draw 1/(n-i) of the remaining orders uniformly at random."""
    rng = random.Random(seed)
    order_set = set(instance.O)
    partitions = []
    for i in range(n_partitions - 1):
        orders = sorted(order_set)
        n_select = min(len(orders), max(1, int(len(orders) * min(1.0, 1 / (n_partitions - i)))))
        selected = set(rng.sample(orders, n_select))
        order_set -= selected
        partitions.append(selected)
    partitions.append(order_set)
    return partitions


def create_similarity_partitions(instance: Instance, n_partitions: int = 5, seed: int = 0) -> list[set[int]]:
    """Successively pick a random seed order plus the 1/(n-i) share of remaining orders most
    similar to it (weighted Jaccard)."""
    from jaccard_similarity import build_similarity_matrix

    sim, _ = build_similarity_matrix(instance.O, instance.orders_req, instance.rt, instance.rt_ret, normalize=False)
    rng = random.Random(seed)
    order_set = set(instance.O)
    partitions = []
    for i in range(n_partitions - 1):
        orders = sorted(order_set)
        seed_order = rng.choice(orders)
        by_sim = sorted(orders, key=lambda o: float("inf") if o == seed_order else sim[min(seed_order, o), max(seed_order, o)])
        n_select = max(1, int(len(orders) * min(1.0, 1 / (n_partitions - i))))
        selected = {seed_order, *by_sim[-n_select:]}
        order_set -= selected
        partitions.append(selected)
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
    (or the first stations if no workloads are given). Stations get 0 if MoveCap < n_stations."""
    if n_stations <= 0:
        return []
    base, rem = divmod(total_movecap, n_stations)
    if workloads is not None and len(workloads) == n_stations and rem > 0:
        bonus = set(sorted(range(n_stations), key=lambda i: workloads[i], reverse=True)[:rem])
        return [base + (1 if i in bonus else 0) for i in range(n_stations)]
    return [base + (1 if i < rem else 0) for i in range(n_stations)]


def partition_orders(inst: Instance, method: str | dict[int, list[int]], seed: int = 0) -> dict[int, list[int]]:
    """{station_index: sorted orders} for a partition method or an explicit assignment.

    Named methods only use the first ``active_station_count(inst)`` stations.
    """
    if isinstance(method, dict):
        return {i: list(method.get(i, [])) for i in range(len(inst.S))}
    n_active = active_station_count(inst)
    if method == "random":
        parts = create_random_partitions(inst, n_active, seed)
    elif method == "similarity":
        parts = create_similarity_partitions(inst, n_active, seed)
    elif method == "load_balanced":
        parts = create_load_balanced_partitions(inst, n_active)
    else:
        raise ValueError(f"Unknown partition method {method!r}; use one of {PARTITION_METHODS} or a dict")
    return {i: sorted(p) for i, p in enumerate(parts)}


# ======================================================================
# Heuristic
# ======================================================================


def run_partitioned_rdi(
    inst: Instance,
    *,
    partition: str | dict[int, list[int]] = "load_balanced",
    horizon: int = 50000,
    seed: int = 0,
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
        if caps[i] == 0:  # no robot left for this station: do not solve it
            warnings.warn(
                f"partitioned RDI: station {s} gets MoveCap 0 (MoveCap {inst.movecap} < {n} stations) "
                f"but was assigned {len(orders)} orders; they stay unscheduled"
            )
            station_feasible[s] = False
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
    sol.feasible = sol.feasible and all(station_feasible.values())

    # Stations were solved independently: check the cross-station bin-copy limits N[k].
    conflicts = [v for v in validate_solution(sol, inst, horizon=horizon) if "Block concurrency" in v]
    if conflicts:
        warnings.warn(
            f"partitioned RDI: stitched solution exceeds the bin-copy limit N[k] for {len(conflicts)} SKU(s) "
            f"(stations are solved independently); marking it infeasible. First: {conflicts[0]}"
        )
        sol.feasible = False

    sol.meta.update(
        {
            "heuristic": "partitioned_rdi",
            "partition": partition if isinstance(partition, str) else "explicit",
            "station_feasible": station_feasible,
            "assignment": {s: list(assignment.get(i, [])) for i, s in enumerate(inst.S)},
            "partition_time": partition_time,
            "copy_conflicts": len(conflicts),
        }
    )
    return sol
