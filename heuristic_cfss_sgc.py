#!/usr/bin/env python3
"""
AutoStore Cluster-First Schedule-Second SGC (CFSS-SGC) Heuristic.

Groups orders by weighted Jaccard similarity (agglomerative clustering) then
schedules each cluster at the best station, maximising bin sharing within the
cluster. Mirrors variable names from the CP model.
"""
from autostore_heuristic import (
    HeuristicState, Solution,
    init_state, commit_plan, validate_solution,
    plan_order_at_station,
    build_viz_handles,
)
import copy
import sys
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# Step 1: Weighted Jaccard similarity
# ============================================================

def compute_demand_count(
        orders_req: dict[int, list[int]],
        K: list[int],
) -> dict[int, int]:
    """Count how many orders need each SKU."""
    count: dict[int, int] = defaultdict(int)
    for o in orders_req:
        for k in orders_req[o]:
            count[k] += 1
    return dict(count)


def compute_weighted_jaccard(
        o1: int, o2: int,
        orders_req: dict[int, list[int]],
        rt: dict[int, int],
        rt_ret: dict[int, int],
) -> float:
    """Retrieval-time-weighted Jaccard similarity between two orders.

    J_w(o, o') = Σ_{k ∈ R_o ∩ R_o'} (rt[k]+rt_ret[k])
                 / Σ_{k ∈ R_o ∪ R_o'} (rt[k]+rt_ret[k])

    Returns 0.0 when both orders are empty or have no union.
    """
    set1 = set(orders_req[o1])
    set2 = set(orders_req[o2])
    intersection = set1 & set2
    union = set1 | set2

    union_weight = sum(rt[k] + rt_ret[k] for k in union)
    if union_weight == 0:
        return 0.0
    intersection_weight = sum(rt[k] + rt_ret[k] for k in intersection)
    return intersection_weight / union_weight


def build_similarity_matrix(
        O: list[int],
        orders_req: dict[int, list[int]],
        rt: dict[int, int],
        rt_ret: dict[int, int],
) -> dict[tuple[int, int], float]:
    """Compute pairwise weighted Jaccard for all order pairs (upper triangle).

    Returns {(o1, o2): similarity} where o1 < o2 in the O list index sense.
    """
    sim: dict[tuple[int, int], float] = {}
    for i in range(len(O)):
        for j in range(i + 1, len(O)):
            o1, o2 = O[i], O[j]
            key = (o1, o2) if o1 < o2 else (o2, o1)
            sim[key] = compute_weighted_jaccard(o1, o2, orders_req, rt, rt_ret)
    return sim


# ============================================================
# Step 2: Agglomerative clustering with capacity constraints
# ============================================================

@dataclass
class OrderCluster:
    """A group of orders to be co-located at one station."""
    order_ids: list[int]
    all_skus: set[int] = field(default_factory=set)
    total_rt_mass: float = 0.0


def agglomerative_cluster(
        O: list[int],
        orders_req: dict[int, list[int]],
        sim_matrix: dict[tuple[int, int], float],
        rt: dict[int, int],
        threshold: float = 0.1,
        max_cluster_orders: int = 8,
) -> list[OrderCluster]:
    """Bottom-up agglomerative clustering with average linkage and capacity constraint.

    Merges the pair of clusters with the highest average-linkage weighted Jaccard
    similarity until all remaining similarities fall below `threshold` or no valid
    (capacity-respecting) merge exists.

    Args:
        threshold: Minimum inter-cluster similarity to allow a merge.
        max_cluster_orders: Maximum number of orders per cluster.

    Returns:
        List of OrderCluster objects.
    """
    # --- Initialise: each order is its own singleton cluster ---
    # cluster_id -> list of order_ids
    clusters: dict[int, list[int]] = {o: [o] for o in O}
    # Cluster sizes for UPGMA average-linkage
    cluster_size: dict[int, int] = {o: 1 for o in O}
    # Inter-cluster similarity: (cid_a, cid_b) -> similarity (cid_a < cid_b)
    # Seeded from pairwise order similarities
    cluster_sim: dict[tuple[int, int], float] = {}
    for (o1, o2), s in sim_matrix.items():
        key = (o1, o2) if o1 < o2 else (o2, o1)
        cluster_sim[key] = s

    active = set(O)

    def _key(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a < b else (b, a)

    while True:
        # Find the best merge candidate
        best_sim = -1.0
        best_pair: Optional[tuple[int, int]] = None
        for (ca, cb), s in cluster_sim.items():
            if ca not in active or cb not in active:
                continue
            if s <= best_sim:
                continue
            # Capacity check
            if cluster_size[ca] + cluster_size[cb] > max_cluster_orders:
                continue
            best_sim = s
            best_pair = (ca, cb)

        if best_pair is None or best_sim < threshold:
            break

        ca, cb = best_pair
        # New cluster id = ca (absorb cb into ca)
        new_orders = clusters[ca] + clusters[cb]
        clusters[ca] = new_orders
        new_size = cluster_size[ca] + cluster_size[cb]
        cluster_size[ca] = new_size
        active.discard(cb)
        del clusters[cb]

        # UPGMA average-linkage update: recompute sim from new ca to all others
        for co in list(active):
            if co == ca:
                continue
            k1 = _key(ca, co)
            k2 = _key(cb, co)
            sim_ca_co = cluster_sim.get(k1, 0.0)  # before merge
            sim_cb_co = cluster_sim.get(k2, 0.0)
            new_sim = (
                (cluster_size[ca] - cluster_size[cb]) * sim_ca_co
                + cluster_size[cb] * sim_cb_co
            ) / new_size
            # Store under canonical key with new ca
            new_key = _key(ca, co)
            cluster_sim[new_key] = new_sim

        # Remove all entries referencing cb
        to_del = [k for k in cluster_sim if cb in k]
        for k in to_del:
            del cluster_sim[k]

    # Build result
    result: list[OrderCluster] = []
    for cid in active:
        order_ids = clusters[cid]
        all_skus: set[int] = set()
        total_mass = 0.0
        for o in order_ids:
            all_skus.update(orders_req[o])
            total_mass += sum(rt[k] for k in orders_req[o])
        result.append(OrderCluster(
            order_ids=order_ids,
            all_skus=all_skus,
            total_rt_mass=total_mass,
        ))
    return result


# ============================================================
# Step 3: State snapshot and tentative cluster scoring
# ============================================================

def snapshot_state(state: HeuristicState) -> HeuristicState:
    """Return a deep copy of HeuristicState for tentative scheduling."""
    return copy.deepcopy(state)


def score_cluster_at_station(
        cluster: "OrderCluster",
        s: int,
        state: HeuristicState,
        orders_req: dict[int, list[int]],
        rt: dict[int, int], rt_ret: dict[int, int], p: dict[int, int],
        N: dict[int, int],
        horizon: int,
        ALPHA: float, BETA: float,
        demand_count: dict[int, int],
        sharing_degree: dict[int, int],
) -> tuple[float, list]:
    """Tentatively score assigning cluster to station s without mutating state.

    Returns (total_score, plans). Returns (inf, []) if any order is infeasible.
    """
    snap = snapshot_state(state)
    plans = []
    total_score = 0.0
    for o in cluster.order_ids:
        plan = plan_order_at_station(
            o, s, snap, orders_req, rt, rt_ret, p, N, horizon, ALPHA, BETA, demand_count
        )
        if plan is None:
            return float("inf"), []
        plans.append(plan)
        commit_plan(plan, snap)
        total_score += plan.score
    return total_score, plans


# ============================================================
# Step 4: Main scheduling loop
# ============================================================

def run_cfss_sgc(
        S: list[int], L: list[int], K: list[int], O: list[int],
        orders_req: dict[int, list[int]],
        rt: dict[int, int], rt_ret: dict[int, int], p: dict[int, int],
        N: dict[int, int],
        horizon: int,
        move_cap: Optional[int] = None,
        ALPHA: float = 1.0,
        BETA: float = 0.0,
        sim_threshold: float = 0.1,
        max_cluster_orders: int = 8,
) -> Solution:
    """Cluster-First Schedule-Second SGC heuristic.

    1. Build weighted-Jaccard similarity matrix for all order pairs.
    2. Agglomeratively cluster orders (average linkage, capacity constraint).
    3. Sort clusters: multi-order clusters by total_rt_mass desc, singletons last.
    4. Greedily assign clusters to stations balancing total RT mass.
    5. Globally interleave, picking earliest available station and its best order.
    6. Return assembled Solution.
    """
    sim_matrix = build_similarity_matrix(O, orders_req, rt, rt_ret)
    clusters = agglomerative_cluster(
        O, orders_req, sim_matrix, rt,
        threshold=sim_threshold,
        max_cluster_orders=max_cluster_orders,
    )

    # Step 5: separate multi-order clusters from singletons
    multi_clusters = [c for c in clusters if len(c.order_ids) > 1]
    singleton_clusters = [c for c in clusters if len(c.order_ids) == 1]

    # Precompute Universal Move Minimization attributes
    demand_count: dict[int, int] = defaultdict(int)
    for req in orders_req.values():
        for k in req:
            demand_count[k] += 1

    req_sets = {o: set(req) for o, req in orders_req.items()}
    sharing_degree: dict[int, int] = {}
    for o in O:
        req = req_sets[o]
        sharing_degree[o] = sum(1 for o2 in O if o2 != o and not req.isdisjoint(req_sets[o2]))

    # Sort multi-clusters by total_rt_mass descending (heaviest first)
    multi_clusters.sort(key=lambda c: c.total_rt_mass, reverse=True)
    # Sort singletons by total_rt_mass descending too (consistent with base SGC spirit)
    singleton_clusters.sort(key=lambda c: c.total_rt_mass, reverse=True)

    scheduled_order = multi_clusters + singleton_clusters

    # --- NEW PHASE 1: Soft Station Assignment ---
    # station_assigned_orders[s] = list of orders assigned to station s
    station_assigned_orders = {s: [] for s in S}
    station_rt_load = {s: 0.0 for s in S}

    for cluster in scheduled_order:
        # Find the station with the least amount of RT mass assigned so far
        best_s = min(S, key=lambda s: station_rt_load[s])

        # Assign all orders in this cluster to this station's pool
        station_assigned_orders[best_s].extend(cluster.order_ids)
        station_rt_load[best_s] += cluster.total_rt_mass

    # --- NEW PHASE 2: Interleaved Scheduling Loop ---
    state = init_state(S, L, K, N, horizon, move_cap)
    order_assignments: dict[int, tuple[int, int, int, int]] = {}
    pick_events_map: dict[tuple[int, int, int], tuple[int, int]] = {}
    failed_orders: list[int] = []

    # Keep track of unassigned orders per station
    unassigned_orders = {s: set(station_assigned_orders[s]) for s in S}

    while any(unassigned_orders.values()):
        # Find the station that is available the earliest AND has pending orders
        # Availability is estimated by the earliest free time across its lanes
        earliest_s = None
        earliest_time = float('inf')

        for s in S:
            if not unassigned_orders[s]:
                continue

            # Find when this station will have a free lane
            s_free_time = min(state.lane_free[(s, ln)] for ln in L) if L else 0
            if s_free_time < earliest_time:
                earliest_time = s_free_time
                earliest_s = s

        if earliest_s is None:
            break  # Should not happen unless all remaining orders are infeasible

        # Evaluate all pending orders for earliest_s
        best_plan = None
        best_score = float('inf')

        for o in unassigned_orders[earliest_s]:
            plan = plan_order_at_station(
                o, earliest_s, state, orders_req, rt, rt_ret, p, N, horizon, ALPHA, BETA, demand_count
            )
            if plan is not None and plan.score < best_score:
                best_score = plan.score
                best_plan = plan

        if best_plan is None:
            # None of the remaining orders for this station can be scheduled (horizon limit)
            failed_orders.extend(unassigned_orders[earliest_s])
            unassigned_orders[earliest_s].clear()
            continue

        # Commit the best plan
        commit_plan(best_plan, state)
        order_assignments[best_plan.order] = (
            best_plan.station, best_plan.lane, best_plan.start, best_plan.end,
        )
        for k, times in best_plan.pick_times.items():
            pick_events_map[(best_plan.order, best_plan.station, k)] = times

        unassigned_orders[earliest_s].remove(best_plan.order)

    makespan = max((end for _, _, _, end in order_assignments.values()), default=0)
    total_moves = sum(len(evts) * 2 for evts in state.station_bin_events.values())
    feasible = len(failed_orders) == 0

    if failed_orders:
        print(f"[CFSS-SGC] WARNING: {len(failed_orders)} orders could not be scheduled: {failed_orders}")

    return Solution(
        order_assignments=order_assignments,
        bin_events=dict(state.station_bin_events),
        makespan=makespan,
        total_moves=total_moves,
        feasible=feasible,
        pick_events=pick_events_map,
    )


# ============================================================
# Step 6: solve_heuristic_instance wrapper
# ============================================================

def solve_heuristic_instance(config: dict, return_raw: bool = False):
    """Run one CFSS-SGC instance described by *config*.

    Accepts the same config keys as the base SGC's ``solve_heuristic_instance``
    plus optional 'sim_threshold' (float) and 'max_cluster_orders' (int).

    Returns::

        {
            'status':          'Feasible' | 'Infeasible',
            'solve_time':      float,
            'objective_value': float | None,
            'num_vars':        0,
            'progress':        [],
            'total_moves':     int,
        }
    """
    import time
    from datagen import generate_data

    num_stations = config.get("stations", 1)
    lanes_per_station = config.get("lanes", 2)
    num_orders = config.get("orders", 7)
    num_skus = config.get("skus", 5)
    seed = config.get("seed", 42)
    pick_touch_time = config.get("pick", 4)
    horizon = config.get("horizon", 10000)
    move_cap = config.get("movecap", None)
    alpha = config.get("alpha", 1.0)
    beta = config.get("beta", 1.0)
    sim_threshold = config.get("sim_threshold", 0.1)
    max_cluster_orders = config.get("max_cluster_orders", 4)

    S, L, K, orders_req, rt, p, N = generate_data(
        num_stations=num_stations,
        lanes_per_station=lanes_per_station,
        num_orders=num_orders,
        num_skus=num_skus,
        seed=seed,
        pick_touch_time=pick_touch_time,
    )
    rt_ret = dict(rt)
    O = sorted(orders_req.keys())

    t0 = time.perf_counter()
    sol = run_cfss_sgc(
        S, L, K, O, orders_req, rt, rt_ret, p, N,
        horizon=horizon, move_cap=move_cap, ALPHA=alpha, BETA=beta,
        sim_threshold=sim_threshold, max_cluster_orders=max_cluster_orders,
    )
    elapsed = time.perf_counter() - t0
    status = "Feasible" if sol.feasible else "Infeasible"
    violations = validate_solution(
        sol, S, L, K, O, orders_req, rt, rt_ret, p, N,
        horizon=horizon, move_cap=move_cap,
    )
    if violations:
        print(f"VALIDATION FAILED ({len(violations)} violations)")
        status = "Invalid"
        for v in violations[:10]:
            print(f"  Violation: {v}")
    else:
        print("Validation PASSED")

    res = {
        "status": status,
        "solve_time": elapsed,
        "objective_value": float(sol.makespan) if sol.feasible else None,
        "num_vars": 0,
        "progress": [],
        "total_moves": sol.total_moves,
    }
    return (res, sol) if return_raw else res


# ============================================================
# CLI entry point
# ============================================================

def main() -> None:
    """CLI entry point for running the CFSS-SGC heuristic standalone."""
    import argparse
    import time
    from datagen import generate_data

    ap = argparse.ArgumentParser(
        description="CFSS-SGC Heuristic for AutoStore Task B (v5)"
    )
    ap.add_argument("--stations", type=int, default=4)
    ap.add_argument("--lanes", type=int, default=4)
    ap.add_argument("--orders", type=int, default=160)
    ap.add_argument("--skus", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pick", type=int, default=4)
    ap.add_argument("--movecap", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--sim_threshold", type=float, default=0.1,
                    help="Minimum inter-cluster weighted Jaccard to allow a merge (default 0.1)")
    ap.add_argument("--max_cluster_orders", type=int, default=4,
                    help="Maximum orders per cluster (default 4)")
    ap.add_argument("--no_vis", action="store_true", help="Skip HTML schedule visualisation")
    args = ap.parse_args()

    print("Generating data...")
    S, L, K, orders_req, rt, p, N = generate_data(
        num_stations=args.stations,
        lanes_per_station=args.lanes,
        num_orders=args.orders,
        num_skus=args.skus,
        seed=args.seed,
        pick_touch_time=args.pick,
    )
    rt_ret = dict(rt)
    O = sorted(orders_req.keys())

    print(f"Stations={len(S)}, Lanes={len(L)}, SKUs={len(K)}, Orders={len(O)}, RobotLimit={args.movecap}\n")

    print(
        f"\nRunning CFSS-SGC (alpha={args.alpha}, beta={args.beta}, "
        f"sim_threshold={args.sim_threshold}, max_cluster_orders={args.max_cluster_orders})..."
    )
    t0 = time.perf_counter()
    sol = run_cfss_sgc(
        S, L, K, O, orders_req, rt, rt_ret, p, N,
        horizon=args.horizon, move_cap=args.movecap,
        ALPHA=args.alpha, BETA=args.beta,
        sim_threshold=args.sim_threshold,
        max_cluster_orders=args.max_cluster_orders,
    )
    elapsed = time.perf_counter() - t0

    print(f"\n=== CFSS-SGC Result ===")
    print(f"Feasible:    {sol.feasible}")
    print(f"Makespan:    {sol.makespan}")
    print(f"Total bin events (moves/2): {sol.total_moves // 2}")
    print(f"Time:        {elapsed:.4f}s")

    violations = validate_solution(
        sol, S, L, K, O, orders_req, rt, rt_ret, p, N,
        horizon=args.horizon, move_cap=args.movecap,
    )
    if violations:
        print(f"VALIDATION FAILED ({len(violations)} violations)")
        for v in violations[:10]:
            print(f"  Violation: {v}")
    else:
        print("Validation PASSED")

    if not args.no_vis:
        try:
            from schedule_visualizer import plot_schedule
            mock_sol, handles = build_viz_handles(
                sol, S, L, K, O, orders_req, rt, rt_ret, p,
            )
            plot_schedule(mock_sol, handles)
        except Exception as exc:
            print(f"[VIS] Skipped: {exc}")


if __name__ == "__main__":
    main()
