"""Cluster-First Schedule-Second SGC (CFSS-SGC).

1. Weighted-Jaccard similarity between all order pairs
   (``normalize_similarity`` selects the normalized variant).
2. Agglomerative clustering, UPGMA average linkage, merging while similarity
   >= ``sim_threshold`` and cluster size <= ``max_cluster_orders``.
3. Soft station assignment: clusters (multi-order first, then singletons, each
   by total load desc) go to the station with the least load so far. The load is
   the rt mass (``balance="rt_mass"``, default) or the picking workload Σ p
   (``balance="work"``).
4. Interleaved scheduling: repeatedly take the station whose earliest lane is
   free first and commit its pending order with the best SGC plan score. With
   ``consecutive=True`` only the orders of the station's current cluster (the first
   one, in assignment order, with pending orders) are candidates, so cluster mates
   are on the lanes at the same time and can share bin visits; if none of them fits,
   all pending orders of the station are tried.

Plans use the SGC planner and respect MoveCap and PickCap. ``extend_share_window`` and
``share_reward`` are the SGC bin-sharing options (see ``heuristic_utils.ShareOptions``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from heuristics.heuristic_utils import ShareOptions, commit_plan, init_state, record_plan
from heuristics.sgc import plan_order_at_station
from heuristics.solution import HeuristicSolution
from instance import Instance


CFSS_BALANCE = ("rt_mass", "work")


@dataclass
class OrderCluster:
    """A group of orders to be co-located at one station."""

    order_ids: list[int]
    all_skus: set[int] = field(default_factory=set)
    total_rt_mass: float = 0.0
    total_work: float = 0.0  # picking workload Σ p


def agglomerative_cluster(
    inst: Instance,
    sim_matrix: dict[tuple[int, int], float],
    threshold: float = 0.1,
    max_cluster_orders: int = 8,
) -> list[OrderCluster]:
    """Bottom-up clustering with average linkage and a cluster-size cap."""
    O, orders_req, rt = inst.O, inst.orders_req, inst.rt
    clusters: dict[int, list[int]] = {o: [o] for o in O}
    cluster_size: dict[int, int] = {o: 1 for o in O}

    def _key(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a < b else (b, a)

    cluster_sim: dict[tuple[int, int], float] = {}
    for (o1, o2), s in sim_matrix.items():
        cluster_sim[_key(o1, o2)] = s

    active = set(O)  # membership only; iterate via sorted(active) so results never depend on set order
    while True:
        best_sim = -1.0
        best_pair: tuple[int, int] | None = None
        for (ca, cb), s in cluster_sim.items():
            if ca not in active or cb not in active or s <= best_sim:
                continue
            if cluster_size[ca] + cluster_size[cb] > max_cluster_orders:
                continue
            best_sim, best_pair = s, (ca, cb)
        if best_pair is None or best_sim < threshold:
            break

        ca, cb = best_pair  # absorb cb into ca
        clusters[ca] = clusters[ca] + clusters[cb]
        new_size = cluster_size[ca] + cluster_size[cb]
        cluster_size[ca] = new_size
        active.discard(cb)
        del clusters[cb]

        # UPGMA update. NB: cluster_size[ca] already holds the merged size here, so the
        # weight of the old ca part is (new_size - |cb|) — matches the original code.
        for co in sorted(active):
            if co == ca:
                continue
            sim_ca_co = cluster_sim.get(_key(ca, co), 0.0)
            sim_cb_co = cluster_sim.get(_key(cb, co), 0.0)
            cluster_sim[_key(ca, co)] = (
                (cluster_size[ca] - cluster_size[cb]) * sim_ca_co + cluster_size[cb] * sim_cb_co
            ) / new_size
        for k in [k for k in cluster_sim if cb in k]:
            del cluster_sim[k]

    result: list[OrderCluster] = []
    for cid in sorted(active):
        order_ids = clusters[cid]
        all_skus: set[int] = set()
        mass = work = 0.0
        for o in order_ids:
            all_skus.update(orders_req[o])
            mass += sum(rt[k] for k in orders_req[o])
            work += sum(inst.p[k] for k in orders_req[o])
        result.append(OrderCluster(order_ids=order_ids, all_skus=all_skus, total_rt_mass=mass, total_work=work))
    return result


def run_cfss(
    inst: Instance,
    *,
    horizon: int = 10000,
    alpha: float = 1.0,
    beta: float = 1.0,
    sim_threshold: float = 0.1,
    max_cluster_orders: int = 4,
    normalize_similarity: bool = False,
    balance: str = "rt_mass",
    consecutive: bool = False,
    extend_share_window: bool = False,
    share_reward: float = 0.0,
) -> HeuristicSolution:
    from jaccard_similarity import build_similarity_matrix

    if balance not in CFSS_BALANCE:
        raise ValueError(f"Unknown CFSS balance {balance!r}; choose from {CFSS_BALANCE}")
    S, L = inst.S, inst.L
    sim_matrix, _ = build_similarity_matrix(inst.O, inst.orders_req, inst.rt, inst.rt_ret, normalize=normalize_similarity)
    clusters = agglomerative_cluster(inst, sim_matrix, threshold=sim_threshold, max_cluster_orders=max_cluster_orders)

    size = (lambda c: c.total_rt_mass) if balance == "rt_mass" else (lambda c: c.total_work)
    multi = sorted((c for c in clusters if len(c.order_ids) > 1), key=size, reverse=True)
    single = sorted((c for c in clusters if len(c.order_ids) == 1), key=size, reverse=True)

    # Phase 1: soft station assignment balancing rt mass (or picking workload)
    station_orders: dict[int, list[int]] = {s: [] for s in S}
    station_clusters: dict[int, list[list[int]]] = {s: [] for s in S}
    station_load = {s: 0.0 for s in S}
    for cluster in multi + single:
        best_s = min(S, key=lambda s: station_load[s])
        station_orders[best_s].extend(cluster.order_ids)
        station_clusters[best_s].append(cluster.order_ids)
        station_load[best_s] += size(cluster)

    # Phase 2: interleaved scheduling
    state = init_state(inst, horizon, share=ShareOptions(extend_window=extend_share_window, reward=share_reward))
    order_assignments: dict[int, tuple[int, int, int, int]] = {}
    pick_events: dict[tuple[int, int, int], tuple[int, int]] = {}
    failed_orders: list[int] = []
    # ordered sets (dict keys): pending orders per station, in assignment order
    unassigned = {s: dict.fromkeys(station_orders[s]) for s in S}

    while any(unassigned.values()):
        earliest_s, earliest_time = None, float("inf")
        for s in S:
            if not unassigned[s]:
                continue
            s_free = min(state.lane_free[(s, ln)] for ln in L) if L else 0
            if s_free < earliest_time:
                earliest_time, earliest_s = s_free, s
        if earliest_s is None:
            break

        def best_of(cands):
            best_plan, best_score = None, float("inf")
            for o in cands:  # assignment order; the first of exactly tied plans wins
                plan = plan_order_at_station(o, earliest_s, state, alpha, beta)
                if plan is not None and plan.score < best_score:
                    best_score, best_plan = plan.score, plan
            return best_plan

        best_plan = None
        if consecutive:
            current = next(c for c in station_clusters[earliest_s] if any(o in unassigned[earliest_s] for o in c))
            best_plan = best_of([o for o in current if o in unassigned[earliest_s]])
        if best_plan is None:
            best_plan = best_of(unassigned[earliest_s])

        if best_plan is None:  # nothing left fits at this station (horizon)
            failed_orders.extend(unassigned[earliest_s])
            unassigned[earliest_s].clear()
            continue

        commit_plan(best_plan, state)
        record_plan(best_plan, order_assignments, pick_events)
        del unassigned[earliest_s][best_plan.order]

    if failed_orders:
        print(f"[CFSS-SGC] WARNING: {len(failed_orders)} orders could not be scheduled: {failed_orders}")

    return HeuristicSolution(
        order_assignments=order_assignments,
        bin_events=dict(state.station_bin_events),
        makespan=max((end for _, _, _, end in order_assignments.values()), default=0),
        total_moves=sum(len(evts) * 2 for evts in state.station_bin_events.values()),
        feasible=not failed_orders,
        pick_events=pick_events,
        meta={"heuristic": "cfss", "normalize_similarity": normalize_similarity, "n_clusters": len(clusters),
              "balance": balance, "consecutive": consecutive,
              **({"extend_share_window": True} if extend_share_window else {}),
              **({"share_reward": share_reward} if share_reward else {})},
    )
