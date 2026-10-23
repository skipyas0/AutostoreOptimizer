"""Sequential Greedy Construction (SGC) and its multi-rule wrapper AMA.

SGC
    Orders are taken one by one in a static *order dispatching rule* order.
    Each order is tentatively planned at every candidate station
    (``plan_order_at_station``) and committed at the station with the lowest
    score ``alpha * end + beta * (end - start)``. Inside an order, SKUs are
    visited in *SKU dispatching rule* order; each SKU is first served from an
    existing bin at the station (``find_shared_bin``) and otherwise a new bin
    copy is fetched as early as copy availability, the pickface and MoveCap allow.

    ``load_balanced=True`` first restricts candidate stations to those whose
    picking workload is <= the current average (fallback: all others).

    ``piggyback_singles=True``: the SKU of an order requested by the most
    unscheduled 1-item orders is picked last; after committing the order, those
    1-item orders are chained on the lane it vacated, all picking from that last
    bin visit (see ``run_sgc``).

    Default rules (sharing_degree asc, demand desc) reproduce the original SGC.

AMA (Adaptive Multi-Attribute)
    Runs SGC for many (order rule, SKU rule) combinations and keeps the best
    by ``HeuristicSolution.objective``. ``mode="full_grid"`` = all 192
    combinations, ``"two_phase"`` = 16 order rules (SKU rule fixed to
    demand desc) then 12 SKU rules for the best order rule. ``load_balanced``
    and ``piggyback_singles`` are passed through to every SGC run.
"""

from __future__ import annotations

from typing import NamedTuple

from heuristics.heuristic_utils import (
    BinEvent,
    ExoState,
    HeuristicState,
    OrderPlan,
    PlanScratch,
    commit_plan,
    earliest_feasible_fetch,
    earliest_lane,
    find_shared_bin,
    init_state,
    record_plan,
)
from heuristics.solution import HeuristicSolution
from instance import Instance


class Rule(NamedTuple):
    """Dispatching rule: sort by ``Instance.order_attrs[o][attr]`` / ``sku_attrs[k][attr]``."""

    attr: str
    descending: bool

    def __str__(self) -> str:
        return f"{self.attr}({'desc' if self.descending else 'asc'})"


ORDER_ATTRS = [
    "sum_rt",
    "order_size",
    "sum_cycle",
    "max_rt",
    "sku_rarity",
    "sku_contention",
    "sharing_degree",
    "min_copies",
]
BIN_ATTRS = ["rt", "p", "cycle", "demand_ratio", "copies", "demand"]
DIRECTIONS = [True, False]  # True = descending

# The 192 (order_rule, sku_rule) combinations, in the order used by the AMA dataset.
ALL_RULES: list[tuple[Rule, Rule]] = [
    (Rule(oa, od), Rule(ba, bd)) for oa in ORDER_ATTRS for od in DIRECTIONS for ba in BIN_ATTRS for bd in DIRECTIONS
]

DEFAULT_ORDER_RULE = Rule("sharing_degree", False)
DEFAULT_SKU_RULE = Rule("demand", True)


# ======================================================================
# Planner (also used by CFSS)
# ======================================================================


def plan_order_at_station(
    o: int,
    s: int,
    state: HeuristicState,
    alpha: float,
    beta: float,
    sku_rule: Rule = DEFAULT_SKU_RULE,
    last_sku: int | None = None,
) -> OrderPlan | None:
    """Tentatively schedule order o at station s. Returns an OrderPlan or None if infeasible.

    ``last_sku`` (piggybacking) moves that SKU to the end of the sku_rule sequence.
    """
    inst = state.inst
    rt, rt_ret, p = inst.rt, inst.rt_ret, inst.p
    lane = earliest_lane(state, s)
    if lane is None:
        return None
    best_lane, t_order_start = lane

    sku_attrs = inst.sku_attrs
    skus_sorted = sorted(inst.orders_req[o], key=lambda k: sku_attrs[k][sku_rule.attr], reverse=sku_rule.descending)
    if last_sku is not None:
        skus_sorted = [k for k in skus_sorted if k != last_sku] + [k for k in skus_sorted if k == last_sku]

    # t_cursor = when the picker (order) is free; current_station_free = when the pickface is free
    t_cursor = t_order_start
    current_station_free = state.pickface_free[s]
    sc = PlanScratch.for_station(state, s)

    for k in skus_sorted:
        shared, pick_start, new_ret_start, new_ret_end = find_shared_bin(
            state.station_bin_events[s] + sc.new_bin_events, k, t_cursor, state, s, sc
        )
        if shared is not None:
            t_cursor, extended = sc.apply_shared_pick(state, s, k, shared, pick_start, new_ret_start, new_ret_end)
            if extended:
                current_station_free = max(current_station_free, new_ret_start)
            continue

        # --- new bin fetch: presence must start after picker AND pickface are ready
        try:
            desired_fetch = max(t_cursor, current_station_free) - rt[k]
            t_fetch, copy_id = earliest_feasible_fetch(k, desired_fetch, state, sc)
        except ValueError:
            return None

        fetch_end = t_fetch + rt[k]
        pick_start = fetch_end
        pick_end = pick_start + p[k]
        return_start = pick_end
        return_end = return_start + rt_ret[k]
        if return_end > state.horizon:
            return None
        return_start = sc.delay_move(state, return_start, rt_ret[k])
        return_end = return_start + rt_ret[k]
        if return_end > state.horizon:
            return None

        t_cursor = pick_end
        current_station_free = max(current_station_free, return_start)
        sc.add_new_bin(
            BinEvent(
                sku=k,
                copy_id=copy_id,
                fetch_start=t_fetch,
                fetch_end=fetch_end,
                presence_start=fetch_end,
                presence_end=return_start,
                return_start=return_start,
                return_end=return_end,
                orders_served=[o],
                pick_intervals=[(pick_start, pick_end)],
            )
        )

    start, end = sc.window(t_order_start)
    return sc.to_plan(o, s, best_lane, start, end, alpha * end + beta * (end - start))


# ======================================================================
# Single-item piggybacking
# ======================================================================


def choose_last_sku(o: int, singles_by_sku: dict[int, list[int]], processed: set[int], sku_seq: list[int]) -> int | None:
    """SKU of order o requested by the most unprocessed 1-item orders (None if no SKU has any).

    Ties go to the SKU that already comes latest in the order's sku_rule sequence.
    """
    best_k, best_n = None, 0
    for k in sku_seq:  # later SKUs win ties (>=)
        n = sum(1 for o2 in singles_by_sku.get(k, ()) if o2 != o and o2 not in processed)
        if n > 0 and n >= best_n:
            best_k, best_n = k, n
    return best_k


def bin_of_pick(plan: OrderPlan, k: int) -> BinEvent:
    """The bin visit (new or shared) that served SKU k in a committed plan."""
    for ev in plan.bin_events:
        if ev.sku == k:
            return ev
    for kk, ev, *_ in plan.shared_picks:
        if kk == k:
            return ev
    raise KeyError(k)


def plan_single_from_bin(
    o: int, s: int, lane: int, ev: BinEvent, state: HeuristicState, alpha: float, beta: float
) -> OrderPlan | None:
    """Plan 1-item order o on (s, lane), picking only from bin visit `ev` (extending its presence if needed)."""
    (k,) = state.inst.orders_req[o]
    t_ready = state.lane_free[(s, lane)]
    sc = PlanScratch.for_station(state, s)
    shared, pick_start, new_ret_start, new_ret_end = find_shared_bin([ev], k, t_ready, state, s, sc)
    if shared is None:
        return None
    sc.apply_shared_pick(state, s, k, shared, pick_start, new_ret_start, new_ret_end)
    start, end = sc.window(t_ready)
    return sc.to_plan(o, s, lane, start, end, alpha * end + beta * (end - start))


# ======================================================================
# SGC
# ======================================================================


def run_sgc(
    inst: Instance,
    *,
    horizon: int = 10000,
    alpha: float = 1.0,
    beta: float = 0.0,
    order_rule: Rule = DEFAULT_ORDER_RULE,
    sku_rule: Rule = DEFAULT_SKU_RULE,
    load_balanced: bool = False,
    piggyback_singles: bool = False,
    exo: ExoState | None = None,
) -> HeuristicSolution:
    """Sequential Greedy Construction with the given dispatching rules.

    piggyback_singles: when an order is planned, the SKU wanted by the most unscheduled
    1-item orders goes last; once the order is committed, those 1-item orders run one
    after another on the lane it vacated, all picking from that last bin visit (its
    presence is extended). The chain stops at the first single that cannot draw from
    the bin (pickface, copy, MoveCap or horizon conflict); the rest are scheduled
    normally later. Then SGC continues in order_rule order, skipping scheduled orders.
    """
    order_rule, sku_rule = Rule(*order_rule), Rule(*sku_rule)
    S, p, orders_req = inst.S, inst.p, inst.orders_req
    state = init_state(inst, horizon, exo)

    order_attrs = inst.order_attrs
    sorted_orders = sorted(inst.O, key=lambda o: order_attrs[o][order_rule.attr], reverse=order_rule.descending)

    order_assignments: dict[int, tuple[int, int, int, int]] = {}
    pick_events: dict[tuple[int, int, int], tuple[int, int]] = {}
    failed_orders: list[int] = []
    station_workload: dict[int, float] = {s: 0.0 for s in S}

    # piggybacking bookkeeping: 1-item orders per SKU (in order_rule order), orders already handled
    singles_by_sku: dict[int, list[int]] = {}
    if piggyback_singles:
        for o in sorted_orders:
            if len(orders_req[o]) == 1:
                singles_by_sku.setdefault(orders_req[o][0], []).append(o)
    processed: set[int] = set()
    n_piggybacked = 0

    for o in sorted_orders:
        if o in processed:
            continue
        processed.add(o)
        last_sku = None
        if piggyback_singles:
            seq = sorted(orders_req[o], key=lambda k: inst.sku_attrs[k][sku_rule.attr], reverse=sku_rule.descending)
            last_sku = choose_last_sku(o, singles_by_sku, processed, seq)

        if load_balanced:
            avg_workload = sum(station_workload.values()) / max(len(S), 1)
            by_load = sorted(S, key=lambda s: (station_workload[s], s))
            cand = [s for s in by_load if station_workload[s] <= avg_workload] or by_load[:1]
            fallback = [s for s in by_load if s not in set(cand)]
        else:
            cand, fallback = S, []

        best_plan: OrderPlan | None = None
        for stations in (cand, fallback):
            if best_plan is not None:
                break
            for s in stations:
                plan = plan_order_at_station(o, s, state, alpha, beta, sku_rule, last_sku)
                if plan is not None and (best_plan is None or plan.score < best_plan.score):
                    best_plan = plan

        if best_plan is None:
            failed_orders.append(o)
            continue
        commit_plan(best_plan, state)
        record_plan(best_plan, order_assignments, pick_events)
        station_workload[best_plan.station] += sum(p.get(k, 0) for k in orders_req[o])

        if last_sku is not None:
            # 1-item orders for last_sku ride on the vacated lane, drawing from the last bin visit
            s, lane = best_plan.station, best_plan.lane
            ev = bin_of_pick(best_plan, last_sku)
            for o2 in singles_by_sku[last_sku]:
                if o2 in processed:
                    continue
                plan2 = plan_single_from_bin(o2, s, lane, ev, state, alpha, beta)
                if plan2 is None:
                    break
                commit_plan(plan2, state)
                record_plan(plan2, order_assignments, pick_events)
                station_workload[s] += p.get(last_sku, 0)
                processed.add(o2)
                n_piggybacked += 1

    if failed_orders:
        print(f"[SGC] WARNING: {len(failed_orders)} orders could not be scheduled: {failed_orders}")

    return HeuristicSolution(
        order_assignments=order_assignments,
        bin_events=dict(state.station_bin_events),
        makespan=max((end for _, _, _, end in order_assignments.values()), default=0),
        total_moves=sum(len(evts) * 2 for evts in state.station_bin_events.values()),
        feasible=not failed_orders,
        pick_events=pick_events,
        meta={
            "heuristic": "lb_sgc" if load_balanced else "sgc",
            "order_rule": str(order_rule),
            "sku_rule": str(sku_rule),
            **({"piggyback_singles": True, "n_piggybacked": n_piggybacked} if piggyback_singles else {}),
        },
    )


# ======================================================================
# AMA = best of several SGC runs
# ======================================================================


def run_ama(
    inst: Instance,
    *,
    horizon: int = 10000,
    alpha: float = 1.0,
    beta: float = 0.0,
    mode: str = "full_grid",
    load_balanced: bool = False,
    piggyback_singles: bool = False,
    verbose: bool = False,
) -> HeuristicSolution:
    """Adaptive Multi-Attribute SGC: run SGC per dispatching-rule pair, keep the best.

    ``load_balanced`` / ``piggyback_singles`` are passed to every SGC run.

    ``meta["best_rule"]`` = (order_rule, sku_rule); ``meta["runs"]`` = one record per SGC run.
    """
    runs: list[dict] = []
    best_sol: HeuristicSolution | None = None
    best_obj = float("inf")
    best_rule = (DEFAULT_ORDER_RULE, DEFAULT_SKU_RULE)

    def sgc(order_rule: Rule, sku_rule: Rule, phase: str) -> tuple[HeuristicSolution, float]:
        sol = run_sgc(
            inst, horizon=horizon, alpha=alpha, beta=beta, order_rule=order_rule, sku_rule=sku_rule,
            load_balanced=load_balanced, piggyback_singles=piggyback_singles,
        )
        obj = sol.objective(alpha, beta, inst.S)
        runs.append(
            {
                "phase": phase,
                "order_attr": order_rule.attr,
                "order_desc": order_rule.descending,
                "bin_attr": sku_rule.attr,
                "bin_desc": sku_rule.descending,
                "feasible": sol.feasible,
                "makespan": float(sol.makespan) if sol.feasible else None,
                "total_moves": sol.total_moves if sol.feasible else None,
                "objective": float(obj) if sol.feasible else None,
            }
        )
        return sol, obj

    if mode == "full_grid":
        for order_rule, sku_rule in ALL_RULES:
            sol, obj = sgc(order_rule, sku_rule, "full_grid")
            if best_sol is None or obj < best_obj:
                best_sol, best_obj, best_rule = sol, obj, (order_rule, sku_rule)

    elif mode == "two_phase":
        # Phase 1: best order rule with the SKU rule fixed to demand desc
        best_order, phase1_obj = DEFAULT_ORDER_RULE, float("inf")
        for oa in ORDER_ATTRS:
            for od in DIRECTIONS:
                sol, obj = sgc(Rule(oa, od), DEFAULT_SKU_RULE, "two_phase_1")
                if best_sol is None or obj < phase1_obj:
                    phase1_obj, best_sol, best_order = obj, sol, Rule(oa, od)
        # Phase 2: best SKU rule for that order rule
        for ba in BIN_ATTRS:
            for bd in DIRECTIONS:
                sol, obj = sgc(best_order, Rule(ba, bd), "two_phase_2")
                if obj < best_obj:
                    best_sol, best_obj, best_rule = sol, obj, (best_order, Rule(ba, bd))
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Choose 'full_grid' or 'two_phase'.")

    if verbose:
        print(f"[AMA-SGC] Best config: order={best_rule[0]}, bin={best_rule[1]}, obj={best_obj:.1f}")

    assert best_sol is not None
    best_sol.meta.update({"heuristic": "ama", "mode": mode, "best_rule": best_rule, "runs": runs,
                          "load_balanced": load_balanced, "piggyback_singles": piggyback_singles})
    return best_sol
