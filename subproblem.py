"""Order-based LNS subproblems.

Instead of freezing ~96 % of the full CP model with constraints (which still has to be
extracted every iteration), each LNS iteration builds a small CP model that contains
only the *active* orders chosen by the neighbourhood. Everything else in the incumbent
stays fixed and enters the subproblem as constants:

  exo_lanes         frozen orders' lane windows            -> lane no_overlap
  exo_station_bins  frozen bin presences at each station   -> one-bin-per-station no_overlap
  exo_blocks        frozen bin cycles (fetch..return)      -> per-SKU bin copy limit N[k]
  exo_moves         frozen fetches and returns             -> MoveCap
  ObjectiveOffsets  frozen part of every KPI               -> exact global objective values
  exo_visits        frozen visits active orders may join   -> PX picks, frozen picks in PickCap

PickCap needs exo data only for the joinable visits: picks lie inside their bin's presence
window and bins at a station never overlap, so only picks on the same frozen visit can
overlap frozen ones.

A bin visit (F, B, R of one (s, k, e)) is frozen if it serves at least one frozen order:
its fetch, presence and return are fixed. Active orders can still pick from it
(``exo_visits``): each gets an optional pick PX[o,s,k,e] inside the fixed presence window,
as one more candidate of alternative(C[o,k,s], ...). The visit's own variables stay
frozen (see docs/KNOWN_ISSUES.md, "Subproblem LNS: joining frozen bin visits").

With the graph neighbourhood (``extract_context_graph``, default in the LNS) the
subproblem frees exactly the variables the full-model LNS frees: frozen orders that pick
from a freed visit are added with all their variables fixed (pinned), active orders'
picks on frozen visits are fixed, and surplus copy slots are fixed absent.

The subproblem result is written back into a full-model solution dict
(``CppSolveResult``), so strategies, logging and acceptance are unchanged.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from cp_model_utils import CppSolveResult, SearchTimeLimitCallback, cp_time_limits
from heuristics import HeuristicSolution, validate_solution
from heuristics.heuristic_utils import BinEvent

ABSENT = {"present": False}


def _iv(start: int, end: int) -> dict:
    return {"present": True, "start": int(start), "end": int(end), "size": int(end - start)}


@dataclass
class ObjectiveOffsets:
    """Contribution of the frozen orders to each KPI of the full model."""

    makespan_lb: int = 0  # latest end of a frozen order
    bin_fetches: int = 0  # number of frozen bin visits
    total_flow_time: int = 0  # sum of frozen order ends
    station_window: dict = field(default_factory=dict)  # s -> (first start, last end) of frozen orders
    station_pick_time: dict = field(default_factory=dict)  # s -> total frozen pick duration
    station_pick_count: dict = field(default_factory=dict)  # s -> number of frozen picks

    def lower_bound(self, kpi: str) -> int | None:
        """Lower bound on the full-model KPI implied by the frozen part alone."""
        if kpi == "makespan":
            return self.makespan_lb
        if kpi == "bin_fetches":
            return self.bin_fetches
        if kpi == "total_flow_time":
            return self.total_flow_time
        if kpi == "max_picks":
            return max(self.station_pick_count.values(), default=0)
        if kpi == "sum_makespans":
            return sum(end for _, end in self.station_window.values())
        return None


class SubInstance:
    """Minimal stand-in for ``Instance`` with only the fields the model builders read."""

    def __init__(self, S, L, K, orders_req, rt, p, N, movecap, pickcap, rt_ret):
        self.S, self.L, self.K = S, L, K
        self.orders_req = orders_req
        self.rt, self.p, self.N = rt, p, N
        self.movecap, self.pickcap, self.rt_ret = movecap, pickcap, rt_ret

    def __iter__(self):
        yield from (self.S, self.L, self.K, self.orders_req, self.rt, self.p, self.N)


@dataclass
class FrozenContext:
    """Everything the subproblem needs to know about the incumbent."""

    active_orders: list
    instance: SubInstance
    exo_lanes: dict
    exo_station_bins: dict
    exo_blocks: dict
    exo_moves: list
    offsets: ObjectiveOffsets
    warm: HeuristicSolution  # incumbent restricted to the active orders
    free_visits: list  # (s, k, e) visits served only by active orders (re-planned)
    frozen_slots: dict  # (s, k) -> set of copy indices e held by frozen visits
    shared_picks: list  # (o, s, k, e): active order picking from a frozen visit
    # Joinable frozen visits: (s, k, e, b_start, b_end, frozen_picks). Active orders
    # get optional picks PX[o,s,k,e] inside the fixed presence window.
    exo_visits: list = field(default_factory=list)
    warm_px: dict = field(default_factory=dict)  # (o, s, k, e) -> incumbent pick on an exo visit
    # Graph neighbourhood (extract_context_graph), None/empty otherwise:
    pinned: set = field(default_factory=set)  # frozen orders on free visits, all variables fixed
    slot_limit: dict | None = None  # (s, k) -> number of free copy slots; others absent
    slot_pool: dict = field(default_factory=dict)  # (s, k) -> full-model copy indices freed
    pin_slot: dict = field(default_factory=dict)  # (o, s, k) -> full copy index of a pinned pick
    incumbent: dict = field(default_factory=dict)  # var_sols of the incumbent (fixed values)


# --------------------------------------------------------------------------- #
# Reading the incumbent
# --------------------------------------------------------------------------- #
def _present(var_sols, var):
    vs = var_sols.get(var.get_name())
    if vs is not None and vs.get("present"):
        return vs["start"], vs["end"]
    return None


def close_under_sharing(solution, handles, active_orders) -> list:
    """Add every order that shares a bin visit with an active order (fixed point).

    After the closure no bin visit serves both active and frozen orders, so the
    incumbent restricted to the active orders is a complete, feasible warm start
    (``FrozenContext.shared_picks`` is empty) and the subproblem can always
    reproduce the incumbent's KPIs.
    """
    vs = solution.var_sols
    P, B = handles["P"], handles["B"]
    demand = handles["orders_demanding_sku"]
    active = set(active_orders)
    visits = []  # orders served by each shared visit
    for (s, k, e), b_var in B.items():
        if _present(vs, b_var) is None:
            continue
        served = [o for o in demand[k] if _present(vs, P[(o, s, k, e)]) is not None]
        if len(served) > 1:
            visits.append(set(served))
    changed = True
    while changed:
        changed = False
        for served in visits:
            if served & active and not served <= active:
                active |= served
                changed = True
    return sorted(active)


def _pick_fits(window, frozen_picks, length, pickcap) -> bool:
    """True if a pick of ``length`` fits in ``window`` next to the frozen picks (PickCap)."""
    points = sorted({*window, *(t for iv in frozen_picks for t in iv)})
    run = 0
    for a, b in zip(points, points[1:]):
        load = sum(1 for st, en in frozen_picks if st <= a and b <= en)
        run = run + (b - a) if load < pickcap else 0
        if run >= length:
            return True
    return False


def extract_context(solution, handles, instance, active_orders,
                    joinable_exo: bool = True) -> FrozenContext:
    """Split the incumbent into the active part (warm start) and the frozen part (exo data).

    ``joinable_exo``: frozen visits of SKUs the active orders need are also passed as
    ``exo_visits``, so active orders can pick from them (False: behaviour before Nov 2026,
    active picks on frozen visits are dropped from the warm start).
    """
    vs = solution.var_sols
    active = set(active_orders)
    S, L = handles["S"], handles["L"]
    I_os, I_os_lane = handles["I_os"], handles["I_os_lane"]
    P, F, R, B, Block = (handles[x] for x in ("P", "F", "R", "B", "Block"))
    demand = handles["orders_demanding_sku"]
    p = handles["p"]

    offsets = ObjectiveOffsets()
    exo_lanes = defaultdict(list)
    window = {}
    assignments = {}

    for o in handles["O"]:
        s_sel = next((s for s in S if _present(vs, I_os[(o, s)])), None)
        if s_sel is None:
            raise ValueError(f"Order {o} is not scheduled in the incumbent")
        st, en = _present(vs, I_os[(o, s_sel)])
        ln_sel = next(ln for ln in L if _present(vs, I_os_lane[(o, s_sel, ln)]))
        assignments[o] = (s_sel, ln_sel, st, en)
        if o in active:
            continue
        exo_lanes[(s_sel, ln_sel)].append((st, en))
        offsets.makespan_lb = max(offsets.makespan_lb, en)
        offsets.total_flow_time += en
        lo, hi = window.get(s_sel, (st, en))
        window[s_sel] = (min(lo, st), max(hi, en))
    offsets.station_window = window

    exo_station_bins = defaultdict(list)
    exo_blocks = defaultdict(list)
    exo_moves = []
    frozen_slots = defaultdict(set)
    free_visits, shared_picks = [], []
    warm_events = defaultdict(list)
    warm_picks = {}
    exo_visits, warm_px = [], {}
    sub_skus = {k for o in active for k in handles["orders_req"][o]}

    for (s, k, e), b_var in B.items():
        b = _present(vs, b_var)
        if b is None:
            continue
        picks = {}
        for o in demand[k]:
            iv = _present(vs, P[(o, s, k, e)])
            if iv is not None:
                picks[o] = iv
        f = _present(vs, F[(s, k, e)])
        r = _present(vs, R[(s, k, e)])
        frozen_served = [o for o in picks if o not in active]

        if frozen_served:
            exo_station_bins[s].append(b)
            exo_blocks[k].append(_present(vs, Block[(s, k, e)]))
            exo_moves.extend([f, r])
            frozen_slots[(s, k)].add(e)
            offsets.bin_fetches += 1
            for o in frozen_served:
                offsets.station_pick_time[s] = offsets.station_pick_time.get(s, 0) + p[k]
                offsets.station_pick_count[s] = offsets.station_pick_count.get(s, 0) + 1
            shared = [o for o in picks if o in active]
            shared_picks.extend((o, s, k, e) for o in shared)
            if joinable_exo and k in sub_skus:
                frozen_picks = sorted(picks[o] for o in frozen_served)
                # Exact pruning: skip visits where no further pick can fit
                if _pick_fits(b, frozen_picks, p[k], instance.pickcap):
                    exo_visits.append((s, k, e, b[0], b[1], frozen_picks))
                    for o in shared:
                        warm_px[(o, s, k, e)] = picks[o]
        else:
            free_visits.append((s, k, e))
            served = sorted(picks)
            warm_events[s].append(
                BinEvent(sku=k, copy_id=e, fetch_start=f[0], fetch_end=f[1],
                         presence_start=b[0], presence_end=b[1],
                         return_start=r[0], return_end=r[1], orders_served=served,
                         pick_intervals=sorted(picks.values()))
            )
            for o in served:
                warm_picks[(o, s, k)] = picks[o]

    # Reduced instance: active orders and the SKUs they need
    orders_req = {o: list(handles["orders_req"][o]) for o in sorted(active)}
    K_sub = sorted(sub_skus)
    sub = SubInstance(
        S=S, L=L, K=K_sub, orders_req=orders_req,
        rt={k: handles["rt"][k] for k in K_sub},
        p={k: p[k] for k in K_sub},
        N={k: handles["N"][k] for k in K_sub},
        movecap=instance.movecap, pickcap=instance.pickcap,
        rt_ret={k: handles["rt_return"][k] for k in K_sub},
    )

    warm_assign = {o: assignments[o] for o in active}
    warm = HeuristicSolution(
        order_assignments=warm_assign,
        bin_events=dict(warm_events),
        makespan=max((a[3] for a in warm_assign.values()), default=0),
        total_moves=2 * sum(len(v) for v in warm_events.values()),
        feasible=True,
        pick_events=warm_picks,
        meta={"heuristic": "lns_subproblem_warmstart"},
    )

    return FrozenContext(
        active_orders=sorted(active),
        instance=sub,
        exo_lanes=dict(exo_lanes),
        exo_station_bins=dict(exo_station_bins),
        exo_blocks={k: v for k, v in exo_blocks.items() if k in sub_skus},
        exo_moves=exo_moves,
        offsets=offsets,
        warm=warm,
        free_visits=free_visits,
        frozen_slots=frozen_slots,
        shared_picks=shared_picks,
        exo_visits=exo_visits,
        warm_px=warm_px,
    )


def extract_context_graph(solution, handles, instance, active_orders, free_visits
                          ) -> FrozenContext:
    """Context for exactly the full model's neighbourhood (``get_optimized_set``).

    Free: I_os, I_os_lane and C of the active orders; F, B, R, Block of ``free_visits``
    (full-model (s, k, e)); P[o,s,k,e] for an active order o on a free visit. Everything
    else is frozen, as in the full-model LNS:
    - a free visit may still carry picks of frozen orders. Those orders are *pinned*:
      they enter the subproblem with every variable fixed, so their fixed picks keep
      constraining the free visit (F -> P -> R, PickCap, presence);
    - an active order's pick on a frozen visit is fixed (PX fixed to the incumbent,
      no joining); on any other frozen visit it stays absent;
    - per (s, k) the subproblem has as many free copy slots as ``free_visits`` has
      there; the remaining slots are fixed absent. Free slots are interchangeable
      (no symmetry breaking), so this is the full neighbourhood up to relabelling the
      copies; a pinned pick keeps its visit's copy index (merge).
    """
    vs = solution.var_sols
    active = set(active_orders)
    V = set(free_visits)
    S, L = handles["S"], handles["L"]
    I_os, I_os_lane = handles["I_os"], handles["I_os_lane"]
    P, F, R, B, Block = (handles[x] for x in ("P", "F", "R", "B", "Block"))
    demand = handles["orders_demanding_sku"]
    p = handles["p"]

    visits = {}  # present (s, k, e) -> (presence, {order: pick})
    for key, b_var in B.items():
        b = _present(vs, b_var)
        if b is None:
            continue
        s, k, e = key
        picks = {}
        for o in demand[k]:
            iv = _present(vs, P[(o, s, k, e)])
            if iv is not None:
                picks[o] = iv
        visits[key] = (b, picks)
    pinned = {o for key, (_, picks) in visits.items() if key in V for o in picks} - active
    sub_orders = active | pinned

    offsets = ObjectiveOffsets()
    exo_lanes = defaultdict(list)
    window = {}
    assignments = {}
    for o in handles["O"]:
        s_sel = next((s for s in S if _present(vs, I_os[(o, s)])), None)
        if s_sel is None:
            raise ValueError(f"Order {o} is not scheduled in the incumbent")
        st, en = _present(vs, I_os[(o, s_sel)])
        ln_sel = next(ln for ln in L if _present(vs, I_os_lane[(o, s_sel, ln)]))
        assignments[o] = (s_sel, ln_sel, st, en)
        if o in sub_orders:
            continue
        exo_lanes[(s_sel, ln_sel)].append((st, en))
        offsets.makespan_lb = max(offsets.makespan_lb, en)
        offsets.total_flow_time += en
        lo, hi = window.get(s_sel, (st, en))
        window[s_sel] = (min(lo, st), max(hi, en))
    offsets.station_window = window

    sub_skus = {k for o in sub_orders for k in handles["orders_req"][o]}
    exo_station_bins = defaultdict(list)
    exo_blocks = defaultdict(list)
    exo_moves = []
    frozen_slots = defaultdict(set)
    free_present, warm_events, warm_picks = [], defaultdict(list), {}
    exo_visits, warm_px, pin_slot = [], {}, {}
    for key, (b, picks) in visits.items():
        s, k, e = key
        f = _present(vs, F[key])
        r = _present(vs, R[key])
        if key in V:
            free_present.append(key)
            served = sorted(picks)
            warm_events[s].append(
                BinEvent(sku=k, copy_id=e, fetch_start=f[0], fetch_end=f[1],
                         presence_start=b[0], presence_end=b[1],
                         return_start=r[0], return_end=r[1], orders_served=served,
                         pick_intervals=sorted(picks.values()))
            )
            for o in served:
                warm_picks[(o, s, k)] = picks[o]
                if o in pinned:
                    pin_slot[(o, s, k)] = e
            continue
        exo_station_bins[s].append(b)
        exo_blocks[k].append(_present(vs, Block[key]))
        exo_moves.extend([f, r])
        frozen_slots[(s, k)].add(e)
        offsets.bin_fetches += 1
        outside = [o for o in picks if o not in sub_orders]
        for o in outside:
            offsets.station_pick_time[s] = offsets.station_pick_time.get(s, 0) + p[k]
            offsets.station_pick_count[s] = offsets.station_pick_count.get(s, 0) + 1
        inside = [o for o in picks if o in sub_orders]
        if inside:  # fixed picks of subproblem orders on a frozen visit
            exo_visits.append((s, k, e, b[0], b[1], sorted(picks[o] for o in outside)))
            for o in inside:
                warm_px[(o, s, k, e)] = picks[o]

    slot_pool = defaultdict(list)
    for (s, k, e) in sorted(V):
        if k in sub_skus:
            slot_pool[(s, k)].append(e)

    orders_req = {o: list(handles["orders_req"][o]) for o in sorted(sub_orders)}
    K_sub = sorted(sub_skus)
    sub = SubInstance(
        S=S, L=L, K=K_sub, orders_req=orders_req,
        rt={k: handles["rt"][k] for k in K_sub},
        p={k: p[k] for k in K_sub},
        N={k: handles["N"][k] for k in K_sub},
        movecap=instance.movecap, pickcap=instance.pickcap,
        rt_ret={k: handles["rt_return"][k] for k in K_sub},
    )
    warm_assign = {o: assignments[o] for o in sub_orders}
    warm = HeuristicSolution(
        order_assignments=warm_assign,
        bin_events=dict(warm_events),
        makespan=max((a[3] for a in warm_assign.values()), default=0),
        total_moves=2 * sum(len(v) for v in warm_events.values()),
        feasible=True,
        pick_events=warm_picks,
        meta={"heuristic": "lns_subproblem_warmstart_graph"},
    )
    return FrozenContext(
        active_orders=sorted(sub_orders),
        instance=sub,
        exo_lanes=dict(exo_lanes),
        exo_station_bins=dict(exo_station_bins),
        exo_blocks={k: v for k, v in exo_blocks.items() if k in sub_skus},
        exo_moves=exo_moves,
        offsets=offsets,
        warm=warm,
        free_visits=free_present,
        frozen_slots=frozen_slots,
        shared_picks=[],
        exo_visits=exo_visits,
        warm_px=warm_px,
        pinned=pinned,
        slot_limit={key: len(v) for key, v in slot_pool.items()},
        slot_pool=dict(slot_pool),
        pin_slot=pin_slot,
        incumbent=vs,
    )


def graph_fixings(ctx: FrozenContext, handles) -> list:
    """(subproblem variable, value) pairs that ``extract_context_graph`` keeps frozen."""
    fix = []
    vs = ctx.incumbent
    U = handles["U"]
    # Copy slots beyond the freed ones are absent
    for (s, k, e) in handles["F"]:
        if e >= ctx.slot_limit.get((s, k), 0):
            fix += [(handles[g][(s, k, e)], ABSENT) for g in ("F", "B", "R", "Block")]
    # Picks on frozen visits are fixed (present as in the incumbent, else absent)
    for key, var in handles["PX"].items():
        iv = ctx.warm_px.get(key)
        fix.append((var, _iv(*iv) if iv is not None else ABSENT))
    # Pinned orders: everything fixed. Their picks on free visits sit in the slot that
    # inject_warmstart gives the visit (rank by fetch start among the (s, k) visits).
    slot_of = {}
    for s, events in ctx.warm.bin_events.items():
        by_sku = defaultdict(list)
        for be in events:
            by_sku[be.sku].append(be)
        for k, evts in by_sku.items():
            for rank, be in enumerate(sorted(evts, key=lambda x: x.fetch_start)):
                for o in be.orders_served:
                    slot_of[(o, s, k)] = rank
    for o in ctx.pinned:
        for (oo, s), var in handles["I_os"].items():
            if oo == o:
                fix.append((var, vs.get(var.get_name(), ABSENT)))
        for (oo, s, ln), var in handles["I_os_lane"].items():
            if oo == o:
                fix.append((var, vs.get(var.get_name(), ABSENT)))
        for (oo, k, s), var in handles["C"].items():
            if oo == o:
                fix.append((var, vs.get(var.get_name(), ABSENT)))
                for e in range(U[k]):
                    pick = ctx.warm.pick_events.get((o, s, k))
                    present = pick is not None and slot_of.get((o, s, k)) == e
                    fix.append((handles["P"][(o, s, k, e)], _iv(*pick) if present else ABSENT))
    return fix


def apply_fixings(mdl, fixings, backend):
    """Fix subproblem variables (constraints, like the full-model LNS freezing)."""
    if backend == "docplex":
        for var, value in fixings:
            if value.get("present"):
                mdl.add(mdl.presence_of(var) == 1)
                mdl.add(mdl.start_of(var) == value["start"])
                mdl.add(mdl.end_of(var) == value["end"])
            else:
                mdl.add(mdl.presence_of(var) == 0)
        return
    names = [var.get_name() for var, _ in fixings]
    present = [1 if v.get("present") else 0 for _, v in fixings]
    start = [v["start"] if v.get("present") else 0 for _, v in fixings]
    end = [v["end"] if v.get("present") else 0 for _, v in fixings]
    mdl.apply_delta_freezing(names, present, start, end, [])


# --------------------------------------------------------------------------- #
# Building and solving
# --------------------------------------------------------------------------- #
def build_submodel(ctx: FrozenContext, backend: str, horizon: int, objective: str):
    """Build the reduced CP model and set the incumbent as its starting point."""
    kwargs = dict(
        add_symmetry_breaking=False,  # lanes are not symmetric next to frozen orders
        horizon=horizon,
        exo_lanes=ctx.exo_lanes,
        exo_blocks=ctx.exo_blocks,
        exo_moves=ctx.exo_moves,
        objective_func=objective,
        exo_station_bins=ctx.exo_station_bins,
        obj_offsets=ctx.offsets,
        exo_visits=ctx.exo_visits,
    )
    import cp_model

    if backend == "docplex":
        mdl, handles = cp_model.build_model(ctx.instance, **kwargs)
        sp_model = mdl
    elif backend == "cpp":
        import cpp_cp_model

        mdl, handles = cpp_cp_model.build_model(ctx.instance, **kwargs)
        sp_model = cpp_cp_model.MockCpoModel()
    else:
        raise ValueError(backend)
    mdl.set_starting_point(warm_start(ctx, sp_model, handles))
    if ctx.slot_limit is not None:
        apply_fixings(mdl, graph_fixings(ctx, handles), backend)
    return mdl, handles


def warm_start(ctx: FrozenContext, sp_model, handles):
    """Complete starting point of the subproblem: the incumbent of the active orders."""
    import cp_model

    sp = cp_model.inject_warmstart(ctx.warm, ctx.warm.pick_events, sp_model, handles,
                                   remap_lanes=False)
    # Incumbent picks of active orders on joinable frozen visits (set after the sweep
    # of inject_warmstart, which marked their consumptions absent)
    for (o, s, k, e), var in handles["PX"].items():
        iv = ctx.warm_px.get((o, s, k, e))
        if iv is None:
            sp.add_interval_var_solution(var, presence=False)
        else:
            sp.add_interval_var_solution(var, presence=True, start=iv[0], end=iv[1])
            sp.add_interval_var_solution(handles["C"][(o, k, s)], presence=True,
                                         start=iv[0], end=iv[1])
    return sp


def solve_submodel(mdl, handles, backend, objective, impr_bound, cfg, progress_callback=None,
                   limits=None):
    """Solve the reduced model. Returns (result, progress records).

    ``limits`` = (engine TimeLimit, search-only limit), see ``cp_time_limits``;
    computed from ``cfg`` without a run budget when omitted.
    """
    time_limit, search_limit = limits if limits is not None else cp_time_limits(cfg)
    if backend == "docplex":
        if impr_bound != -1:
            mdl.add(handles[objective] <= impr_bound)
        kwargs = {
            "LogVerbosity": "Quiet",
            "Workers": cfg["workers"],
            "Presolve": cfg["presolve"],
            "SearchType": cfg["search_type"],
        }
        if time_limit is not None:
            kwargs["TimeLimit"] = time_limit
        if cfg.get("only_first_sol", False):
            kwargs["SolutionLimit"] = 1
        if search_limit is not None:
            mdl.add_solver_callback(SearchTimeLimitCallback(search_limit))
        progress = []
        if progress_callback is not None:
            mdl.add_solver_callback(progress_callback)
            progress_callback.reset()
        sol = mdl.solve(**kwargs)
        if progress_callback is not None:
            progress = progress_callback.records
        return sol, progress

    res = mdl.solve(
        time_limit if time_limit is not None else -1,
        impr_bound,
        cfg["workers"],
        cfg["presolve"],
        cfg["search_type"],
        cfg.get("only_first_sol", False),
        collect_progress=True,
        collect_deltas=False,
        search_time_limit=search_limit if search_limit is not None else -1,
    )
    return res, res.get("progress") or []


def _timings(sol, backend):
    """(total, extraction, search) engine times of a subproblem solve."""
    if backend == "cpp":
        return sol.total_time, sol.extraction_time, sol.solve_time
    total = sol.get_solve_time() or 0.0
    extraction = sol.get_info("ExtractionTime") or 0.0
    return total, extraction, max(0.0, total - extraction)


# --------------------------------------------------------------------------- #
# Writing the result back into the full solution
# --------------------------------------------------------------------------- #
def merge(solution, ctx: FrozenContext, sub_sol, sub_handles, handles, written=None) -> dict:
    """Full-model var_sols with the active orders replaced by the subproblem result.

    ``written``: optional set that receives the name of every variable the merge sets
    (a superset of the changed ones; the solution delta log diffs over it).
    """
    hs = HeuristicSolution.from_cp(sub_sol, sub_handles)
    new = dict(solution.var_sols)
    S, L, U = handles["S"], handles["L"], handles["U"]
    I_os, I_os_lane, C, P = (handles[x] for x in ("I_os", "I_os_lane", "C", "P"))
    F, R, B, Block = (handles[x] for x in ("F", "R", "B", "Block"))
    orders_req = handles["orders_req"]

    def put(var, value):
        new[var.get_name()] = value
        if written is not None:
            written.add(var.get_name())

    # 1. Clear the old schedule of the active orders and of their own bin visits
    for o in ctx.active_orders:
        for s in S:
            put(I_os[(o, s)], ABSENT)
            for ln in L:
                put(I_os_lane[(o, s, ln)], ABSENT)
            for k in orders_req[o]:
                put(C[(o, k, s)], ABSENT)
                for e in range(U[k]):
                    put(P[(o, s, k, e)], ABSENT)
    for key in ctx.free_visits:
        for dic in (F, B, R, Block):
            put(dic[key], ABSENT)

    # 2. Orders
    for o, (s, ln, st, en) in hs.order_assignments.items():
        put(I_os[(o, s)], _iv(st, en))
        put(I_os_lane[(o, s, ln)], _iv(st, en))
        for k in orders_req[o]:
            # from C, not hs.pick_events: picks on exo visits (PX) are not in hs
            c = sub_sol.get_var_solution(sub_handles["C"][(o, k, s)])
            put(C[(o, k, s)], _iv(c.get_start(), c.get_end()))

    # 3. Bin visits: take copy indices not held by frozen visits
    for s, events in hs.bin_events.items():
        by_sku = defaultdict(list)
        for be in events:
            by_sku[be.sku].append(be)
        for k, evts in by_sku.items():
            evts.sort(key=lambda be: be.fetch_start)
            if ctx.slot_limit is not None:
                # graph neighbourhood: a visit with a pinned pick keeps its copy index,
                # the others take the remaining freed indices
                anchored = {}
                for be in evts:
                    for o in be.orders_served:
                        if o in ctx.pinned:
                            anchored[id(be)] = ctx.pin_slot[(o, s, k)]
                pool = iter(e for e in ctx.slot_pool.get((s, k), [])
                            if e not in set(anchored.values()))
                try:
                    slots = [anchored[id(be)] if id(be) in anchored else next(pool)
                             for be in evts]
                except StopIteration:
                    raise RuntimeError(f"No free copy index for SKU {k} at station {s}")
            else:
                taken = ctx.frozen_slots.get((s, k), set())
                slots = [e for e in range(U[k]) if e not in taken]
            if len(evts) > len(slots):
                raise RuntimeError(f"No free copy index for SKU {k} at station {s}")
            for be, e in zip(evts, slots):
                put(F[(s, k, e)], _iv(be.fetch_start, be.fetch_end))
                put(B[(s, k, e)], _iv(be.presence_start, be.presence_end))
                put(R[(s, k, e)], _iv(be.return_start, be.return_end))
                put(Block[(s, k, e)], _iv(be.fetch_start, be.return_end))
                for o in be.orders_served:
                    put(P[(o, s, k, e)], _iv(*hs.pick_events[(o, s, k)]))

    # 4. Picks on joinable frozen visits: e is the visit's copy index in the full model
    for key, var in sub_handles["PX"].items():
        px = sub_sol.get_var_solution(var)
        if px is not None and px.is_present():
            put(P[key], _iv(px.get_start(), px.get_end()))
    return new


def violates_bound(ctx: FrozenContext, objective: str, impr_bound: int) -> bool:
    """True if the frozen part alone already breaks the improvement bound (no need to solve)."""
    lb = ctx.offsets.lower_bound(objective)
    return impr_bound != -1 and lb is not None and lb > impr_bound


def infeasible_result() -> CppSolveResult:
    return CppSolveResult({"status": "Infeasible", "var_solutions": {}, "objective": None,
                           "TotalTime": 0.0, "ExtractionTime": 0.0, "SolveTime": 0.0})


def to_full_result(solution, ctx, sub_sol, sub_handles, handles, backend, objective):
    """Wrap a subproblem result as a CppSolveResult on the full model (merging if solved)."""
    status = sub_sol.get_solve_status()
    total, extraction, search = _timings(sub_sol, backend)
    result = {
        "status": status,
        "TotalTime": total,
        "ExtractionTime": extraction,
        "SolveTime": search,
        "var_solutions": {},
        "objective": None,
    }
    if status in ("Optimal", "Feasible"):
        written = set()
        result["var_solutions"] = merge(solution, ctx, sub_sol, sub_handles, handles, written)
        result["written_vars"] = written  # for the solution delta log
        kpis = dict(sub_sol.get_kpis() or {})
        result["kpis"] = {k: kpis[k] for k in
                          ("makespan", "bin_fetches", "total_flow_time", "idle_time", "max_picks",
                           "sum_makespans")
                          if k in kpis}
        result["objective"] = result["kpis"].get(objective)
    return CppSolveResult(result)


def solve_iteration(solution, handles, instance, active_orders, *, backend, horizon,
                    objective, impr_bound, cfg, progress_callback=None, limits=None,
                    joinable_exo=True, free_visits=None):
    """One complete subproblem iteration: extract, build, solve, merge.

    ``free_visits`` given: the graph neighbourhood (``extract_context_graph``).
    Returns (CppSolveResult on the full model, progress records, FrozenContext).
    """
    if free_visits is not None:
        ctx = extract_context_graph(solution, handles, instance, active_orders, free_visits)
    else:
        ctx = extract_context(solution, handles, instance, active_orders, joinable_exo)
    if violates_bound(ctx, objective, impr_bound):
        return infeasible_result(), [], ctx
    mdl, sub_handles = build_submodel(ctx, backend, horizon, objective)
    sub_sol, progress = solve_submodel(mdl, sub_handles, backend, objective, impr_bound, cfg,
                                       progress_callback, limits)
    result = to_full_result(solution, ctx, sub_sol, sub_handles, handles, backend, objective)
    return result, progress, ctx


def check_merged(result, handles, instance, horizon) -> list[str]:
    """Validate a merged full solution and its reported KPIs (slow; for debugging/tests)."""
    hs = HeuristicSolution.from_cp(result, handles)
    problems = list(validate_solution(hs, instance, horizon=horizon))
    kpis = result.get_kpis()
    station_end = defaultdict(int)
    for s, _, _, end in hs.order_assignments.values():
        station_end[s] = max(station_end[s], end)
    expected = {
        "makespan": hs.makespan,
        "bin_fetches": sum(len(v) for v in hs.bin_events.values()),
        "total_flow_time": sum(a[3] for a in hs.order_assignments.values()),
        "sum_makespans": sum(station_end.values()),
    }
    for name, value in expected.items():
        if name in kpis and abs(kpis[name] - value) > 1e-6:
            problems.append(f"KPI {name}: reported {kpis[name]}, recomputed {value}")
    missing = set(handles["O"]) - set(hs.order_assignments)
    if missing:
        problems.append(f"Orders not scheduled after merge: {sorted(missing)[:10]}")
    return problems
