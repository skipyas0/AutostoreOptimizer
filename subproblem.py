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

PickCap needs no exo data: picks lie inside their bin's presence window and bins at a
station never overlap, so new picks cannot overlap frozen ones.

A bin visit is frozen if it serves at least one frozen order. Active orders that
currently share such a visit lose access to it (see docs/KNOWN_ISSUES.md,
"Subproblem LNS: shared bin visits").

The subproblem result is written back into a full-model solution dict
(``CppSolveResult``), so strategies, logging and acceptance are unchanged.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from cp_model_utils import CppSolveResult
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


# --------------------------------------------------------------------------- #
# Reading the incumbent
# --------------------------------------------------------------------------- #
def _present(var_sols, var):
    vs = var_sols.get(var.get_name())
    if vs is not None and vs.get("present"):
        return vs["start"], vs["end"]
    return None


def extract_context(solution, handles, instance, active_orders) -> FrozenContext:
    """Split the incumbent into the active part (warm start) and the frozen part (exo data)."""
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
            shared_picks.extend((o, s, k, e) for o in picks if o in active)
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
    sub_skus = {k for req in orders_req.values() for k in req}
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
    )


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
    )
    if backend == "docplex":
        import cp_model

        mdl, handles = cp_model.build_model(ctx.instance, **kwargs)
        sp = cp_model.inject_warmstart(ctx.warm, ctx.warm.pick_events, mdl, handles,
                                       remap_lanes=False)
        mdl.set_starting_point(sp)
    elif backend == "cpp":
        import cpp_cp_model

        mdl, handles = cpp_cp_model.build_model(ctx.instance, **kwargs)
        cpp_cp_model.inject_warmstart(ctx.warm, ctx.warm.pick_events, mdl, handles,
                                      remap_lanes=False)
    else:
        raise ValueError(backend)
    return mdl, handles


def solve_submodel(mdl, handles, backend, objective, impr_bound, cfg, progress_callback=None):
    """Solve the reduced model. Returns (result, progress records)."""
    if backend == "docplex":
        if impr_bound != -1:
            mdl.add(handles[objective] <= impr_bound)
        kwargs = {
            "TimeLimit": cfg["iter_time_limit"],
            "LogVerbosity": "Quiet",
            "Workers": cfg["workers"],
            "Presolve": cfg["presolve"],
            "SearchType": cfg["search_type"],
        }
        if cfg.get("only_first_sol", False):
            kwargs["SolutionLimit"] = 1
        progress = []
        if progress_callback is not None:
            mdl.add_solver_callback(progress_callback)
            progress_callback.reset()
        sol = mdl.solve(**kwargs)
        if progress_callback is not None:
            progress = progress_callback.records
        return sol, progress

    res = mdl.solve(
        cfg["iter_time_limit"],
        impr_bound,
        cfg["workers"],
        cfg["presolve"],
        cfg["search_type"],
        cfg.get("only_first_sol", False),
        collect_progress=True,
        collect_deltas=False,
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
def merge(solution, ctx: FrozenContext, sub_sol, sub_handles, handles) -> dict:
    """Full-model var_sols with the active orders replaced by the subproblem result."""
    hs = HeuristicSolution.from_cp(sub_sol, sub_handles)
    new = dict(solution.var_sols)
    S, L, U = handles["S"], handles["L"], handles["U"]
    I_os, I_os_lane, C, P = (handles[x] for x in ("I_os", "I_os_lane", "C", "P"))
    F, R, B, Block = (handles[x] for x in ("F", "R", "B", "Block"))
    orders_req = handles["orders_req"]

    def put(var, value):
        new[var.get_name()] = value

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
            ps, pe = hs.pick_events[(o, s, k)]
            put(C[(o, k, s)], _iv(ps, pe))

    # 3. Bin visits: take copy indices not held by frozen visits
    for s, events in hs.bin_events.items():
        by_sku = defaultdict(list)
        for be in events:
            by_sku[be.sku].append(be)
        for k, evts in by_sku.items():
            taken = ctx.frozen_slots.get((s, k), set())
            slots = [e for e in range(U[k]) if e not in taken]
            evts.sort(key=lambda be: be.fetch_start)
            if len(evts) > len(slots):
                raise RuntimeError(f"No free copy index for SKU {k} at station {s}")
            for be, e in zip(evts, slots):
                put(F[(s, k, e)], _iv(be.fetch_start, be.fetch_end))
                put(B[(s, k, e)], _iv(be.presence_start, be.presence_end))
                put(R[(s, k, e)], _iv(be.return_start, be.return_end))
                put(Block[(s, k, e)], _iv(be.fetch_start, be.return_end))
                for o in be.orders_served:
                    put(P[(o, s, k, e)], _iv(*hs.pick_events[(o, s, k)]))
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
        result["var_solutions"] = merge(solution, ctx, sub_sol, sub_handles, handles)
        kpis = dict(sub_sol.get_kpis() or {})
        result["kpis"] = {k: kpis[k] for k in
                          ("makespan", "bin_fetches", "total_flow_time", "idle_time", "max_picks")
                          if k in kpis}
        result["objective"] = result["kpis"].get(objective)
    return CppSolveResult(result)


def solve_iteration(solution, handles, instance, active_orders, *, backend, horizon,
                    objective, impr_bound, cfg, progress_callback=None):
    """One complete subproblem iteration: extract, build, solve, merge.

    Returns (CppSolveResult on the full model, progress records, FrozenContext).
    """
    ctx = extract_context(solution, handles, instance, active_orders)
    if violates_bound(ctx, objective, impr_bound):
        return infeasible_result(), [], ctx
    mdl, sub_handles = build_submodel(ctx, backend, horizon, objective)
    sub_sol, progress = solve_submodel(mdl, sub_handles, backend, objective, impr_bound, cfg,
                                       progress_callback)
    result = to_full_result(solution, ctx, sub_sol, sub_handles, handles, backend, objective)
    return result, progress, ctx


def check_merged(result, handles, instance, horizon) -> list[str]:
    """Validate a merged full solution and its reported KPIs (slow; for debugging/tests)."""
    hs = HeuristicSolution.from_cp(result, handles)
    problems = list(validate_solution(hs, instance, horizon=horizon))
    kpis = result.get_kpis()
    expected = {
        "makespan": hs.makespan,
        "bin_fetches": sum(len(v) for v in hs.bin_events.values()),
        "total_flow_time": sum(a[3] for a in hs.order_assignments.values()),
    }
    for name, value in expected.items():
        if name in kpis and abs(kpis[name] - value) > 1e-6:
            problems.append(f"KPI {name}: reported {kpis[name]}, recomputed {value}")
    missing = set(handles["O"]) - set(hs.order_assignments)
    if missing:
        problems.append(f"Orders not scheduled after merge: {sorted(missing)[:10]}")
    return problems
