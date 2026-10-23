"""HeuristicSolution: the common output of every heuristic, plus validators.

A HeuristicSolution can be
  * checked against the problem definition  -> ``sol.validate(inst, horizon)``
  * injected as a CP warm start               -> ``sol.to_warmstart(mdl, handles)`` (docplex)
                                                 ``sol.to_cpp_warmstart(mdl, handles)`` (C++ core)
  * checked against the CP model's indexing   -> ``sol.validate_warmstart(handles)``
  * visualised                                -> ``plot_schedule(*sol.to_viz(inst))``
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from heuristics.heuristic_utils import BinEvent
from instance import Instance


@dataclass
class HeuristicSolution:
    """Complete schedule produced by a heuristic (or converted from a CP solution)."""

    order_assignments: dict[int, tuple[int, int, int, int]]  # o -> (s, ln, start, end)
    bin_events: dict[int, list[BinEvent]]  # s -> BinEvents
    makespan: int
    total_moves: int
    feasible: bool
    pick_events: dict[tuple[int, int, int], tuple[int, int]] = field(default_factory=dict)  # (o, s, k) -> (start, end)
    # Free-form info: heuristic name, params, runtime, AMA runs, per-station feasibility...
    meta: dict = field(default_factory=dict)

    def __setstate__(self, state: dict) -> None:
        # Pickles written before `meta` existed.
        self.__dict__.update({"meta": {}, **state})

    # ------------------------------------------------------------------ CP
    def check_warmstart(self, handles) -> None:
        """Raise ValueError unless this solution is a feasible warm start for the CP model in `handles`.

        Rejects solutions flagged infeasible and any violation reported by validate_warmstart
        except the diagnostic "Symmetry" notes (lanes are remapped by inject_warmstart).
        """
        if not self.feasible:
            raise ValueError(
                f"Refusing to warm-start from an infeasible heuristic solution "
                f"({self.meta.get('name', self.meta.get('heuristic', '?'))}); see sol.meta / sol.validate()."
            )
        hard = [v for v in validate_warmstart(self, self.pick_events, handles) if not v.startswith("Symmetry")]
        if hard:
            raise ValueError(f"Warm start violates {len(hard)} CP constraint(s), e.g.: {hard[:3]}")

    def to_warmstart(self, mdl, handles, strict: bool = True):
        """docplex CpoStartingPoint for a model built by cp_model.build_model.

        strict (default): raise instead of silently passing an infeasible warm start.
        """
        import cp_model

        if strict:
            self.check_warmstart(handles)
        return cp_model.inject_warmstart(self, self.pick_events, mdl, handles)

    def to_cpp_warmstart(self, mdl, handles, strict: bool = True):
        """Apply this solution as warm start to a cpp_cp_model model (returns the mock starting point)."""
        import cpp_cp_model

        if strict:
            self.check_warmstart(handles)
        return cpp_cp_model.inject_warmstart(self, self.pick_events, mdl, handles)

    def validate_warmstart(self, handles) -> list[str]:
        return validate_warmstart(self, self.pick_events, handles)

    # ------------------------------------------------------------------ checks / metrics
    def validate(self, inst: Instance, horizon: int = 10000) -> list[str]:
        return validate_solution(self, inst, horizon=horizon)

    def objective(self, alpha: float, beta: float, stations: Iterable[int]) -> float:
        """alpha * C_max + beta * sum_s (last_pick_s - first_pick_s); inf if infeasible."""
        if not self.feasible:
            return float("inf")
        c_max = float(self.makespan)
        if beta == 0.0:
            return alpha * c_max
        span_sum = 0.0
        for s in stations:
            picks = [(start, end) for (ss, ln, start, end) in self.order_assignments.values() if ss == s]
            if picks:
                span_sum += max(pe for ps, pe in picks) - min(ps for ps, pe in picks)
        return alpha * c_max + beta * span_sum

    # ------------------------------------------------------------------ CP -> heuristic
    @classmethod
    def from_cp(cls, cp_sol, handles: dict) -> "HeuristicSolution | None":
        """Convert a CP solution on the variables in `handles` (from cp_model / cpp_cp_model.build_model).

        Accepts docplex ``CpoSolveResult``, ``cp_model_utils.CppSolveResult`` (``get_var_solution``)
        and CP-SAT-style results (``BooleanValue``/``Value``). Returns None if there is no solution.
        """
        if cp_sol is None or (hasattr(cp_sol, "is_solution") and not cp_sol.is_solution()):
            return None

        if hasattr(cp_sol, "BooleanValue"):
            def present(x):
                return x is not None and cp_sol.BooleanValue(x.pres)

            def bounds(x):
                return cp_sol.Value(x.start), cp_sol.Value(x.end)
        else:
            def present(x):
                if x is None:
                    return False
                vs = cp_sol.get_var_solution(x)
                return vs is not None and vs.is_present()

            def bounds(x):
                vs = cp_sol.get_var_solution(x)
                return vs.get_start(), vs.get_end()

        I_os, I_os_lane, P, F, B, R, U = (handles[k] for k in ("I_os", "I_os_lane", "P", "F", "B", "R", "U"))
        orders_req = handles["orders_req"]
        S, L, K, O = handles["S"], handles["L"], handles["K"], handles["O"]

        order_assignments = {}
        for o in O:
            s_sel = next((s for s in S if present(I_os[(o, s)])), None)
            if s_sel is not None:
                ln_sel = next(ln for ln in L if present(I_os_lane[(o, s_sel, ln)]))
                order_assignments[o] = (s_sel, ln_sel, *bounds(I_os[(o, s_sel)]))

        bin_events: dict[int, list[BinEvent]] = {s: [] for s in S}
        pick_events = {}
        for s in S:
            for k in K:
                for e in range(U[k]):
                    if (s, k, e) not in B or not present(B[(s, k, e)]):
                        continue
                    served, picks = [], []
                    for o in O:
                        key = (o, s, k, e)
                        if k in orders_req[o] and key in P and present(P[key]):
                            served.append(o)
                            pick_events[(o, s, k)] = bounds(P[key])
                            picks.append(pick_events[(o, s, k)])
                    fs, fe = bounds(F[(s, k, e)])
                    bs, be = bounds(B[(s, k, e)])
                    rs, re = bounds(R[(s, k, e)])
                    bin_events[s].append(
                        BinEvent(sku=k, copy_id=e, fetch_start=fs, fetch_end=fe, presence_start=bs,
                                 presence_end=be, return_start=rs, return_end=re,
                                 orders_served=served, pick_intervals=sorted(picks))
                    )

        return cls(
            order_assignments=order_assignments,
            bin_events=bin_events,
            makespan=max((a[3] for a in order_assignments.values()), default=0),
            total_moves=2 * sum(len(v) for v in bin_events.values()),
            feasible=True,
            pick_events=pick_events,
            meta={"heuristic": "cp"},
        )

    # ------------------------------------------------------------------ composition
    @classmethod
    def stitch(cls, sols: Iterable["HeuristicSolution | None"]) -> "HeuristicSolution | None":
        """Union of solutions on disjoint orders (e.g. per-station subproblems)."""
        valid = [s for s in sols if s is not None]
        if not valid:
            return None
        order_assignments, pick_events = {}, {}
        bin_events: dict[int, list[BinEvent]] = defaultdict(list)
        for sol in valid:
            order_assignments.update(sol.order_assignments)
            pick_events.update(sol.pick_events)
            for s, evts in sol.bin_events.items():
                bin_events[s].extend(evts)
        return cls(
            order_assignments=order_assignments,
            bin_events=dict(bin_events),
            makespan=max((b[3] for b in order_assignments.values()), default=0),
            total_moves=sum(len(evts) * 2 for evts in bin_events.values()),
            feasible=all(sol.feasible for sol in valid),
            pick_events=pick_events,
        )

    # ------------------------------------------------------------------ visualisation
    def to_viz(self, inst: Instance) -> tuple["_MockCPSol", dict]:
        """(mock_cp_sol, handles) for schedule_visualizer.plot_schedule, no docplex needed."""
        return build_viz_handles(self, inst)


# ======================================================================
# Validators
# ======================================================================


def _sweep_exceeds(intervals: list[tuple[int, int]], cap: int) -> tuple[int, int] | None:
    """First (concurrency, t) where more than `cap` intervals overlap, else None."""
    sweep = []
    for a, b in intervals:
        sweep.append((a, +1))
        sweep.append((b, -1))
    sweep.sort(key=lambda x: (x[0], x[1]))
    concurrent = 0
    for t, delta in sweep:
        concurrent += delta
        if concurrent > cap:
            return concurrent, t
    return None


def validate_solution(sol: HeuristicSolution, inst: Instance, *, horizon: int = 10000) -> list[str]:
    """Check a solution against the problem definition. Returns violations (empty = valid).

    MoveCap and PickCap are taken from the instance.
    """
    S, K, N, rt, rt_ret = inst.S, inst.K, inst.N, inst.rt, inst.rt_ret
    orders_req = inst.orders_req
    move_cap, pick_cap = inst.movecap, inst.pickcap
    violations: list[str] = []

    # 1. Every order assigned
    for o in inst.O:
        if o not in sol.order_assignments:
            violations.append(f"Order {o} not assigned")

    # 2. Lane capacity
    lane_usage: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for o, (s, ln, start, end) in sol.order_assignments.items():
        lane_usage[(s, ln)].append((start, end, o))
    for (s, ln), intervals in lane_usage.items():
        intervals.sort()
        for i in range(len(intervals) - 1):
            _, end_i, oi = intervals[i]
            start_j, _, oj = intervals[i + 1]
            if end_i > start_j:
                violations.append(f"Lane overlap at S{s} L{ln}: order {oi} ends {end_i} > order {oj} starts {start_j}")

    # 3. Pickface: at most one bin present per station
    for s in S:
        pres = sorted((ev.presence_start, ev.presence_end, ev.sku) for ev in sol.bin_events.get(s, []))
        for i in range(len(pres) - 1):
            _, end_i, k_i = pres[i]
            start_j, _, k_j = pres[i + 1]
            if end_i > start_j:
                violations.append(
                    f"Pickface overlap at S{s}: SKU {k_i} presence ends {end_i} > SKU {k_j} starts {start_j}"
                )

    # 4. Bin copy concurrency <= N[k]
    for k in K:
        blocks = [(ev.fetch_start, ev.return_end) for s in S for ev in sol.bin_events.get(s, []) if ev.sku == k]
        if len(blocks) <= N[k]:
            continue
        hit = _sweep_exceeds(blocks, N[k])
        if hit:
            violations.append(f"SKU {k} Block concurrency {hit[0]} > N[{k}]={N[k]} at t={hit[1]}")

    # 5. MoveCap
    moves = []
    for s in S:
        for ev in sol.bin_events.get(s, []):
            moves.append((ev.fetch_start, ev.fetch_end))
            moves.append((ev.return_start, ev.return_end))
    hit = _sweep_exceeds(moves, move_cap)
    if hit:
        violations.append(f"MoveCap violation: {hit[0]} > {move_cap} at t={hit[1]}")

    # 6. Bin event timing consistency
    for s in S:
        for ev in sol.bin_events.get(s, []):
            k = ev.sku
            if ev.fetch_end - ev.fetch_start != rt[k]:
                violations.append(f"S{s} SKU {k}: fetch duration {ev.fetch_end - ev.fetch_start} != rt[{k}]={rt[k]}")
            if ev.return_end - ev.return_start != rt_ret[k]:
                violations.append(
                    f"S{s} SKU {k}: return duration {ev.return_end - ev.return_start} != rt_ret[{k}]={rt_ret[k]}"
                )
            if ev.presence_start != ev.fetch_end:
                violations.append(f"S{s} SKU {k}: presence_start {ev.presence_start} != fetch_end {ev.fetch_end}")
            if ev.presence_end != ev.return_start:
                violations.append(f"S{s} SKU {k}: presence_end {ev.presence_end} != return_start {ev.return_start}")

    # 7. Horizon
    for o, (s, ln, start, end) in sol.order_assignments.items():
        if end > horizon:
            violations.append(f"Order {o} end {end} exceeds horizon {horizon}")

    # 8. Pick completeness
    for o, (s, ln, start, end) in sol.order_assignments.items():
        for k in orders_req[o]:
            if not any(ev.sku == k and o in ev.orders_served for ev in sol.bin_events.get(s, [])):
                violations.append(f"Order {o} missing pick for SKU {k} at station {s}")

    # 9. PickCap per bin event
    if pick_cap is not None and sol.pick_events:
        for s in S:
            for ev in sol.bin_events.get(s, []):
                k = ev.sku
                ev_picks = []
                for o_srv in ev.orders_served:
                    pe_tuple = sol.pick_events.get((o_srv, s, k))
                    if pe_tuple is not None:
                        ps, pe = pe_tuple
                        if ev.presence_start <= ps and pe <= ev.presence_end:
                            ev_picks.append((ps, pe))
                if len(ev_picks) <= pick_cap:
                    continue
                hit = _sweep_exceeds(ev_picks, pick_cap)
                if hit:
                    violations.append(f"PickCap violation at S{s} SKU {k}: {hit[0]} > {pick_cap} at t={hit[1]}")

    return violations


def validate_warmstart(sol: HeuristicSolution, pick_events: dict, handles: dict) -> list[str]:
    """Check consistency with the CP model's indexing / constraints (handles from build_model)."""
    violations = []
    U, S, L = handles["U"], handles["S"], handles["L"]

    # 1. Copy count check
    for s in S:
        counts = defaultdict(int)
        for be in sol.bin_events.get(s, []):
            counts[be.sku] += 1
        for k, count in counts.items():
            if count > U.get(k, 0):
                violations.append(
                    f"Station {s} SKU {k}: Heuristic used {count} trips, CP model limit U[{k}]={U.get(k, 0)}"
                )

    # 2. Pick within presence
    for (o, s_sel, k), (ps, pe) in pick_events.items():
        cands = [be for be in sol.bin_events.get(s_sel, []) if be.sku == k]
        if not any(be.presence_start <= ps and pe <= be.presence_end for be in cands):
            violations.append(
                f"Pick {o}@{s_sel} SKU {k} [{ps},{pe}] not covered by any BinEvent presence interval at station."
            )

    # 3. Order window == span of its picks (CP: span(I_os, [C]))
    for o in sol.order_assignments:
        s_sel, ln_sel, t_start, t_end = sol.order_assignments[o]
        picks = [(ps, pe) for (oo, ss, kk), (ps, pe) in pick_events.items() if oo == o]
        if not picks:
            continue
        min_p = min(ps for ps, pe in picks)
        max_p = max(pe for ps, pe in picks)
        if t_start != min_p:
            violations.append(f"Order {o}: Heuristic Start {t_start} != Min Pick Start {min_p}")
        if t_end != max_p:
            violations.append(f"Order {o}: Heuristic End {t_end} != Max Pick End {max_p}")

    # 4. Lane fill-order symmetry breaking (diagnostic; inject_warmstart remaps lanes)
    for s in S:
        lane_counts = defaultdict(int)
        for o, (s_a, ln_a, _, _) in sol.order_assignments.items():
            if s_a == s:
                lane_counts[ln_a] += 1
        for i in range(len(L) - 1):
            ci, cn = lane_counts.get(L[i], 0), lane_counts.get(L[i + 1], 0)
            if ci < cn:
                violations.append(
                    f"Symmetry (A) raw lanes at station {s}: "
                    f"lane {L[i]} has {ci} orders < lane {L[i + 1]} has {cn} "
                    f"(inject_warmstart will remap)"
                )

    # 5. Bin copy prefix ordering
    for s in S:
        by_sku = defaultdict(list)
        for be in sol.bin_events.get(s, []):
            by_sku[be.sku].append(be)
        for k, k_evts in by_sku.items():
            k_sorted = sorted(k_evts, key=lambda x: x.fetch_start)
            for i in range(len(k_sorted) - 1):
                if k_sorted[i].presence_end > k_sorted[i + 1].presence_start:
                    violations.append(
                        f"Symmetry (B) at station {s} SKU {k}: "
                        f"B[e={i}].presence_end={k_sorted[i].presence_end} > "
                        f"B[e={i + 1}].presence_start={k_sorted[i + 1].presence_start}"
                    )

    # 6. Order completeness
    for o in handles["O"]:
        if o not in sol.order_assignments:
            violations.append(f"Order {o} not assigned in solution")

    # 7. Lane no-overlap
    lane_usage: dict = defaultdict(list)
    for o, (s_a, ln_a, t_s, t_e) in sol.order_assignments.items():
        lane_usage[(s_a, ln_a)].append((t_s, t_e, o))
    for (s, ln), intervals in lane_usage.items():
        intervals.sort()
        for i in range(len(intervals) - 1):
            _, end_i, o_i = intervals[i]
            start_j, _, o_j = intervals[i + 1]
            if end_i > start_j:
                violations.append(
                    f"Lane overlap at S{s} L{ln}: order {o_i} ends {end_i} > order {o_j} starts {start_j}"
                )

    # 8. Pickface no-overlap
    for s in S:
        pres = sorted((be.presence_start, be.presence_end, be.sku) for be in sol.bin_events.get(s, []))
        for i in range(len(pres) - 1):
            _, end_i, k_i = pres[i]
            start_j, _, k_j = pres[i + 1]
            if end_i > start_j:
                violations.append(f"Pickface overlap at S{s}: SKU {k_i} ends {end_i} > SKU {k_j} starts {start_j}")

    # 9. Bin timing consistency
    rt, rt_ret = handles["rt"], handles["rt_return"]
    for s in S:
        for be in sol.bin_events.get(s, []):
            k = be.sku
            if k not in rt:
                continue
            if be.fetch_end - be.fetch_start != rt[k]:
                violations.append(f"S{s} SKU {k}: fetch duration {be.fetch_end - be.fetch_start} != rt[{k}]={rt[k]}")
            if be.return_end - be.return_start != rt_ret[k]:
                violations.append(
                    f"S{s} SKU {k}: return duration {be.return_end - be.return_start} != rt_return[{k}]={rt_ret[k]}"
                )
            if be.presence_start != be.fetch_end:
                violations.append(f"S{s} SKU {k}: presence_start {be.presence_start} != fetch_end {be.fetch_end}")
            if be.presence_end != be.return_start:
                violations.append(f"S{s} SKU {k}: presence_end {be.presence_end} != return_start {be.return_start}")

    # 10. Block concurrency <= N[k]
    N_map = handles.get("N")
    if N_map is not None:
        for k in handles["K"]:
            blocks = [(be.fetch_start, be.return_end) for s in S for be in sol.bin_events.get(s, []) if be.sku == k]
            if len(blocks) <= N_map.get(k, 1):
                continue
            hit = _sweep_exceeds(blocks, N_map[k])
            if hit:
                violations.append(f"Block concurrency SKU {k}: {hit[0]} > N[{k}]={N_map[k]} at t={hit[1]}")

    # 11. MoveCap
    mc = handles.get("move_cap")
    if mc is not None:
        moves = []
        for s in S:
            for be in sol.bin_events.get(s, []):
                moves.append((be.fetch_start, be.fetch_end))
                moves.append((be.return_start, be.return_end))
        hit = _sweep_exceeds(moves, mc)
        if hit:
            violations.append(f"MoveCap violation: {hit[0]} > {mc} at t={hit[1]}")

    # 12. PickCap: simultaneous picks per station (CP: always_in(sum of pick pulses, 0, pick_cap))
    pc = handles.get("pick_cap")
    if pc is not None:
        for s in S:
            hit = _sweep_exceeds([iv for (o, ss, k), iv in pick_events.items() if ss == s], pc)
            if hit:
                violations.append(f"PickCap violation at S{s}: {hit[0]} > {pc} at t={hit[1]}")

    # 13. Horizon
    hz = handles.get("horizon")
    if hz is not None:
        late = [be for s in S for be in sol.bin_events.get(s, []) if be.return_end > hz]
        if late:
            violations.append(f"Horizon violation: {len(late)} bin visit(s) return after {hz}")

    return violations


# ======================================================================
# Visualisation bridge (schedule_visualizer.plot_schedule without docplex)
# ======================================================================


class _MockIV:
    """Placeholder for a CP interval variable — only used as a unique dict key."""

    __slots__ = ()


@dataclass
class _MockIVSol:
    present: bool
    start: int = 0
    end: int = 0

    def is_present(self) -> bool:
        return self.present

    def get_start(self) -> int:
        return self.start

    def get_end(self) -> int:
        return self.end


class _MockCPSol:
    """Stand-in for a docplex CpoSolveResult, backed by plain dicts."""

    def __init__(self) -> None:
        self._d: dict[int, _MockIVSol] = {}

    def _set(self, iv: _MockIV, **kw) -> None:
        self._d[id(iv)] = _MockIVSol(**kw)

    def get_var_solution(self, iv: _MockIV) -> _MockIVSol:
        return self._d.get(id(iv), _MockIVSol(present=False))


def build_viz_handles(sol: HeuristicSolution, inst: Instance) -> tuple[_MockCPSol, dict]:
    """Convert a solution to (mock_cp_sol, handles) for ``plot_schedule(mock_sol, handles)``."""
    S, L, K, orders_req, rt, p = inst.S, inst.L, inst.K, inst.orders_req, inst.rt, inst.p
    O, rt_ret = inst.O, inst.rt_ret
    msol = _MockCPSol()
    I_os, I_os_lane, F, B, R, P, C = {}, {}, {}, {}, {}, {}, {}

    # Per-(s, k) visit index: copy_id is reused across visits, so it can't be the key.
    visit_counters: dict[tuple[int, int], int] = defaultdict(int)
    visit_idx_lookup: dict[tuple[int, int, int], int] = {}  # (s, k, presence_start) -> visit idx

    for o in O:
        assignment = sol.order_assignments.get(o)
        s_sel, ln_sel, t_start, t_end = assignment if assignment else (None, None, 0, 0)
        for s in S:
            iv = _MockIV()
            I_os[(o, s)] = iv
            msol._set(iv, present=(s == s_sel), start=t_start, end=t_end)
            for ln in L:
                iv2 = _MockIV()
                I_os_lane[(o, s, ln)] = iv2
                msol._set(iv2, present=(s == s_sel and ln == ln_sel), start=t_start, end=t_end)

    for s, evts in sol.bin_events.items():
        for be in evts:
            k = be.sku
            e = visit_counters[(s, k)]
            visit_counters[(s, k)] += 1
            visit_idx_lookup[(s, k, be.presence_start)] = e
            for store, a, b in ((F, be.fetch_start, be.fetch_end), (B, be.presence_start, be.presence_end),
                                (R, be.return_start, be.return_end)):
                iv = _MockIV()
                store[(s, k, e)] = iv
                msol._set(iv, present=True, start=a, end=b)

    U: dict[int, int] = {k: 0 for k in K}
    for (s, k), count in visit_counters.items():
        U[k] = max(U[k], count)

    for (o, s, k), (ps, pe) in sol.pick_events.items():
        matched = next(
            (be for be in sol.bin_events.get(s, []) if be.sku == k and be.presence_start <= ps < be.presence_end),
            None,
        )
        if matched is not None:
            e = visit_idx_lookup.get((s, k, matched.presence_start))
            if e is not None:
                iv_p = _MockIV()
                P[(o, s, k, e)] = iv_p
                msol._set(iv_p, present=True, start=ps, end=pe)
        iv_c = _MockIV()
        C[(o, k, s)] = iv_c
        msol._set(iv_c, present=True, start=ps, end=pe)

    handles = {
        "I_os": I_os, "I_os_lane": I_os_lane, "C": C, "P": P, "F": F, "R": R, "B": B, "U": U,
        "orders_req": orders_req, "rt": rt, "rt_return": rt_ret, "p": p,
        "S": S, "L": L, "K": K, "O": O,
    }
    return msol, handles
