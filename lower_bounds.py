"""Lower bounds on robot work W and makespan C (prototype).

    python lower_bounds.py [GEN... instance names]      # table over precalculated_instances/

Robot work of a schedule: W = sum over bin visits of rt[k] + rt_ret[k].

Bounds on W
-----------
``global``  one visit per required SKU.

``lane``    one visit per SKU is not enough for a SKU shared by many orders, because each
            order holds a lane from its first to its last pick.

    Claim. Take a visit v of SKU k at station s with presence [a, b). Let o be an order
    served by v, and let k not be o's first SKU. Then o is in a lane of s at instant a.
    Proof: o had an earlier pick from another bin. That pick ended at or before a, since
    the pickface holds one bin at a time. o's lane runs from its first pick start (< a) to
    its last pick end (> a).
    So v serves at most L orders for which k is not the first SKU. By symmetry, at
    instant b it serves at most L orders for which k is not the last SKU.

    Summed over the visits of k (any stations): for any feasible schedule, with f_o / l_o
    the first / last SKU of order o,
        L * v_k >= #{o requiring k : f_o != k}   and   L * v_k >= #{o requiring k : l_o != k}.
    A single-line order has f_o = l_o = k and counts in neither. A multi-line order has
    f_o != l_o.
    W >= min over choices of (f_o, l_o) and integers v_k >= 1 of sum_k w_k v_k. This is a
    small ILP, solved with CP-SAT. Only SKUs with more than L multi-line orders can need
    v_k > 1, so only orders touching such SKUs enter the model. If the time limit stops
    the search, the solver's proven objective bound is used, so the result stays valid.
    It implies v_k >= ceil(m_k / 2L) (m_k = multi-line orders on k). It is strictly
    stronger when an order has cheap SKUs to use as its first / last pick: its hot SKUs
    then count on both sides.

    Not used here: station assignment, timing, copies N[k], MoveCap. So the bound holds
    for any horizon and any number of stations.

Bounds on C (makespan = last pick end; returns are not part of C)
-----------------------------------------------------------------
``picker``  some station does at least a 1/S share of the pick time, serially per bin and
            one bin at a time (PickCap orders in parallel). Its first pick is at or after
            the shortest fetch: C >= min rt + ceil(P / (S * PickCap)).
``robot``   every useful fetch ends before C. Returns start at presence end, so the
            return work after C comes from returns under way at C (<= MoveCap) or from bins
            still at a pickface at C (<= S), each from a distinct useful visit:
            MoveCap * C >= W_lb - return_spill (sum of the MoveCap + S largest return times).

Sequence relaxations (exact CP-SAT models, small instances only)
----------------------------------------------------------------
``seq_relaxation_lb``      min W over one sequence of bin presentations with <= L orders open
                           across any gap. Exact for an unconstrained horizon, so it is the
                           true minimum robot work; the lane bound is a relaxation of it.
``coupled_relaxation_lb``  min C over one sequence per station with station pick load
                           <= PickCap * (C - min rt) and MoveCap * C >= W - spill. Captures
                           the trade-off "finish by C => split sharing groups over stations".
Findings (Nov 2026): CP-SAT proves the W relaxation quickly on tiny instances, and there the
lane bound is 0-5.5% below the true minimum W (13 proven instances, L = 1-2). Beyond
~10 orders the W relaxation is not proven within 120 s, but its incumbent (a real one-station
schedule) beats every CP Optimizer min-W schedule, so min W is within 0-14% of the lane bound.
It cannot prove the coupled one beyond the cuts even at 16 orders. Its relaxed solutions show
it would close at most ~25-45% of the C gap; the rest is timing (fetch lead times, MoveCap over
time). Calibration: scripts/lb_calibration, results/lb_calibration/summary.md.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass

from instance import Instance


@dataclass
class WorkBound:
    value: int  # valid lower bound on W
    global_lb: int
    optimal: bool  # ILP solved to optimality (otherwise value = proven bound)
    hot_skus: int  # SKUs with more than L multi-line orders
    model_orders: int
    extra_visits: int  # sum over SKUs of (v_k - 1) at the ILP solution


def _lanes(inst: Instance) -> int:
    return len(inst.L) if hasattr(inst.L, "__len__") else int(inst.L)


def visit_work(inst: Instance) -> dict[int, int]:
    return {k: inst.rt[k] + inst.rt_ret[k] for o in inst.O for k in inst.orders_req[o]}


def work_lb_global(inst: Instance) -> int:
    return sum(visit_work(inst).values())


def work_lb_lane(inst: Instance, time_limit: float = 30.0, workers: int = 8) -> WorkBound:
    from ortools.sat.python import cp_model

    L = _lanes(inst)
    w = visit_work(inst)
    req = {o: sorted(set(inst.orders_req[o])) for o in inst.O}
    multi = defaultdict(list)  # k -> multi-line orders requiring k
    for o, ks in req.items():
        if len(ks) > 1:
            for k in ks:
                multi[k].append(o)
    hot = sorted(k for k, os_ in multi.items() if len(os_) > L)
    glb = sum(w.values())
    if not hot:
        return WorkBound(glb, glb, True, 0, 0, 0)

    orders = sorted({o for k in hot for o in multi[k]})
    m = cp_model.CpModel()
    f, l = {}, {}
    for o in orders:
        for k in req[o]:
            f[o, k] = m.new_bool_var(f"f{o}_{k}")
            l[o, k] = m.new_bool_var(f"l{o}_{k}")
            m.add(f[o, k] + l[o, k] <= 1)
        m.add_exactly_one(f[o, k] for k in req[o])
        m.add_exactly_one(l[o, k] for k in req[o])
    v = {}
    for k in hot:
        mk = len(multi[k])
        v[k] = m.new_int_var(1, math.ceil(mk / L), f"v{k}")
        m.add(L * v[k] >= mk - sum(f[o, k] for o in multi[k]))
        m.add(L * v[k] >= mk - sum(l[o, k] for o in multi[k]))
    m.minimize(sum(w[k] * (v[k] - 1) for k in hot))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    status = solver.solve(m)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"lane bound ILP: no solution ({solver.status_name(status)})")
    optimal = status == cp_model.OPTIMAL
    extra = math.ceil(solver.best_objective_bound - 1e-6)  # proven bound (= optimum if OPTIMAL)
    return WorkBound(
        value=glb + extra, global_lb=glb, optimal=optimal, hot_skus=len(hot), model_orders=len(orders),
        extra_visits=sum(solver.value(v[k]) - 1 for k in hot),
    )


@dataclass
class SeqBound:
    value: int  # valid lower bound (proven bound of the relaxation): W, or C for the coupled one
    upper: int | None  # best relaxation solution found
    optimal: bool
    runtime: float
    work_at_upper: int | None = None  # coupled: W of the best relaxed solution


def return_spill(inst: Instance) -> int:
    """Upper bound on the return work that can fall after the makespan C.

    Returns start at presence end. After C the only return work left is from returns under
    way at C (<= MoveCap) and from bins still at a pickface at C (<= S). Each comes from a
    distinct visit, and SKU k has at most m_k visits (one per order requiring it), so the spill
    is at most the sum of the MoveCap + S largest return times in that multiset.
    """
    cnt = defaultdict(int)
    for o in inst.O:
        for k in set(inst.orders_req[o]):
            cnt[k] += 1
    rets = sorted((inst.rt_ret[k] for k, c in cnt.items() for _ in range(c)), reverse=True)
    return sum(rets[: inst.movecap + len(inst.S)])


def _seq_model(inst: Instance, coupled: bool, hint=None, cuts: bool = True):
    """CP-SAT model of the sequence relaxation (see seq_relaxation_lb / coupled_relaxation_lb)."""
    from ortools.sat.python import cp_model

    L = _lanes(inst)
    w = visit_work(inst)
    req = {o: sorted(set(inst.orders_req[o])) for o in inst.O}
    T = sum(len(ks) for ks in req.values())
    Ks = sorted(w)
    B = list(range(len(inst.S))) if coupled else [0]  # blocks = stations
    pos = [(b, t) for b in B for t in range(T)]
    m = cp_model.CpModel()
    y = {(b, t, k): m.new_bool_var("") for (b, t) in pos for k in Ks}
    for b, t in pos:
        m.add_at_most_one(y[b, t, k] for k in Ks)
        if t + 1 < T:  # used positions first within a block
            m.add(sum(y[b, t, k] for k in Ks) >= sum(y[b, t + 1, k] for k in Ks))
    orders = sorted(req)
    a = {}
    if coupled:  # order -> station; identical stations: order i goes to a station <= i
        for i, o in enumerate(orders):
            for b in B:
                a[o, b] = m.new_bool_var("") if b <= i else m.new_constant(0)
            m.add_exactly_one(a[o, b] for b in B)
    x = {}
    for o, ks in req.items():
        for k in ks:
            for b, t in pos:
                x[o, k, b, t] = m.new_bool_var("")
                m.add_implication(x[o, k, b, t], y[b, t, k])
                if coupled:
                    m.add_implication(x[o, k, b, t], a[o, b])
            m.add_exactly_one(x[o, k, b, t] for b, t in pos)
    for b in B:
        for t in range(T - 1):
            ops = []
            for o, ks in req.items():
                if len(ks) == 1:
                    continue
                # open across gap (t, t+1) of block b iff picked at <= t and picks at >= t+1 there
                st = m.new_bool_var("")
                nd = m.new_bool_var("")
                m.add_bool_or([st.Not(), nd.Not(), op := m.new_bool_var("")])
                ops.append(op)
                for k in ks:
                    for t2 in range(t + 1):
                        m.add_implication(x[o, k, b, t2], st)
                    for t2 in range(t + 1, T):
                        m.add_implication(x[o, k, b, t2], nd)
            m.add(sum(ops) <= L)
    work = sum(w[k] * y[b, t, k] for (b, t) in pos for k in Ks)
    if coupled:
        used = list(w)
        rtmin = min(inst.rt[k] for k in used)
        P = {o: sum(inst.p[k] for k in req[o]) for o in orders}
        ub = H_DEFAULT
        C = m.new_int_var(0, ub, "C")
        for b in B:
            m.add(sum(P[o] * a[o, b] for o in orders) <= inst.pickcap * (C - rtmin))
        m.add(inst.movecap * C >= work - return_spill(inst))
        if cuts:  # valid bounds implied by the relaxation, given to the solver as cuts
            wl = work_lb_lane(inst, time_limit=30, workers=8).value
            m.add(work >= wl)
            m.add(C >= makespan_lb(inst, work_lb=wl).value)
        m.minimize(C)
    else:
        C = None
        if cuts:
            m.add(work >= work_lb_lane(inst, time_limit=30, workers=8).value)
        m.minimize(work)

    if hint is not None:
        seqs = {}
        for i, s in enumerate(sorted(hint.bin_events)):
            seqs[i if coupled else 0] = seqs.get(i if coupled else 0, []) + [
                (ev.sku, set(ev.orders_served)) for ev in sorted(hint.bin_events[s], key=lambda e: e.presence_start)]
        if all(len(v) <= T for v in seqs.values()):
            for (b, t, k), var in y.items():
                sq = seqs.get(b, [])
                m.add_hint(var, t < len(sq) and sq[t][0] == k)
            for (o, k, b, t), var in x.items():
                sq = seqs.get(b, [])
                m.add_hint(var, t < len(sq) and sq[t][0] == k and o in sq[t][1])
    return m, work, C


H_DEFAULT = 10000


def _solve(m, time_limit, workers):
    import time

    from ortools.sat.python import cp_model

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    t0 = time.perf_counter()
    status = solver.solve(m)
    ok = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    return solver, ok, status == cp_model.OPTIMAL, time.perf_counter() - t0


def seq_relaxation_lb(inst: Instance, time_limit: float = 60.0, workers: int = 8, hint=None) -> SeqBound:
    """Time-free sequence relaxation of min W, solved exactly (small instances only).

    One pickface presents SKUs at positions 1..T. Every order line is picked at one position
    presenting its SKU. An order is open from its first to its last pick position, and at
    most L orders are open across any gap between consecutive positions.
    Valid: concatenate the stations' visit sequences of a feasible schedule. At the instant
    between two consecutive presences of a station, every order that has picked before and
    picks after is in a lane (<= L), and no order crosses a station boundary.
    Exact for an unconstrained horizon: one station can run any relaxed sequence bin after bin
    (orders closing at a position pick before the ones opening there), so the optimum is the
    true minimum robot work. It implies the lane bound.
    T = number of order lines: an optimal relaxed sequence has no empty positions.
    `hint`: a HeuristicSolution whose concatenated visit sequence is used as a solver hint.
    """
    m, work, _ = _seq_model(inst, coupled=False, hint=hint)
    solver, ok, opt, rt = _solve(m, time_limit, workers)
    glb = work_lb_global(inst)
    val = max(glb, math.ceil(solver.best_objective_bound - 1e-6))
    return SeqBound(val, int(solver.objective_value) if ok else None, opt, rt)


def coupled_relaxation_lb(inst: Instance, time_limit: float = 60.0, workers: int = 8, hint=None) -> SeqBound:
    """Makespan bound from the sequence relaxation with one sequence per station (small instances).

    Minimise C subject to:
      * each order is assigned to one station, and each station has its own sequence (as in
        seq_relaxation_lb, with lanes <= L across every gap);
      * station pick load <= PickCap * (C - min rt): picks are serial per bin and one bin is
        at the pickface at a time, and none starts before the first fetch ends;
      * MoveCap * C >= W - return_spill: all useful fetches end before C.
    Valid: a feasible schedule with makespan C gives such a relaxed solution.
    This captures the trade-off that the separate bounds miss. Finishing by C caps each
    station's load, so order groups that share SKUs are split over stations, and each split
    costs extra visits.
    """
    m, work, C = _seq_model(inst, coupled=True, hint=hint)
    solver, ok, opt, rt = _solve(m, time_limit, workers)
    val = math.ceil(solver.best_objective_bound - 1e-6)
    return SeqBound(val, int(solver.objective_value) if ok else None, opt, rt,
                    work_at_upper=int(solver.value(work)) if ok else None)


@dataclass
class MakespanBound:
    value: int
    picker: int
    robot: int
    robot_global: int  # robot bound with the global W bound, for comparison


def makespan_lb(inst: Instance, work_lb: int | None = None, **kw) -> MakespanBound:
    if work_lb is None:
        work_lb = work_lb_lane(inst, **kw).value
    used = set(visit_work(inst))
    S = len(inst.S)
    P = sum(inst.p[k] for o in inst.O for k in inst.orders_req[o])
    picker = min(inst.rt[k] for k in used) + math.ceil(P / (S * inst.pickcap))
    spill = return_spill(inst)

    def robot(wlb: int) -> int:
        return max(0, math.ceil((wlb - spill) / inst.movecap))

    r, rg = robot(work_lb), robot(work_lb_global(inst))
    return MakespanBound(value=max(picker, r), picker=picker, robot=r, robot_global=rg)


def main() -> None:
    import argparse
    import os

    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("instances", nargs="*")
    ap.add_argument("--instances-dir", default="precalculated_instances")
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--warm-starts", default="results/warmstart_benchmark_copied/warm_starts.csv",
                    help="CSV with instance, robot_work, makespan columns (best start W)")
    ap.add_argument("--lns-runs", default="results/warmstart_benchmark_copied/analysis/runs.csv",
                    help="CSV with instance, makespan, robot_work columns (best LNS C)")
    a = ap.parse_args()
    names = a.instances or sorted(d for d in os.listdir(a.instances_dir)
                                  if os.path.exists(os.path.join(a.instances_dir, d, "instance.pkl")))
    ws = pd.read_csv(a.warm_starts) if os.path.exists(a.warm_starts) else None
    runs = pd.read_csv(a.lns_runs) if os.path.exists(a.lns_runs) else None
    rows = []
    for n in names:
        inst = Instance.from_pickle(os.path.join(a.instances_dir, n, "instance.pkl"))
        wb = work_lb_lane(inst, time_limit=a.time_limit)
        cb = makespan_lb(inst, work_lb=wb.value)
        row = dict(instance=n, **{f"W_{k}": v for k, v in asdict(wb).items()}, **{f"C_{k}": v for k, v in asdict(cb).items()})
        if ws is not None and n in set(ws.instance):
            row["W_best_start"] = int(ws[ws.instance == n].robot_work.min())
        if runs is not None and n in set(runs.instance):
            r = runs[runs.instance == n]
            row["C_best_lns"] = int(r.makespan.min())
            row["W_best_lns"] = int(r.robot_work.min())
        rows.append(row)
    df = pd.DataFrame(rows).set_index("instance")
    df["W_lane_vs_global_%"] = 100 * (df.W_value / df.W_global_lb - 1)
    if "W_best_start" in df:
        df["W_gap_start_%"] = 100 * (df.W_best_start / df.W_value - 1)
    if "W_best_lns" in df:
        df["W_gap_lns_%"] = 100 * (df.W_best_lns / df.W_value - 1)
    if "C_best_lns" in df:
        df["C_gap_%"] = 100 * (df.C_best_lns / df.C_value - 1)
        df["C_gap_globalW_%"] = 100 * (df.C_best_lns / df[["C_picker", "C_robot_global"]].max(axis=1) - 1)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(df.round(1).to_string())


if __name__ == "__main__":
    main()
