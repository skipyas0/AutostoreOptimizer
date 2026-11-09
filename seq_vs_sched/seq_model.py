"""CP model for joint order assignment + order/bin sequencing, minimising bin fetches.

Time is replaced by *positions*: every bin visit at a station takes exactly one
position, so intervals live on a discrete "visit index" axis.

Variables (all optional intervals):
  X[o,s]      order o's window at station s; exactly one station per order
  V[s,k,e]    e-th visit of a bin of SKU k at station s, size 1
  Q[o,s,k,e]  order o picks SKU k from visit V[s,k,e], size 1, starts with V
  C[o,k,s]    alternative over e of Q[o,s,k,e]; spans X[o,s]

Constraints:
  * one visit per position at a station        no_overlap(V[s,*,*])
  * at most |L| open orders per station          sum pulse(X[o,s]) <= |L|
  * every order line served exactly once         alternative(C[o,k,s], Q[o,s,k,*])
  * a visit exists iff some order picks from it  presence(V) == max presence(Q)

Objective: minimise the number of visits (= bin fetches), i.e. maximise how
many order lines each visit serves. Optional lexicographic tie-breaker
`secondary`:
  "none"     pure fetch minimisation
  "balance"  then minimise the busiest station's visit count
             (the pure model has no reason to spread work over stations)

Not modelled, on purpose (this is the time-free abstraction of the papers):
fetch/return/pick times, MoveCap, PickCap, bin copies N[k], cross-station timing.
"""

from __future__ import annotations

import time
from collections import defaultdict

from docplex.cp.model import CpoModel
from docplex.cp import modeler

from .sequence import SequenceSolution, Visit

SECONDARY = ("none", "balance")


def build_sequence_model(inst, secondary: str = "none", symmetry_breaking: bool = True):
    if secondary not in SECONDARY:
        raise ValueError(f"secondary must be one of {SECONDARY}")
    S, L = list(inst.S), list(inst.L)
    orders_req = inst.orders_req
    O = sorted(orders_req)

    demand = defaultdict(list)  # sku -> orders needing it
    for o in O:
        for k in orders_req[o]:
            demand[k].append(o)
    active_K = sorted(demand)
    U = {k: len(demand[k]) for k in active_K}  # never need more visits than demanding orders
    H = sum(len(orders_req[o]) for o in O)  # positions per station: one visit per order line suffices

    mdl = CpoModel(name="order_bin_sequencing")

    X = {(o, s): mdl.interval_var(optional=True, end=(0, H), name=f"X[{o},{s}]") for o in O for s in S}
    V, Q, C = {}, {}, {}
    for s in S:
        for k in active_K:
            for e in range(U[k]):
                V[(s, k, e)] = mdl.interval_var(size=1, optional=True, end=(0, H), name=f"V[{s},{k},{e}]")
                for o in demand[k]:
                    Q[(o, s, k, e)] = mdl.interval_var(size=1, optional=True, name=f"Q[{o},{s},{k},{e}]")

    # order -> exactly one station
    for o in O:
        mdl.add(mdl.sum(mdl.presence_of(X[(o, s)]) for s in S) == 1)

    for o in O:
        for s in S:
            Cs = []
            for k in orders_req[o]:
                C[(o, k, s)] = mdl.interval_var(size=1, optional=True, name=f"C[{o},{k},{s}]")
                Cs.append(C[(o, k, s)])
                mdl.add(mdl.presence_of(C[(o, k, s)]) == mdl.presence_of(X[(o, s)]))
                mdl.add(mdl.alternative(C[(o, k, s)], [Q[(o, s, k, e)] for e in range(U[k])]))
            mdl.add(mdl.span(X[(o, s)], Cs))

    for (s, k, e), v in V.items():
        picks = [Q[(o, s, k, e)] for o in demand[k]]
        for q in picks:
            mdl.add(mdl.start_at_start(v, q))
            mdl.add(mdl.presence_of(q) <= mdl.presence_of(v))
        mdl.add(mdl.presence_of(v) == mdl.max([mdl.presence_of(q) for q in picks]))

    for s in S:
        mdl.add(mdl.no_overlap([V[(s, k, e)] for k in active_K for e in range(U[k])]))
        mdl.add(mdl.sum(mdl.pulse(X[(o, s)], 1) for o in O) <= len(L))

    # redundant: each demanded SKU is visited at least once somewhere (helps the bound)
    for k in active_K:
        mdl.add(mdl.sum(mdl.presence_of(V[(s, k, e)]) for s in S for e in range(U[k])) >= 1)

    if symmetry_breaking:
        # copies of (s,k) used as a prefix and in time order
        for s in S:
            for k in active_K:
                for e in range(U[k] - 1):
                    mdl.add(mdl.presence_of(V[(s, k, e + 1)]) <= mdl.presence_of(V[(s, k, e)]))
                    mdl.add(mdl.end_before_start(V[(s, k, e)], V[(s, k, e + 1)]))
        # stations are interchangeable in this model: first order goes to the first station
        mdl.add(mdl.presence_of(X[(O[0], S[0])]) == 1)

    station_visits = {s: mdl.sum(mdl.presence_of(V[(s, k, e)]) for k in active_K for e in range(U[k])) for s in S}
    fetches = mdl.sum(station_visits.values())
    max_station = mdl.max(list(station_visits.values()))
    mdl.add_kpi(fetches, "bin_fetches")
    mdl.add_kpi(max_station, "max_station_fetches")
    if secondary == "none":
        mdl.minimize(fetches)
    else:
        mdl.add(modeler.minimize_static_lex([fetches, max_station]))

    handles = dict(X=X, V=V, Q=Q, C=C, U=U, S=S, L=L, O=O, active_K=active_K, demand=demand, H=H)
    return mdl, handles


def extract_sequence_model_solution(sol, handles, meta=None) -> SequenceSolution:
    def present(x):
        vs = sol.get_var_solution(x)
        return vs is not None and vs.is_present()

    X, V, Q, U = handles["X"], handles["V"], handles["Q"], handles["U"]
    assignment = {o: s for (o, s), x in X.items() if present(x)}
    visits = {}
    for s in handles["S"]:
        evs = []
        for k in handles["active_K"]:
            for e in range(U[k]):
                v = V[(s, k, e)]
                if present(v):
                    served = tuple(sorted(o for o in handles["demand"][k] if present(Q[(o, s, k, e)])))
                    evs.append((sol.get_var_solution(v).get_start(), k, served))
        visits[s] = [Visit(k, served) for _, k, served in sorted(evs)]
    return SequenceSolution(assignment, visits, meta=dict(meta or {}))


def solve_sequence_model(
    inst,
    secondary: str = "none",
    time_limit: float = 60,
    workers: int = 4,
    seed: int = 0,
    log_output=None,
    symmetry_breaking: bool = True,
) -> SequenceSolution | None:
    """Build + solve the sequencing model. meta holds status, bound and runtime."""
    t0 = time.perf_counter()
    mdl, h = build_sequence_model(inst, secondary=secondary, symmetry_breaking=symmetry_breaking)
    res = mdl.solve(TimeLimit=time_limit, Workers=workers, RandomSeed=seed, LogVerbosity="Quiet" if log_output is None else "Normal",
                    log_output=log_output)
    if not res or not res.is_solution():
        return None
    bounds = res.get_objective_bounds() or [None]
    meta = {
        "source": f"seq_{secondary}",
        "status": res.get_solve_status(),
        "fetch_bound": bounds[0],
        "solve_time": res.get_solve_time(),
        "wall_time": time.perf_counter() - t0,
    }
    seq = extract_sequence_model_solution(res, h, meta)
    meta["optimal"] = seq.n_fetches == meta["fetch_bound"]
    return seq
