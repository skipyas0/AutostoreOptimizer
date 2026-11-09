"""Restrict the scheduling CP model (cp_model.build_model) to a given SequenceSolution.

`constrain_schedule_to_sequence(mdl, handles, seq, level)` only *adds*
constraints to an already built model, so cp_model.py stays untouched.

Levels (each includes the previous one):
  "assignment"  order -> station is fixed; lanes, visits and timing are free
  "visits"      the set of bin visits per station and which visit serves each
                order line are fixed (P / F / B / R presences); their order is free
  "sequence"    the bin visits at each station also follow the given order
                (end(B_j) <= start(B_{j+1})); only lanes and timing are free

Visit j of SKU k at station s is mapped to copy index e = (number of earlier
visits of k at s), which matches the model's symmetry breaking (copies of
(s,k) are used as a prefix and in time order).

A sequence that is feasible for the sequencing model is always realisable by
the scheduling model at every level: run the visits in the given order and
picks one after another, and the lane intervals keep the same overlap
structure. Only the horizon can get in the way.
"""

from __future__ import annotations

from collections import defaultdict

from .sequence import SequenceSolution

LEVELS = ("assignment", "visits", "sequence")


def visit_copy_index(seq: SequenceSolution) -> dict[int, list[tuple[int, int]]]:
    """station -> [(sku, copy index e)] in sequence order."""
    out = {}
    for s, vs in seq.visits.items():
        seen = defaultdict(int)
        lst = []
        for v in vs:
            lst.append((v.sku, seen[v.sku]))
            seen[v.sku] += 1
        out[s] = lst
    return out


def constrain_schedule_to_sequence(mdl, handles: dict, seq: SequenceSolution, level: str = "sequence") -> int:
    """Add constraints so that every schedule follows `seq` at `level`. Returns #constraints added."""
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    I_os, P, F, B = handles["I_os"], handles["P"], handles["F"], handles["B"]
    S, O, U, orders_req = handles["S"], handles["O"], handles["U"], handles["orders_req"]
    n = 0

    def fix(iv, present: bool):
        nonlocal n
        mdl.add(mdl.presence_of(iv) == int(present))
        n += 1

    # assignment
    for o in O:
        for s in S:
            fix(I_os[(o, s)], seq.assignment[o] == s)
    if level == "assignment":
        return n

    # visits and which visit serves each order line
    copies = visit_copy_index(seq)
    used_B = set()
    used_P = set()
    for s, vs in seq.visits.items():
        for v, (k, e) in zip(vs, copies[s]):
            if e >= U[k]:
                raise ValueError(f"sequence uses {e + 1} visits of sku {k} at station {s}, model allows {U[k]}")
            used_B.add((s, k, e))
            for o in v.orders:
                used_P.add((o, s, k, e))
    for key, b in B.items():
        fix(b, key in used_B)
        fix(F[key], key in used_B)  # redundant with model coupling, but cheap
    for key, p in P.items():
        fix(p, key in used_P)
    if level == "visits":
        return n

    # visit order per station
    for s, lst in copies.items():
        for (k1, e1), (k2, e2) in zip(lst, lst[1:]):
            mdl.add(mdl.end_before_start(B[(s, k1, e1)], B[(s, k2, e2)]))
            n += 1
    return n


# ---------------------------------------------------------------------- feasible timing of a sequence
def _max_overlap(intervals, lo: int, hi: int) -> int:
    """Max number of `intervals` [s, e) overlapping any point of [lo, hi)."""
    ev = []
    for s, e in intervals:
        s, e = max(s, lo), min(e, hi)
        if s < e:
            ev += [(s, 1), (e, -1)]
    best = cur = 0
    for _, d in sorted(ev, key=lambda x: (x[0], x[1])):
        cur += d
        best = max(best, cur)
    return best


def greedy_schedule_from_sequence(inst, seq: SequenceSolution):
    """A feasible timed schedule (HeuristicSolution) that follows `seq` exactly.

    Visits keep their order per station; picks of one visit run back to back.
    The next visit to place is the one (over all stations) that can start its
    fetch earliest, subject to MoveCap, bin copies N[k] and the station port.
    Not optimised: it is a guaranteed warm start for the restricted CP solves.
    """
    from heuristics.heuristic_utils import BinEvent
    from heuristics.solution import HeuristicSolution

    rt, rr, p, N, cap = inst.rt, inst.rt_ret, inst.p, inst.N, inst.movecap
    moves: list[tuple[int, int]] = []
    blocks = defaultdict(list)  # sku -> [(fetch start, return end)]
    port_free = {s: 0 for s in seq.visits}
    nxt = {s: 0 for s in seq.visits}
    bin_events = {s: [] for s in inst.S}
    pick_events = {}

    def ok(t, k, d):
        f, b, r = (t, t + rt[k]), (t + rt[k], t + rt[k] + d), (t + rt[k] + d, t + rt[k] + d + rr[k])
        return (_max_overlap(moves, *f) < cap and _max_overlap(moves, *r) < cap
                and _max_overlap(blocks[k], f[0], r[1]) < N[k])

    def earliest(s):
        v = seq.visits[s][nxt[s]]
        k, d = v.sku, p[v.sku] * len(v.orders)
        lb = max(0, port_free[s] - rt[k])
        ends = {e for _, e in moves} | {e for _, e in blocks[k]}
        cands = sorted({lb} | {e - off for e in ends for off in (0, rt[k], rt[k] + d) if e - off >= lb})
        return next(t for t in cands if ok(t, k, d))  # the last end always works

    while any(nxt[s] < len(seq.visits[s]) for s in seq.visits):
        t, s = min((earliest(s), s) for s in seq.visits if nxt[s] < len(seq.visits[s]))
        v = seq.visits[s][nxt[s]]
        k = v.sku
        fs, ps = t, t + rt[k]
        picks = []
        for i, o in enumerate(v.orders):
            pick_events[(o, s, k)] = (ps + i * p[k], ps + (i + 1) * p[k])
            picks.append(pick_events[(o, s, k)])
        pe = ps + p[k] * len(v.orders)
        re = pe + rr[k]
        moves += [(fs, ps), (pe, re)]
        blocks[k].append((fs, re))
        port_free[s] = pe
        bin_events[s].append(BinEvent(sku=k, copy_id=0, fetch_start=fs, fetch_end=ps, presence_start=ps,
                                      presence_end=pe, return_start=pe, return_end=re,
                                      orders_served=list(v.orders), pick_intervals=picks))
        nxt[s] += 1

    # order windows, then lanes by greedy interval colouring per station
    win = {}
    for (o, s, k), (a, b) in pick_events.items():
        w = win.get(o)
        win[o] = (s, a, b) if w is None else (s, min(w[1], a), max(w[2], b))
    order_assignments = {}
    for s in inst.S:
        lane_free = {ln: 0 for ln in inst.L}
        for o in sorted((o for o, w in win.items() if w[0] == s), key=lambda o: win[o][1]):
            _, a, b = win[o]
            ln = next((ln for ln in inst.L if lane_free[ln] <= a), None)
            if ln is None:
                raise ValueError(f"sequence needs more than {len(inst.L)} lanes at station {s}")
            lane_free[ln] = b
            order_assignments[o] = (s, ln, a, b)
    return HeuristicSolution(
        order_assignments=order_assignments, bin_events=bin_events,
        makespan=max(w[2] for w in win.values()), total_moves=len(moves), feasible=True,
        pick_events=pick_events, meta={"heuristic": "greedy_from_sequence"},
    )
