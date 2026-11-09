"""Structural similarity between two SequenceSolutions of the same instance.

Stations are interchangeable in the sequencing model, so station labels are
matched first (the permutation that keeps the most orders at the same
station). All scores are in [0, 1], 1 = identical.

  assign_match       share of orders at the same station, after station matching
  assign_rand        Rand index of the two order partitions (label-free)
  order_prec_agree   order pairs at the same station in both solutions: share
                     with the same relation (<, =, >) of their first visit
  visit_seq_sim      1 - normalised Levenshtein distance between the SKU
                     sequences of matched stations (weighted by length)
  visit_set_jaccard  Jaccard similarity of the multisets of (station, sku) visits
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations, permutations

from .sequence import SequenceSolution


def _station_matching(a: SequenceSolution, b: SequenceSolution) -> dict[int, int]:
    """Station map a -> b that maximises the number of orders kept at the same station."""
    sa, sb = sorted(set(a.visits) | set(a.assignment.values())), sorted(set(b.visits) | set(b.assignment.values()))
    sb = sb + [None] * max(0, len(sa) - len(sb))
    best, best_map = -1, {}
    for perm in permutations(sb, len(sa)):  # few stations: brute force is fine
        m = dict(zip(sa, perm))
        score = sum(m[s] == b.assignment.get(o) for o, s in a.assignment.items())
        if score > best:
            best, best_map = score, m
    return best_map


def _levenshtein(x: list, y: list) -> int:
    prev = list(range(len(y) + 1))
    for i, xi in enumerate(x, 1):
        cur = [i]
        for j, yj in enumerate(y, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (xi != yj)))
        prev = cur
    return prev[-1]


def compare(a: SequenceSolution, b: SequenceSolution) -> dict:
    orders = sorted(a.assignment)
    m = _station_matching(a, b)
    out = {"assign_match": sum(m[a.assignment[o]] == b.assignment[o] for o in orders) / len(orders)}

    pairs = list(combinations(orders, 2))
    same = lambda sol, o1, o2: sol.assignment[o1] == sol.assignment[o2]  # noqa: E731
    out["assign_rand"] = sum(same(a, *p) == same(b, *p) for p in pairs) / len(pairs) if pairs else 1.0

    wa, wb = a.order_windows(), b.order_windows()
    sign = lambda x: (x > 0) - (x < 0)  # noqa: E731
    co = [(o1, o2) for o1, o2 in pairs if same(a, o1, o2) and same(b, o1, o2)]
    out["order_prec_pairs"] = len(co)
    out["order_prec_agree"] = (
        sum(sign(wa[o1][1] - wa[o2][1]) == sign(wb[o1][1] - wb[o2][1]) for o1, o2 in co) / len(co) if co else float("nan")
    )

    dist = tot = 0
    for s_a, s_b in m.items():
        x = [v.sku for v in a.visits.get(s_a, [])]
        y = [v.sku for v in b.visits.get(s_b, [])] if s_b is not None else []
        dist += _levenshtein(x, y)
        tot += max(len(x), len(y))
    out["visit_seq_sim"] = 1 - dist / tot if tot else 1.0

    ca = Counter((m[s], v.sku) for s, vs in a.visits.items() for v in vs)
    cb = Counter((s, v.sku) for s, vs in b.visits.items() for v in vs)
    inter, union = sum((ca & cb).values()), sum((ca | cb).values())
    out["visit_set_jaccard"] = inter / union if union else 1.0
    return out


def describe(seq: SequenceSolution, n_lanes: int) -> dict:
    """Per-solution stats used in the benchmark tables."""
    per = seq.station_fetches()
    open_max = seq.max_open_orders()
    return {
        "fetches": seq.n_fetches,
        "max_station_fetches": max(per.values(), default=0),
        "orders_per_visit": seq.mean_orders_per_visit(),
        "max_open_orders": max(open_max.values(), default=0),
        "seq_lane_feasible": all(v <= n_lanes for v in open_max.values()),
    }
