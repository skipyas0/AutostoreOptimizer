"""SequenceSolution: an assignment + sequencing plan, without timing.

This is the solution format of the order/bin *sequencing* problem used in much
of the RCS/RS literature: each order goes to a station, and each station gets
an ordered list of bin visits, each visit serving a set of orders. No times,
no robots. The number of bin fetches is the number of visits.

Sources of a SequenceSolution:
  * the sequencing CP model           -> ``seq_model.solve_sequence_model``
  * a scheduling CP solution          -> ``extract_sequence(cp_sol, handles)``   (deliverable 3)
  * any HeuristicSolution             -> ``sequence_from_schedule(heur_sol)``

Semantics used by the sequencing model ("position semantics"): an order is open
from the position of its first visit to the position of its last visit
(inclusive), and at most |L| orders can be open at a station at any position.
A schedule can be slightly denser than that: within one bin visit, an order
can finish and another can start in the same lane. ``max_open_orders`` shows
when an extracted sequence uses this.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Visit:
    sku: int
    orders: tuple[int, ...]  # orders that pick from this visit


@dataclass
class SequenceSolution:
    assignment: dict[int, int]  # order -> station
    visits: dict[int, list[Visit]]  # station -> visits in sequence order
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ basic stats
    @property
    def n_fetches(self) -> int:
        return sum(len(v) for v in self.visits.values())

    def station_fetches(self) -> dict[int, int]:
        return {s: len(v) for s, v in self.visits.items()}

    def station_orders(self) -> dict[int, list[int]]:
        out = defaultdict(list)
        for o, s in self.assignment.items():
            out[s].append(o)
        return {s: sorted(v) for s, v in out.items()}

    def mean_orders_per_visit(self) -> float:
        n = self.n_fetches
        return sum(len(v.orders) for vs in self.visits.values() for v in vs) / n if n else 0.0

    def order_windows(self) -> dict[int, tuple[int, int, int]]:
        """order -> (station, first visit index, last visit index)."""
        win = {}
        for s, vs in self.visits.items():
            for i, v in enumerate(vs):
                for o in v.orders:
                    if o in win:
                        _, a, _ = win[o]
                        win[o] = (s, a, i)
                    else:
                        win[o] = (s, i, i)
        return win

    def order_sequence(self, s: int) -> list[int]:
        """Orders of station s, by (first visit, last visit, id)."""
        win = self.order_windows()
        return sorted((o for o, w in win.items() if w[0] == s), key=lambda o: (win[o][1], win[o][2], o))

    def max_open_orders(self) -> dict[int, int]:
        """Per station, the max number of orders open at one visit (position semantics)."""
        win = self.order_windows()
        out = {}
        for s, vs in self.visits.items():
            load = [0] * len(vs)
            for o, (ss, a, b) in win.items():
                if ss == s:
                    for i in range(a, b + 1):
                        load[i] += 1
            out[s] = max(load, default=0)
        return out

    # ------------------------------------------------------------------ checks
    def validate(self, inst, check_lanes: bool = True) -> list[str]:
        """Violations of the sequencing problem's constraints (empty list = feasible)."""
        errs = []
        orders_req = inst.orders_req
        served = defaultdict(int)
        for s, vs in self.visits.items():
            for i, v in enumerate(vs):
                if not v.orders:
                    errs.append(f"station {s} visit {i} (sku {v.sku}) serves no order")
                for o in v.orders:
                    if v.sku not in orders_req[o]:
                        errs.append(f"order {o} does not need sku {v.sku} (station {s} visit {i})")
                    if self.assignment.get(o) != s:
                        errs.append(f"order {o} served at station {s} but assigned to {self.assignment.get(o)}")
                    served[(o, v.sku)] += 1
        for o, req in orders_req.items():
            if o not in self.assignment:
                errs.append(f"order {o} unassigned")
            for k in req:
                if served[(o, k)] != 1:
                    errs.append(f"order {o} sku {k} served {served[(o, k)]} times")
        if check_lanes:
            for s, m in self.max_open_orders().items():
                if m > len(inst.L):
                    errs.append(f"station {s}: {m} orders open at once > {len(inst.L)} lanes")
        return errs

    # ------------------------------------------------------------------ (de)serialisation
    def to_dict(self) -> dict:
        return {
            "assignment": {str(o): s for o, s in self.assignment.items()},
            "visits": {str(s): [[v.sku, list(v.orders)] for v in vs] for s, vs in self.visits.items()},
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SequenceSolution":
        return cls(
            assignment={int(o): int(s) for o, s in d["assignment"].items()},
            visits={int(s): [Visit(int(k), tuple(int(o) for o in os)) for k, os in vs] for s, vs in d["visits"].items()},
            meta=d.get("meta", {}),
        )

    def pretty(self) -> str:
        lines = [f"{self.n_fetches} fetches, {self.mean_orders_per_visit():.2f} orders/visit"]
        for s in sorted(self.visits):
            seq = " ".join(f"{v.sku}{list(v.orders)}" for v in self.visits[s])
            lines.append(f"  S{s} ({len(self.visits[s])} visits, orders {self.order_sequence(s)}): {seq}")
        return "\n".join(lines)


# ---------------------------------------------------------------------- deliverable 3
def sequence_from_schedule(heur_sol, meta: dict | None = None) -> SequenceSolution:
    """Assignment + per-station visit sequence of a timed schedule (HeuristicSolution).

    Visits are ordered by the time the bin arrives at the pickface (ties: fetch start).
    """
    assignment = {o: a[0] for o, a in heur_sol.order_assignments.items()}
    visits = {}
    for s, evs in heur_sol.bin_events.items():
        evs = sorted(evs, key=lambda be: (be.presence_start, be.fetch_start, be.sku))
        visits[s] = [Visit(be.sku, tuple(sorted(be.orders_served))) for be in evs if be.orders_served]
    return SequenceSolution(assignment, visits, meta=dict(meta or {}, makespan=heur_sol.makespan))


def extract_sequence(cp_sol, handles: dict, meta: dict | None = None) -> SequenceSolution | None:
    """Assignment + sequences of a scheduling CP solution (cp_model.build_model handles)."""
    from heuristics.solution import HeuristicSolution

    hs = HeuristicSolution.from_cp(cp_sol, handles)
    return None if hs is None else sequence_from_schedule(hs, meta)
