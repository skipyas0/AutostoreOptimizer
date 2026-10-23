"""Shared case definitions + solution fingerprinting for the equivalence check.

This module must stay importable by BOTH the pre-refactor code (old adapter)
and the refactored code (new adapter), so it only imports `datagen` and
`instance` (both of which exist on both sides) and never a heuristic module.
"""

from __future__ import annotations

import os
from collections import defaultdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

TIGHT_HORIZON = 400  # small enough to exercise the "infeasible within horizon" paths

# Real instances used in LNS experiments (small/medium ones only).
REAL_INSTANCES = [
    "GEN1-1-1-100-10-2",
    "GEN100-2-2-1000-50-20",
    "GEN102-3-2-1000-250-20",
]

# Deterministic exogenous state for the RDI-with-ExoState case.
EXO_BATCH_START = 30


def _exo_for(inst):
    S, L = inst.S, inst.L
    used = sorted({k for req in inst.orders_req.values() for k in req})
    return {
        "lanes": {(S[0], L[0]): [(0, 50)]},
        "blocks": {k: [(0, 100)] for k in used[:3]},
        "moves": [(10, 40), (20, 60)],
        "batch_start_time": EXO_BATCH_START,
    }


# ---------------------------------------------------------------------------
# Instances
# ---------------------------------------------------------------------------


def instance_specs() -> list[dict]:
    specs = []
    for st in (1, 2, 4):
        for ln in (1, 3):
            for no in (8, 30, 80):
                for mc in (None, 3, 20):
                    for hz in (10000, TIGHT_HORIZON):
                        specs.append(
                            {
                                "id": f"gen-s{st}-l{ln}-o{no}-m{mc}-h{hz}",
                                "kind": "gen",
                                "gen": dict(
                                    num_stations=st,
                                    lanes_per_station=ln,
                                    num_orders=no,
                                    num_skus=200 * no,
                                    seed=1000 + st * 100 + ln * 10 + no,
                                    pick_touch_time=4,
                                    movecap=mc,
                                ),
                                "horizon": hz,
                                "pickcap": None,
                            }
                        )
    # pickcap = 2 variants
    for no in (8, 30, 80):
        specs.append(
            {
                "id": f"gen-s2-l3-o{no}-m3-h10000-pc2",
                "kind": "gen",
                "gen": dict(
                    num_stations=2,
                    lanes_per_station=3,
                    num_orders=no,
                    num_skus=200 * no,
                    seed=7000 + no,
                    pick_touch_time=4,
                    movecap=3,
                ),
                "horizon": 10000,
                "pickcap": 2,
            }
        )
    for name in REAL_INSTANCES:
        path = os.path.join(REPO_ROOT, "precalculated_instances", name, "instance.pkl")
        if os.path.exists(path):
            specs.append(
                {"id": f"pkl-{name}", "kind": "pkl", "path": path, "horizon": 10000, "pickcap": None}
            )
    return specs


def build_instance(spec):
    if spec["kind"] == "gen":
        from datagen import generate_data

        inst = generate_data(**spec["gen"])
    else:
        from instance import Instance

        inst = Instance.from_pickle(spec["path"])
    if spec.get("pickcap") is not None:
        inst._pickcap = spec["pickcap"]
    return inst


# ---------------------------------------------------------------------------
# Cases: (case_id, heuristic, params, extras) per instance
# ---------------------------------------------------------------------------


def cases_for(spec, inst) -> list[tuple[str, str, dict, dict]]:
    """Return [(case_id, heuristic_key, params, extras)].

    params use the NEW (refactored) parameter names; the old adapter translates.
    extras: {"warmstart": bool}
    """
    hz = spec["horizon"]
    n_orders = len(inst.orders_req)
    n_st = len(inst.S)
    small = n_orders <= 30
    out = []

    def add(name, heur, params, warm=False):
        out.append((f"{spec['id']}|{name}", heur, {"horizon": hz, **params}, {"warmstart": warm}))

    add("sgc", "sgc", {}, warm=small)
    add("sgc_lb", "sgc", {"load_balanced": True})
    add("sgc_beta", "sgc", {"alpha": 1.0, "beta": 0.5})
    for tb in ("sharing_degree", "sum_rt_asc", "best_score"):
        add(f"rdi_{tb}", "rdi", {"tiebreaker": tb}, warm=small and tb == "sharing_degree")
    if small:
        for tb in ("sharing_degree", "sum_rt_asc"):
            for lazy in (True, False):
                for rk in (2, 3):
                    if lazy and rk == 2:
                        continue  # already covered above
                    add(f"rdi_{tb}_lazy{lazy}_k{rk}", "rdi", {"tiebreaker": tb, "use_lazy": lazy, "regret_k": rk})
        add("rdi_beta", "rdi", {"alpha": 1.0, "beta": 0.5})
        add("rdi_exo", "rdi", {"exo": _exo_for(inst)})
    add("cfss", "cfss", {})
    add("cfss_norm", "cfss", {"normalize_similarity": True})
    if small:
        add("cfss_wrapper_defaults", "cfss", {"beta": 1.0, "max_cluster_orders": 4})
    for rule in ("max_sharing", "critical_path", "readiness_weighted"):
        add(f"gbs_{rule}", "gbs", {"rule": rule})
    add("gbs_adaptive", "gbs", {"rule": "adaptive"})

    if spec["kind"] == "gen" and n_st > 1 and hz == 10000 and inst.movecap is not None:
        for method in ("random", "similarity", "load_balanced", "explicit"):
            params = {"partition": method, "seed": 123, "horizon": 50000}
            out.append((f"{spec['id']}|prdi_{method}", "partitioned_rdi", params, {"warmstart": False}))
        if n_orders <= 30:
            for method in ("random", "similarity", "load_balanced", "greedy", "lb_greedy", "full_rdi"):
                out.append((f"{spec['id']}|bench_{method}", "bench_metrics", {"method": method, "seed": 123}, {}))
    return out


def explicit_assignment(inst) -> dict[int, list[int]]:
    """Deterministic order->station-index assignment (round robin)."""
    out = {i: [] for i in range(len(inst.S))}
    for idx, o in enumerate(inst.O):
        out[idx % len(inst.S)].append(o)
    return out


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------


class _MockSP:
    def __init__(self):
        self.state_dict = {}

    def add_interval_var_solution(self, var, presence=False, start=None, end=None, size=None):
        d = {"present": 1 if presence else 0}
        if start is not None:
            d["start"] = start
        if end is not None:
            d["end"] = end
        if size is not None:
            d["size"] = size
        self.state_dict[var.get_name()] = d

    def add_integer_var_solution(self, var, value):
        self.state_dict[var.get_name()] = {"value": value}


class _MockModel:
    def create_empty_solution(self):
        return _MockSP()


def warmstart_fingerprint(sol, inst, horizon, validate_warmstart_fn):
    import cp_model

    _, handles = cp_model.build_model(inst, add_symmetry_breaking=True, horizon=horizon)
    sp = cp_model.inject_warmstart(sol, sol.pick_events, _MockModel(), handles)
    return {
        "state": sorted([k, sorted(v.items())] for k, v in sp.state_dict.items()),
        "violations": validate_warmstart_fn(sol, sol.pick_events, handles),
    }


def _ev(ev):
    return [
        ev.sku,
        ev.copy_id,
        ev.fetch_start,
        ev.fetch_end,
        ev.presence_start,
        ev.presence_end,
        ev.return_start,
        ev.return_end,
        list(ev.orders_served),
        [list(x) for x in getattr(ev, "pick_intervals", [])],
    ]


def fingerprint(sol, violations=None) -> dict | None:
    if sol is None:
        return None
    return {
        "feasible": bool(sol.feasible),
        "makespan": sol.makespan,
        "total_moves": sol.total_moves,
        "order_assignments": sorted([o, *v] for o, v in sol.order_assignments.items()),
        # list order within a station is preserved on purpose (inject_warmstart relies on it)
        "bin_events": {str(s): [_ev(e) for e in evs] for s, evs in sorted(sol.bin_events.items())},
        "bin_events_keys": [s for s in sol.bin_events.keys()],
        "pick_events": sorted([list(k), list(v)] for k, v in sol.pick_events.items()),
        "violations": violations,
    }


def strip_times(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if k not in ("total_time", "partition_time", "solve_time")}
