"""Constructive heuristics for joint order & bin scheduling in RCS/RS (AutoStore-like) systems.

Every heuristic has the signature ``run_x(inst: Instance, *, horizon=..., **params) -> HeuristicSolution``
(MoveCap / PickCap always come from the Instance).

    from heuristics import run_heuristic, HEURISTICS
    sol = run_heuristic("rdi", inst, horizon=10000)
    sol.validate(inst, 10000); sp = sol.to_warmstart(mdl, handles)

``HEURISTICS`` maps a registry name to ``(function, default params)``; see
docs/HEURISTICS.md for the algorithms and how to add a new one.
"""

from __future__ import annotations

import time
from typing import Callable

from heuristics.cfss import run_cfss
from heuristics.heuristic_utils import BinEvent, ExoState
from heuristics.partitioned_rdi import PARTITION_METHODS, run_partitioned_rdi
from heuristics.rdi import TIEBREAKERS, run_rdi
from heuristics.sgc import ALL_RULES, Rule, run_ama, run_sgc
from heuristics.solution import HeuristicSolution, build_viz_handles, validate_solution, validate_warmstart
from instance import Instance

HEURISTICS: dict[str, tuple[Callable[..., HeuristicSolution], dict]] = {
    "sgc": (run_sgc, {}),
    "lb_sgc": (run_sgc, {"load_balanced": True}),
    "sgc_singles": (run_sgc, {"piggyback_singles": True}),
    "lb_sgc_singles": (run_sgc, {"load_balanced": True, "piggyback_singles": True}),
    # bin-sharing variants: wait for an existing bin up to the own new-fetch alternative and
    # credit lambda=1 x (rt + rt_ret) robot time per shared pick
    "sgc_share": (run_sgc, {"extend_share_window": True, "share_reward": 1.0}),
    "lb_sgc_share": (run_sgc, {"load_balanced": True, "extend_share_window": True, "share_reward": 1.0}),
    "sgc_share_window": (run_sgc, {"extend_share_window": True}),
    "ama": (run_ama, {"mode": "full_grid"}),
    "ama_two_phase": (run_ama, {"mode": "two_phase"}),
    "ama_singles": (run_ama, {"mode": "full_grid", "piggyback_singles": True}),
    "ama_two_phase_singles": (run_ama, {"mode": "two_phase", "piggyback_singles": True}),
    "lb_ama_two_phase_singles": (run_ama, {"mode": "two_phase", "load_balanced": True, "piggyback_singles": True}),
    "ama_two_phase_share": (run_ama, {"mode": "two_phase", "extend_share_window": True, "share_reward": 1.0}),
    "rdi": (run_rdi, {"tiebreaker": "sum_rt_asc"}),
    "rdi_sharing_degree": (run_rdi, {"tiebreaker": "sharing_degree"}),
    "rdi_best_score": (run_rdi, {"tiebreaker": "best_score"}),
    "rdi_share": (run_rdi, {"extend_share_window": True, "share_reward": 1.0, "legacy_share_bonus": 0.0}),
    "rdi_share_window": (run_rdi, {"extend_share_window": True}),
    "rdi_share_reward": (run_rdi, {"share_reward": 1.0, "legacy_share_bonus": 0.0}),
    "partitioned_rdi_random": (run_partitioned_rdi, {"partition": "random"}),
    "partitioned_rdi_similarity": (run_partitioned_rdi, {"partition": "similarity"}),
    "partitioned_rdi_load_balanced": (run_partitioned_rdi, {"partition": "load_balanced"}),
    "partitioned_rdi_load_balanced_share": (
        run_partitioned_rdi,
        {"partition": "load_balanced", "extend_share_window": True, "share_reward": 1.0, "legacy_share_bonus": 0.0},
    ),
    "cfss": (run_cfss, {}),
    "cfss_normalized": (run_cfss, {"normalize_similarity": True}),
}

# Alternative spellings accepted by run_heuristic (e.g. LNS precalculation configs).
ALIASES = {
    "greedy": "sgc",
    "standard_greedy": "sgc",
    "standard_sgc": "sgc",
    "full_greedy": "sgc",
    "load_balanced_greedy": "lb_sgc",
    "load_balanced_sgc": "lb_sgc",
    "load_balancing_sgc": "lb_sgc",
    "load_balancing_greedy": "lb_sgc",
    "lb_greedy": "lb_sgc",
    "rdi_sgc": "rdi",
    "full_rdi": "rdi",
    "ama_sgc": "ama",
    "cfss_sgc": "cfss",
}
# NB: "load_balanced_rdi" / "random_rdi" / "similarity_rdi" in LNS precalculation configs
# mean the CP-based station decomposition (decomposition.py), not partitioned_rdi_*.


def resolve(name: str) -> str:
    key = name.lower().replace("-", "_")
    key = ALIASES.get(key, key)
    if key not in HEURISTICS:
        raise ValueError(f"Unknown heuristic {name!r}. Available: {sorted(HEURISTICS)} (+ aliases {sorted(ALIASES)})")
    return key


def run_heuristic(name: str, inst: Instance, **params) -> HeuristicSolution:
    """Run a registered heuristic; `params` override its registry defaults.

    Adds ``meta["name"]`` and ``meta["runtime"]`` (wall-clock seconds).
    """
    key = resolve(name)
    fn, defaults = HEURISTICS[key]
    t0 = time.perf_counter()
    sol = fn(inst, **{**defaults, **params})
    if sol is not None:
        sol.meta["name"] = key
        sol.meta["runtime"] = time.perf_counter() - t0
    return sol


__all__ = [
    "ALIASES",
    "ALL_RULES",
    "BinEvent",
    "ExoState",
    "HEURISTICS",
    "HeuristicSolution",
    "PARTITION_METHODS",
    "Rule",
    "TIEBREAKERS",
    "build_viz_handles",
    "resolve",
    "run_ama",
    "run_cfss",
    "run_heuristic",
    "run_partitioned_rdi",
    "run_rdi",
    "run_sgc",
    "validate_solution",
    "validate_warmstart",
]
