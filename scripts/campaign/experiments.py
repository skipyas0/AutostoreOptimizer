"""Campaign experiment matrix: every experiment as (instances x starts x arms), each job one LNS (or CP) process.

Read by ``run.py``. A *job* is one (experiment, arm, start, instance) and runs ``runs`` LNS runs in one
matheuristic.py process (or one monolithic CP solve per seed). Hypotheses refer to the campaign plan in
``README.md``.

Configuration names
- ``BASE``: the subproblem LNS with the settings of the warm-start benchmark (``run_one.sh`` of
  scripts/warmstart_benchmark), i.e. the configuration all recent results were obtained with.
- ``FINAL``: the configuration the S3 experiments use. It equals BASE until the S2 screening has been
  analysed; then it is set here (and the code is frozen again) before S3 starts.

Resource estimates (``MEM_GB``, ``SETUP_S``) are per process and feed the scheduler only. Values marked
"guess" are replaced by the pilot (P0) measurements.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------------------------
# LNS configurations (matheuristic.py flags; True -> --flag, False -> --no-flag)
# ---------------------------------------------------------------------------------------------

BASE = {
    "backend": "cpp", "workers": 1, "lns-model": "subproblem", "neighbourhood": "graph",
    "symmetry-breaking": False, "iter-time-limit": 3, "iter-time-mode": "search",
    "strat-preset": "bin_sharing", "strat-choice": "uniform",
    "cp-objective-phase-schedule": "mmb", "cp-objective-phase-duration": 1,
    "improvement-constr": True, "improvement-ref": "current", "rrt-threshold": 0.05,
    "stagnation-th": 50, "default-percent": 0.2, "barcode": False,
}


def cfg(base: dict, **kw) -> dict:
    return {**base, **{k.replace("_", "-"): v for k, v in kw.items()}}


# S2 (2026-11-26, results/S1_S2_campaign): the per-iteration search limit is the only factor with signal,
# 10 s vs 3 s: -0.83% C (9/10 instances, p = 0.004), formed by 120 s and stable to 600 s. Objectives and
# strategy presets are within noise (|d| <= 0.3%). default-percent does not reach the bin_sharing strategies
# (all timeslice-based), so neighbourhood size is screened in S2b via default-timeslice-width.
FINAL = cfg(BASE, iter_time_limit=10)  # fallback until S2b is analysed


# ---------------------------------------------------------------------------------------------
# Resource model (per process). orders -> GB / seconds; measured values from the server logs.
# ---------------------------------------------------------------------------------------------

MEM_GB = {  # peak RSS with margin for 1 h growth. P0 (server, 2026-11-25): sub 15 iters 1.1/1.7/4.4/8.3 GB,
    # sub 1 h 5.3 (600) / 10.1 (800); full 15 iters 1.6/3.6/9.1/19.1; CP at 120 s 10.8 (600) / 20.9 (800)
    ("subproblem", 300): 1.5, ("subproblem", 400): 2.5, ("subproblem", 600): 6.5, ("subproblem", 800): 12.0,
    ("full", 300): 3.0, ("full", 400): 6.0, ("full", 600): 14.0, ("full", 800): 28.0,
    ("cp", 300): 4.0, ("cp", 400): 7.0, ("cp", 600): 14.0, ("cp", 800): 28.0,  # 300/400: extrapolated
}
SETUP_S = {300: 40, 400: 45, 600: 70, 800: 130}  # per run: wall time beyond the time limit (P0_long: 58 / 108 s)
ITER_S = {  # median s/iter on an idle machine (P0_lns); at 7 concurrent jobs x1.26 (P0_concurrency)
    ("subproblem", 300): 3.2, ("subproblem", 400): 3.5, ("subproblem", 600): 5.4, ("subproblem", 800): 6.6,
    ("full", 300): 4.2, ("full", 400): 7.4, ("full", 600): 17.6, ("full", 800): 33.3,
}


def orders_of(instance: str) -> int:
    return int(instance.split("-")[4])


@dataclass
class Experiment:
    name: str
    hypothesis: str
    sets: list[str]  # instance sets of instances.py; filtered by `shapes` (stations, orders) if given
    starts: list[str]  # start labels of starts.py
    arms: dict[str, dict]  # arm -> matheuristic flags (or {"cp": True, ...} for monolithic CP)
    runs: int = 2
    time_limit: float | None = 3600.0  # per run; None -> iters
    iters: int | None = None
    shapes: list[tuple[int, int]] | None = None
    instance_glob: str = "*"  # further filter on instance names
    seeds: list[int] = field(default_factory=lambda: [1])  # CP arms only
    stage: str = "S3"

    def model(self, arm: str) -> str:
        a = self.arms[arm]
        return "cp" if a.get("cp") else a["lns-model"]

    def mem_gb(self, arm: str, instance: str) -> float:
        return MEM_GB[(self.model(arm), orders_of(instance))]

    def est_seconds(self, arm: str, instance: str) -> float:
        n = orders_of(instance)
        if self.model(arm) == "cp":
            return len(self.seeds) * (self.time_limit + SETUP_S[n])
        per_run = self.time_limit if self.time_limit else self.iters * ITER_S[(self.model(arm), n)]
        return self.runs * (per_run + SETUP_S[n])


TENTH = [(10, 400), (15, 600)]
EVAL_STARTS = ["lb_sgc", "rdi_lb_share", "cfss_share", "gbs_no_defer", "gbs_max_sharing", "gbs_cluster_lpt",
               "gbs_regret", "gbs_regret_lc", "gbs_evolve_work", "gbs_evolve_makespan"]


def _screen_arms() -> dict[str, dict]:
    arms = {"base": BASE}
    # H6: objective schedules (phase duration in iterations)
    arms["obj_m"] = cfg(BASE, cp_objective_phase_schedule="m", cp_objective_phase_duration=0)
    for sched, dur in (("mmmb", 4), ("mmmf", 4), ("mmms", 4), ("bsm", 5)):
        arms[f"obj_{sched}{dur}"] = cfg(BASE, cp_objective_phase_schedule=sched, cp_objective_phase_duration=dur)
    arms["no_impr"] = cfg(BASE, improvement_constr=False)
    # H7: strategy presets and adaptive choice
    arms["strat_all"] = cfg(BASE, strat_preset="all")
    arms["strat_seem"] = cfg(BASE, strat_preset="seem_to_work_best")
    arms["strat_all_adapt"] = cfg(BASE, strat_preset="all", strat_choice="adaptive")
    arms["strat_bs_adapt"] = cfg(BASE, strat_choice="adaptive")
    # H8: neighbourhood size x per-iteration search limit (BASE = 0.2 / 3 s)
    for pct in (0.1, 0.2, 0.3):
        for itl in (2, 3, 5, 10):
            if (pct, itl) != (0.2, 3):
                arms[f"nbh{int(pct * 100)}_it{itl}"] = cfg(BASE, default_percent=pct, iter_time_limit=itl)
    return arms


EXPERIMENTS: dict[str, Experiment] = {}


def _add(e: Experiment) -> None:
    EXPERIMENTS[e.name] = e


# --- P0: pilots (resource measurements; minutes to ~2 h) ---------------------------------------
_add(Experiment("P0_lns", "resources: peak memory, setup and s/iter per model and size",
                sets=["eval", "scale"], starts=["gbs_cluster_lpt"],
                arms={"sub": BASE, "full": cfg(BASE, lns_model="full")},
                runs=1, time_limit=None, iters=15, stage="P0", instance_glob="GEN1-*",
                shapes=[(5, 300), (10, 400), (15, 600), (20, 800)]))
_add(Experiment("P0_long", "resources: memory growth over a full 1 h subproblem run", sets=["eval", "scale"],
                starts=["gbs_cluster_lpt"], arms={"sub": BASE}, runs=1, stage="P0", instance_glob="GEN1-*",
                shapes=[(15, 600), (20, 800)]))
_add(Experiment("P0_cp", "resources: monolithic CP memory and build time", sets=["eval", "scale"],
                starts=["gbs_cluster_lpt"], arms={"cp": {"cp": True}}, time_limit=120, stage="P0",
                instance_glob="GEN1-*", shapes=[(15, 600), (20, 800)]))

# --- S2: configuration screening on the tuning instances (one factor at a time around BASE) ------
_add(Experiment("S2_screen", "H6-H8: objectives, strategies, neighbourhood size x search limit",
                sets=["tune"], shapes=TENTH, starts=["gbs_cluster_lpt"], arms=_screen_arms(),
                runs=2, time_limit=600, stage="S2"))

# S2b: the real neighbourhood size (timeslice width; S2's default-percent does not reach bin_sharing) x longer
# search limits. 20 min runs, so 20-30 s iterations still get 40-60 iterations.
_S2B = {f"it{itl}_w{w}": cfg(BASE, iter_time_limit=itl, default_timeslice_width=w)
        for itl, w in ((10, 100), (20, 100), (30, 100), (10, 50), (10, 200), (20, 200))}
_add(Experiment("S2b_screen", "H8: timeslice width x search limit (follow-up of S2)", sets=["tune"], shapes=TENTH,
                starts=["gbs_cluster_lpt"], arms=_S2B, runs=2, time_limit=1200, stage="S2"))

# --- S3: main experiments on eval / scale / regime instances with FINAL ----------------------------
_add(Experiment("H1_starts", "H1/H4/H16: warm-start dose-response; headline (subproblem LNS, 1 h)",
                sets=["eval"], starts=EVAL_STARTS, arms={"final": FINAL}, runs=2))
_add(Experiment("H2_scale", "H2: 20 stations / 800 orders", sets=["scale"],
                starts=["lb_sgc", "gbs_cluster_lpt", "gbs_regret_lc", "gbs_evolve_work"],
                arms={"final": FINAL}, runs=2))
_add(Experiment("H3_objective", "H3: can W-targeted objective phases recover a poor start's robot work?",
                sets=["eval"], shapes=TENTH, starts=["lb_sgc", "gbs_regret_lc"],
                arms={"obj_mmmb4": cfg(FINAL, cp_objective_phase_schedule="mmmb", cp_objective_phase_duration=4),
                      "obj_bsm5": cfg(FINAL, cp_objective_phase_schedule="bsm", cp_objective_phase_duration=5)},
                runs=2))
_add(Experiment("H5_full", "H5/H16: full-model LNS at equal wall time", sets=["eval", "scale"],
                starts=["lb_sgc", "gbs_regret_lc"], arms={"full": cfg(FINAL, lns_model="full")}, runs=2))
_add(Experiment("H10_regime", "H10: start effect in other regimes (PickCap 2, MoveCap 80, 2 copies)",
                sets=["regime_lns"], starts=["lb_sgc_share", "gbs_regret_lc"], arms={"final": FINAL}, runs=2))
_add(Experiment("H16_cp", "H16: monolithic CP, warm-started, 1 h", sets=["eval", "scale"],
                starts=["lb_sgc", "gbs_regret_lc"], arms={"cp": {"cp": True}}, seeds=[1]))
