# Known issues

These issues were found during the heuristics refactor (October 2026). The refactor commit `0424939` kept them, so it
reproduces the pre-refactor commit `f391800` exactly. The follow-up commits resolved most of them. Each fix commit
records its effect, measured with `tests/regression/equivalence.py --ref <previous commit>`.

## Open

### Subproblem LNS: shared bin visits (mitigated by `--close-sharing`, default on)
In subproblem mode (`--lns-model subproblem`, `subproblem.py`) a bin visit is frozen as soon as it serves at least one
frozen order. Its fetch, presence and return become fixed intervals in the subproblem. Two consequences:
- **Active orders cannot join a frozen visit.** In the full-model LNS an active order can still pick from a visit
  that also serves frozen orders (the visit may move as long as it covers the frozen picks). In a subproblem it
  needs a new visit, so sharing with frozen orders is never created, and the subproblem never removes a fetch by
  merging into one.
- **The incumbent may not be representable.** If an active order currently picks from a frozen visit, its old pick
  is dropped from the warm start, and the subproblem must give it a new visit. The starting point is then worse than
  the incumbent in every KPI (one extra fetch per dropped pick, often more flow time, sometimes more makespan).

How often (measured Nov 2026): 43% of the strategies' neighbourhoods on the 600-order benchmark instance
(`GEN4-15-3-28750-600-40`) contain such a pick, and most neighbourhoods on instances with ≤ 1000 SKUs. Without a fix,
every such iteration starts behind the incumbent; with `--improvement-constr` the `bin_fetches` / `idle_time` /
`total_flow_time` phases then almost never find a solution, and accepted solutions slowly lose bin sharing.

**Mitigation:** `--close-sharing` (default) adds every frozen order that shares a visit with an active order to the
neighbourhood (`subproblem.close_under_sharing`, fixed point). The warm start then reproduces the incumbent exactly
(all five KPIs, checked in `tests/test_subproblem.py`). It grows neighbourhoods by about 2.5% on the benchmark
instance and costs about 0.03 s per iteration. The logged unfrozen set (`num_optimized`, barcodes) stays the
strategy's selection. `--no-close-sharing` restores the behaviour of the subproblem experiments before Nov 2026.
The first consequence (no new sharing with frozen orders) remains; a fix would pass frozen visits that serve active
orders as joinable "exo visits" with fixed fetch/return but open picks.

### Acceptance is by makespan only
`Solver.rrt_acceptance` compares makespans, whatever the active objective. In a `bin_fetches` / `total_flow_time` /
`idle_time` phase, a solution that improves that KPI at equal makespan is logged as `*_No_Improve` (and accepted); one
that also worsens makespan beyond the RRT threshold is rejected. To see whether such phases work, compare the
phase's KPI (`current_<kpi>` columns of `experiment_dataframe.pkl`) before and after each iteration, not the status.

### Refetched-SKU strategies select most of the instance
`refetched_skus` and `cycle_time_weighted_refetched_skus` (re-enabled in `df6a647`) pick the top `default_percent`
of SKUs by number of fetches, and every order needing one of them becomes active. The most refetched SKUs are the
popular ones, so on the 600-order benchmark instance they select about 470–500 of 600 orders (all other strategies:
24–150). One such iteration took 140–170 s (presolve of an almost full model) with `--iter-time-limit 5`, as
`--iter-time-mode search` does not bound extraction / presolve. Since the code freeze both use a fixed `p = 0.01 * severity`
(independent of `--default-percent`); they still select about 340 of 600 orders at severity 1.

### Sharing under PickCap = 1 is structurally limited
With PickCap = 1, sharing a bin visit means picking *after* the previous order, so the presence must be stretched
while the next order's picker is ready.

On benchmark instances (robot-bound: makespan is about 8.5× the pure picking time per station):
- If every SKU's order lines shared one bin visit, pick/bin would reach about 0.45.
- The planners reach about 0.025 (SGC) and 0.05 (RDI). With PickCap = |L| they reached about 0.05 and 0.10.

The main reason is the forward-only, append-only construction. When a later order needs the same SKU, the bin has
usually already left (67% of failed lookups for SGC, 84% for RDI); presence extension is rarely the blocker.

The opt-in `extend_share_window` and `share_reward` options (see HEURISTICS.md) roughly triple sharing and cut
makespan by about 3%. A structural fix would plan bin-centrically, e.g. serving all pending orders that need SKU k at
a station from one visit (a generalisation of single-item piggybacking).

### RDI's legacy share bonus
`legacy_share_bonus` (default 0.5) credits `rt + p` per shared pick. It was added together with pickcap in `ff676ba`
and was never calibrated. A shared pick saves `rt + rt_ret` of robot time, and no pick time under PickCap = 1.
The `share_reward` (λ) option uses `rt + rt_ret`; the default keeps the legacy term so existing results stay
reproducible.

### Partitioned RDI ignores cross-station bin-copy limits (now detected)

Stations are solved independently, each assuming all `N[k]` copies are free, so the stitched solution can exceed
`N[k]`. This is now **detected**:
- `run_partitioned_rdi` warns and sets `feasible = False`;
- `meta["copy_conflicts"]` counts the SKUs that are over their limit;
- the solution can therefore never be used as a warm start (see "strict warm starts" below).

Per-station feasibility (`meta["station_feasible"]`), and hence the benchmark metrics, are unaffected. The CP-based
`decomposition.solve_station_decomposition` avoids the problem by passing `exo_blocks` between stations.
**To do:** coordinate copies across stations, e.g. by solving stations sequentially with `ExoState.blocks`.

### Stations without a robot (MoveCap < |S|) are simply switched off

Partitioned RDI and the spatial decompositions use only the first `min(|S|, MoveCap)` stations
(`active_station_count`). With an explicit assignment (e.g. from the RL policy), a station that gets MoveCap 0 is not
solved: its orders stay unscheduled, a warning is issued and the station is marked infeasible. **To do:** something smarter, such as sharing
robots across stations or time-multiplexing them.

### RDI tiebreakers `best_score` and `sharing_degree` are identical

The priority key is `(priority, #shared, sharing_degree, tiebreak, -best_score)`. With `tiebreaker="sharing_degree"`
the 4th element duplicates the 3rd, and with `"best_score"` it duplicates the 5th, so both give the same ordering.
The results were identical on all 114 regression instances. Only `sum_rt_asc` (now the default) changes behaviour.

### RDI planner: absolute `rt[k]` bound (time decomposition)

`plan_order_gap_filling` uses `desired_presence = max(t_cursor, rt[k])`, so `rt[k]` acts as an *absolute* bound.
With `ExoState.batch_start_time > 0` the meaningful bound would be `batch_start_time + rt[k]`. This is harmless today
because `t_cursor >= batch_start_time`. A note sits at that line; revisit it when working on time-based decomposition.

### AMA `two_phase` quirk (accepted)

The phase-1 objective is never stored in `best_obj`, so phase 2 always replaces the phase-1 winner with its first
feasible run. Phase 2 includes the phase-1 combination, so the final result is still the best run seen.

### AMA dataset is not reproducible with the current planner

AMA crashed at `f391800`. It now runs the current SGC, which handles pickcap and extends bin presence. The dataset
behind `ama_sgc_dataset_analysis.ipynb` / `learning_ama_archive/` was produced with an older planner, so
`trainset_generator.py` gives different numbers.

### Legacy scripts are broken (not maintained)

These scripts still import the removed modules (`autostore_heuristic`, `heuristic_*`) or `solve_heuristic_instance`:
- `benchmarking/*`
- `analyze_heuristics/*`
- `scripts/run_custom_warmstart*.py`
- `scripts/verify_p_explosion*.py`

To port one, build the `Instance` with `generate_data(..., movecap=...)` and call `heuristics.run_heuristic(name, inst)`.

### Accepted behaviour

Partitioned RDI with zero orders returns an empty feasible solution.

---

## Resolved

| Issue | Resolution | Commit |
|---|---|---|
| AMA crashed (`find_shared_bin` returned 4 values since `ff676ba`; the old planner also leaked state and ignored pickcap) | AMA = best of several SGC runs (`run_ama`, modes `full_grid` and `two_phase`; registry `ama`, `ama_two_phase`, `*_singles`) | `0424939` |
| GBS ignored PickCap, reported false feasibility, `adaptive` crashed | GBS removed (poor performance) | `ed5cadc` |
| `movecap=None` and pickcap-less pickles (fallback `len(L)`) | `Instance` requires integer `movecap >= 1`, `pickcap >= 1`; `generate_data(movecap=...)` is required. The 35 legacy instance pickles were migrated with pickcap = \|L\| (`scripts/migrate_heuristic_pickles.py --instances`) | `ed5cadc`, `a5d4c62` |
| Duplicated CP→heuristic converters (the decomposition one left `orders_served` empty); duplicated mock CP models | `HeuristicSolution.from_cp` used everywhere; `matheuristic` uses `HeuristicSolution.to_(cpp_)warmstart` | `ed5cadc` |
| An infeasible heuristic solution could silently become a CP warm start | Strict warm starts: `to_warmstart`/`to_cpp_warmstart` raise on infeasible solutions or any hard `validate_warmstart` violation, which now also checks PickCap and horizon. Decomposition falls back to a CP cold start. All 50 saved warm starts pass | `ed5cadc` |
| CFSS ignored PickCap (43/114 instances invalid) | CFSS plans with `inst.pickcap`: 0 PickCap violations, makespan +4% on the affected cases | `9679066` |
| Ties in RDI/CFSS were broken by set iteration order | Ordered dicts / sorted iteration (RDI results unchanged; CFSS ±0 on average) | `f509335` |
| Partitioned RDI gave stations MoveCap 0 (all their orders failed) | Robot-less stations switched off (see Open) | `a5d4c62` |
| Random partitions not reproducible (global RNG, consumed by imports) | Local `random.Random(seed)`, default `seed=0`; similarity partitions likewise | `a5d4c62` |
| Defaults differed between entry points | RDI `tiebreaker="sum_rt_asc"` everywhere (registry `rdi`; `rdi_sharing_degree` and `rdi_best_score` remain as variants); CFSS defaults `beta=1.0, max_cluster_orders=4`; horizon 10000 everywhere (benchmark, partitioned RDI, RL reward, RL decomposition) | `4d3f486` |
| Benchmark: infeasible monolithic solutions got `W_k = 0`; `movecap=None` crashed the metrics | `W_k` is always the assigned workload; infeasible stations get the 50000 makespan penalty; `movecap` can no longer be None | `4d3f486` |
| RDI lazy cache could commit a stale plan that broke MoveCap: when deciding which cached plans to re-plan after a commit, only *new* bin moves were compared, not the shifted return of an extended shared bin. It did not occur in the regression grid with default settings, but did with the sharing options (`MoveCap violation: 5 > 4`) | `_plan_moves` includes extended shared-bin returns; regression test `test_rdi_lazy_cache_keeps_movecap_with_extensions` | after `4d3f486` |

Found in the pre-code-freeze audit (Nov 2026); runs with these settings from before are affected as noted:

| Issue | Resolution |
|---|---|
| docplex full-model LNS: after the first iteration, auxiliary variables of the solve result (`station_span_*`, `shift_len_*`, `shift_cost_*`) were frozen with the rest, pinning every station's span, so makespan could not improve (every iteration `Optimal_No_Improve` / `Infeasible`). Since `b3f3bad` (9 Oct 2026); the C++ backend was not affected | `freezing_utils.lns_decision_vars`: only LNS decision variables are frozen. Test `tests/test_lns_full_docplex.py` |
| docplex full-model LNS with `--improvement-constr`: the bound constraint was never removed (`CpoModel.add` returns `None`), so bounds piled up (with objective cycling: bounds on every KPI at once → infeasible) | the expression is kept and removed before the next solve |
| docplex full-model LNS with `--no-delta-freezing` crashed (the strategy result was passed instead of the variable set) | passes `to_optimize` |
| `--improvement-constr` bounded non-makespan objectives by the *best* value seen in the run, makespan by the *current* one. With makespan acceptance the incumbent drifts away from those bests, so the bound soon became unreachable (iterations time out with `Unknown`) | `--improvement-ref current` (default) bounds every objective by the incumbent's value; `best` restores the old behaviour |
| `--runs > 1`: the objective, the cycle position and `best_idle_time` / `best_max_picks` carried over from the previous run (run 2 started mid-cycle, and the improvement bound could refer to the wrong objective) | reset at the start of every run |
| `--cp-objective X` crashed when X was not in `--cp-objective-phase-schedule` | starts at the first phase |
| The initial solution had no `idle_time` / `max_picks` KPIs, so their first phases had no improvement bound | `matheuristic.heuristic_kpis` (same definitions as the CP models) |
| A `w` (wages) phase with `--backend cpp` or in subproblem mode silently optimised makespan | rejected with an error; the C++ wrapper rejects unknown objectives |
| Two runs started in the same second on the same instance wrote into the same experiment folder | `VisualLogger` waits for a free timestamp (folder name format unchanged) |
| `--strat-preset bottlenecks` silently dropped `refetched_skus` (old name) | preset fixed; `bottlenecks` and `seem_to_work_best` are now CLI choices |
| `analyze_experiment1.py` picked the "latest" experiment by text sort of `DD-MM-YYYY` names; `analyze_lns_experiments.py` crashed on unfinished experiment folders | sort by parsed date; unfinished folders are skipped |

## Removed dead code


- `compute_U`, `static_priority_sort` (now `Rule("sharing_degree", False)`)
- `compute_dirty_set` (RDI), `score_cluster_at_station` / `snapshot_state` (CFSS)
- the unreachable half of `earliest_feasible_fetch`
- GBS-only `DifferenceArray.delay_for_movecap` / `range_max`, `BinCopyPool.peek_nth`
- `cp_model_utils.cp_sol_to_solution`, `decomposition.cp_solution_to_heuristic`
- the unused `manim` import in `instance.py`
