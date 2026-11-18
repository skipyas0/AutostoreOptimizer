# Known issues

These issues were found during the heuristics refactor (October 2026). The refactor commit `0424939` kept them, so it
reproduces the pre-refactor commit `f391800` exactly. The follow-up commits resolved most of them. Each fix commit
records its effect, measured with `tests/regression/equivalence.py --ref <previous commit>`.

## Open

### Subproblem LNS: frozen bin visits cannot move (joinable since Nov 2026, `--joinable-exo`, default on)
Terms: a *bin visit* is the triple `F[s,k,e]`, `B[s,k,e]`, `R[s,k,e]` (fetch, presence, return); a *pick*
`P[o,s,k,e]` assigns the consumption `C[o,k,s]` of order o to that visit (`alternative(C, {P[o,s,k,e]}ₑ)`).

In subproblem mode (`--lns-model subproblem`, `subproblem.py`) a visit is frozen as soon as it serves at least one
frozen order: its F, B and R become fixed intervals (`exo_moves`, `exo_station_bins`, `exo_blocks`).

**Fixed (`--joinable-exo`):** frozen visits of SKUs that an active order needs are also passed as `exo_visits`. Each
active order o gets an optional pick `PX[o,s,k,e]` (size `p[k]`) with its domain inside the fixed presence window,
as one more candidate of `alternative(C[o,k,s], …)`. `PX` and the frozen picks of the visit enter the station's
PickCap cumulative, and `PX` enters the pick sums of `idle_time` / `max_picks`. Visits where no further pick fits
(PickCap) are not passed. Consequences:
- The incumbent is always representable: an active order's pick on a frozen visit is warm-started on `PX`. Before,
  that pick was dropped from the warm start and CP had to open a new visit (+1 fetch); on the 600-order benchmark
  instance (`GEN4-15-3-28750-600-40`), 12 of 12 random 100-order neighbourhoods then had no solution within 5 s.
  Now all 12 reproduce the incumbent's KPIs exactly.
- Active orders can join a frozen visit, where its fixed presence window has room.

Validated in `tests/test_subproblem.py`: the subproblem optimum equals the optimum of the full docplex model with
the frozen orders and their visits fixed (all six KPIs, both backends); every merged solution is feasible in the
full model with the reported KPIs; the warm start is feasible when all variables are fixed to it.
`--no-joinable-exo` builds exactly the subproblems from before the option existed (identical `.cpo`).

**Still open: the frozen visit cannot move or stretch.** In the full-model LNS a visit that serves frozen orders
keeps free F, B, R (it only has to cover the frozen picks), so it can be stretched to take one more pick. Here its
presence window is fixed, and with PickCap = 1 a join needs an idle gap of at least `p[k]` inside it. On the
benchmark instance only about 9 of about 600 candidate frozen visits per 100-order neighbourhood have such a gap
(`PX` adds under 1% to the pick variables), and no join was found in 12 fetch-minimising solves of 5 s. A fix
would make the return of a joinable visit a variable (`R` movable, `B` end free, bounded below by the frozen picks).
The `bin_sharing` strategies (`refetched_sku_pick_slices`, `refetched_sku_fetch_density`,
`fetch_pick_ratio_timeslice`) work around this by selecting all orders in a time window around repeated visits of
a SKU, so the visits that could be merged are all active.

`--close-sharing` (default on, independent of `--joinable-exo`) still adds every frozen order that shares a visit
with an active order to the neighbourhood (`subproblem.close_under_sharing`, fixed point; about +2.5% orders and
0.03 s per iteration on the benchmark instance). It is no longer needed for a valid warm start, but it makes those
shared visits fully movable. `--no-close-sharing --no-joinable-exo` restores the subproblem experiments before
Nov 2026, in which neighbourhoods with shared picks (43% of the strategies' neighbourhoods on the benchmark
instance) started behind the incumbent.

### Acceptance is by makespan only
`Solver.rrt_acceptance` compares makespans, whatever the active objective. In a `bin_fetches` / `total_flow_time` /
`idle_time` phase, a solution that improves that KPI at equal makespan is logged as `*_No_Improve` (and accepted); one
that also worsens makespan beyond the RRT threshold is rejected. To see whether such phases work, compare the
phase's KPI (`current_<kpi>` columns of `experiment_dataframe.pkl`) before and after each iteration, not the status.

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
| `refetched_skus` / `cycle_time_weighted_refetched_skus` (re-enabled in `df6a647`) took all of the top `default_percent` SKUs by refetches. These are the popular SKUs, so about 470–500 of 600 orders became active on the benchmark instance (other strategies: 24–150), and one iteration took 140–170 s with `--iter-time-limit 5` | `match_random_demand`: a random subset of those top SKUs whose total demand (orders needing the SKU) stays within the expected demand of `strategy_random_skus` at the same `p`. Now about 170 orders and 11–12 s per iteration on the benchmark instance, the same size as `random_skus` |
| `analyze_experiment1.py` picked the "latest" experiment by text sort of `DD-MM-YYYY` names; `analyze_lns_experiments.py` crashed on unfinished experiment folders | sort by parsed date; unfinished folders are skipped |

## Removed dead code


- `compute_U`, `static_priority_sort` (now `Rule("sharing_degree", False)`)
- `compute_dirty_set` (RDI), `score_cluster_at_station` / `snapshot_state` (CFSS)
- the unreachable half of `earliest_feasible_fetch`
- GBS-only `DifferenceArray.delay_for_movecap` / `range_max`, `BinCopyPool.peek_nth`
- `cp_model_utils.cp_sol_to_solution`, `decomposition.cp_solution_to_heuristic`
- the unused `manim` import in `instance.py`
