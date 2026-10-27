# Known issues

These issues were found during the heuristics refactor (October 2026). The refactor commit `0424939` kept them, so it
reproduces the pre-refactor commit `f391800` exactly. The follow-up commits resolved most of them. Each fix commit
records its effect, measured with `tests/regression/equivalence.py --ref <previous commit>`.

## Open

### Subproblem LNS: shared bin visits
In subproblem mode (`--lns-model subproblem`, `subproblem.py`) a bin visit is frozen as soon as it serves at least one
frozen order. Its fetch, presence and return become fixed intervals in the subproblem. Two consequences:
- **Active orders cannot join a frozen visit.** In the full-model LNS an active order can still pick from a visit
  that also serves frozen orders (the visit may move as long as it covers the frozen picks). In a subproblem it
  needs a new visit, so sharing with frozen orders is never created, and the subproblem never removes a fetch by
  merging into one.
- **The incumbent may not be representable.** If an active order currently picks from a frozen visit, its old pick
  is dropped from the warm start, and the subproblem must give it a new visit. The incumbent's objective may then be
  unreachable inside the subproblem. The frozen visit itself stays unchanged, so it is never shortened either.

How often: rare on the benchmark instances (about 2 000 bin visits for about 2 010 order lines on the 600-order
instances, so almost no sharing). Common on instances with few SKUs: in `tests/test_subproblem.py` (24 orders, 60
SKUs) most random neighbourhoods contain such an order. The number per iteration is in `FrozenContext.shared_picks`
and is logged at debug level.

Possible fixes: close each neighbourhood under sharing (add every frozen order that shares a visit with an active
order, as `_close_under_sharing` in the tests does), or pass frozen visits that serve active orders as joinable
"exo visits" with fixed fetch/return but open picks.

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

## Removed dead code

- `compute_U`, `static_priority_sort` (now `Rule("sharing_degree", False)`)
- `compute_dirty_set` (RDI), `score_cluster_at_station` / `snapshot_state` (CFSS)
- the unreachable half of `earliest_feasible_fetch`
- GBS-only `DifferenceArray.delay_for_movecap` / `range_max`, `BinCopyPool.peek_nth`
- `cp_model_utils.cp_sol_to_solution`, `decomposition.cp_solution_to_heuristic`
- the unused `manim` import in `instance.py`
