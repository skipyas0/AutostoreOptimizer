# Known issues

These were found during the heuristics refactor (October 2026). The refactor **kept them on purpose**, so it
reproduces the reference commit `f391800` exactly; see `tests/regression/equivalence.py`. Two exceptions are marked ⚑.

Evidence is from the equivalence grid (114 instances, 2259 cases) unless stated otherwise.

## Severity: high

### 1. AMA crashed after the pickcap change. ⚑ Reframed.
- **At the reference commit:** `run_sgc_parameterised` and `run_ama_sgc` crashed with
  `ValueError: too many values to unpack`. Since `ff676ba` ("fixing pickcap in heuristics"), `find_shared_bin`
  returns 4 values, while AMA unpacked 2.
- **Further bugs in the old AMA planner.** It never passed `pick_cap`. It never considered extending a bin's presence.
  It appended picks to *committed* bins' `pick_intervals` while only tentatively planning, so state leaked between
  candidate stations. It also produced 3-tuple `shared_picks`, which `commit_plan` cannot consume.
- **Now:** AMA is `run_ama` = "run SGC for each dispatching-rule pair, keep the best". SGC has `order_rule` / `sku_rule`
  parameters, and its default rules reproduce the old SGC bit-for-bit.
- **Consequence:** the AMA dataset (`ama_sgc_dataset_analysis.ipynb`, `learning_ama_archive/`) was produced with an older
  planner, which had no pickcap and no presence extension. Regenerating it with `trainset_generator.py` gives different numbers.

### 2. CFSS ignores PickCap
`run_cfss` plans with `pick_cap=None` (`init_state(..., ignore_pickcap=True)`), so several orders can pick from the same
bin simultaneously. 43 of 114 instances end up "feasible" but violate PickCap, e.g. `PickCap violation at S0 SKU 1341: 2 > 1`.
**Fix:** drop `ignore_pickcap=True` in `cfss.py` (one line), then re-baseline.

### 3. GBS ignores PickCap and reports false feasibility
- Every order served by a bin visit "picks" during the **whole** presence window at the same time
  (`schedule_bin_event`). With PickCap 1 this violates the cap whenever more than one order shares a bin.
- `feasible` is effectively always `True`. The check counts the *keys* of a per-SKU pick dictionary that is pre-filled
  with `(0, 0)` defaults, so orders that were never picked (horizon hit, loop aborted) still count as complete.
  This happened in 75 of 114 instances (feasible=True with "missing pick" and PickCap violations).
- `alpha` and `beta` are only used to choose between rules in `rule="adaptive"`. GBS also does not support `ExoState`
  or batch start times.

### 4. ⚑ GBS adaptive crashed (fixed minimally)
`run_gbs_adaptive` rebuilt `Instance(...)` without the required `movecap` / `seed` and crashed with a TypeError. The
deep copies were pointless, so `run_gbs(rule="adaptive")` now runs the three rules on the same instance and applies the
original selection logic. The harness compares it against that logic applied to the old `run_gbs`.

### 5. Partitioned RDI: MoveCap split can give a station 0 robots
`distribute_movecap(3, 4)` returns `[1, 1, 1, 0]`. A station with MoveCap 0 can never fetch a bin, so all of its orders
fail. This happens in all 24 partitioned cases with 4 stations and MoveCap 3. It affects `benchmark_models.py`
(methods 1–3 and RL), `rl_order_assignment.evaluate_assignment_reward` and `decomposition.solve_*`.

## Severity: medium

### 6. Partitioned RDI ignores cross-station bin-copy limits
Stations are solved independently, each assuming all `N[k]` copies are free. The stitched solution can therefore exceed
`N[k]` (`SKU k Block concurrency 2 > N[k]=1`), which happened in 11 of 108 partitioned cases. The CP-based
`decomposition.solve_station_decomposition` avoids this by passing `exo_blocks` between stations.

### 7. Random partitions are not reproducible
`create_random_partitions` draws from the **global** `random` module, and the benchmark never seeds it. Importing
`neighborhood_selection` (and through it `matheuristic_plots`) also consumes global random numbers. Pass
`seed=` to `run_partitioned_rdi`; it seeds after importing.

### 8. RDI tiebreakers: `best_score` is the same as `sharing_degree`
The priority key is `(priority, #shared, sharing_degree, tiebreak, -best_score)`. With `tiebreaker="sharing_degree"` the
4th element duplicates the 3rd, and with `"best_score"` it duplicates the 5th. Both give the same ordering, and the
results were identical on all 114 instances. Only `sum_rt_asc` changes the behaviour.

### 9. Defaults that differ between entry points
- RDI: the function default tiebreaker is `sharing_degree`; the old CLI and `solve_heuristic_instance` default was
  `sum_rt_asc`, which now lives on as registry entry `rdi_sum_rt`.
- CFSS: the function defaults are `beta=0.0, max_cluster_orders=8`; the old `solve_heuristic_instance` wrapper used
  `beta=1.0, max_cluster_orders=4`.
- `validate_solution(horizon=10000)` by default, while benchmark / partitioned runs use horizon 50000. Always pass the
  horizon you solved with.

### 10. Ties are broken by set iteration order
RDI (`for o in unscheduled`) and CFSS (`for o in unassigned[s]`, `active` clusters) pick the first-seen candidate among
exact ties while iterating a `set` of order ids. The result is deterministic for a given construction history, but
fragile: building the set differently changes results. The refactor preserves the exact construction.

## Severity: low / cosmetic

11. **Benchmark metric inconsistency.** For an infeasible *monolithic* solution, every station gets `W_k = 0`. For an
    infeasible station of a *partitioned* solution, `W_k` is its assigned workload. Both are preserved in `solution_metrics`.
    `compute_metrics_from_station_results` also fails for `movecap=None` (`max(None, 1)`).
12. **`Instance.pickcap`** falls back to `len(L)` for objects without `_pickcap` (very old pickles), but defaults to 1 for
    new instances.
13. **RDI planner:** `desired_presence = max(t_cursor, rt[k])` uses `rt[k]` as an absolute lower bound on time. This is
    harmless at batch start 0 but odd with `batch_start_time > 0`.
14. **AMA `two_phase`:** the phase-1 objective is never stored in `best_obj`, so phase 2 always replaces the phase-1
    winner with its first feasible run. Because phase 2 includes the phase-1 combination, the final result is still
    the best run seen.
15. **Duplicated CP→heuristic converters:** `cp_model_utils.cp_sol_to_solution` and
    `decomposition.cp_solution_to_heuristic` overlap. `MockCpoModel` / `MockCpoStartingPoint` exist in both
    `matheuristic.py` and `cpp_cp_model.py`.
16. **Partitioned RDI with zero orders** returns an empty feasible solution; the old stitching returned `None`.

## Removed dead code
- `compute_U`, `static_priority_sort` (now `Rule("sharing_degree", False)`)
- `compute_dirty_set` (RDI), `score_cluster_at_station` / `snapshot_state` (CFSS)
- the unreachable second half of `earliest_feasible_fetch`
- the unused `manim` import in `instance.py`

## Legacy code broken by the refactor (not maintained)
These still import the removed modules (`autostore_heuristic`, `heuristic_*`) or `solve_heuristic_instance(config)`:
- `benchmarking/*`
- `analyze_heuristics/*`
- `scripts/run_custom_warmstart*.py`

Port them by generating the `Instance` with `datagen.generate_data(...)` and calling `heuristics.run_heuristic(name, inst, horizon=...)`.
