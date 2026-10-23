# Heuristics (`heuristics/` package)

Every heuristic is a plain function with the same contract:

```python
run_x(inst: Instance, *, horizon: int = ..., **algorithm_params) -> HeuristicSolution
```

- MoveCap and PickCap are always read from the `Instance`, never passed separately.
- No `*args` and no instance generation inside: the caller builds or loads the `Instance`.
- Parameters are keyword-only, explicit, and have defaults.

```python
from heuristics import run_heuristic, run_rdi, HEURISTICS
sol = run_heuristic("lb_sgc", inst, horizon=10000)          # by registry name
sol = run_rdi(inst, horizon=10000, tiebreaker="sum_rt_asc")  # or directly
violations = sol.validate(inst, horizon=10000)
sp = sol.to_warmstart(mdl, handles)          # docplex model from cp_model.build_model
sol.to_cpp_warmstart(mdl, handles)           # C++ model from cpp_cp_model.build_model
fig = plot_schedule(*sol.to_viz(inst))       # schedule_visualizer
```

CLI: `python -m heuristics --list`, `python -m heuristics <name> [--orders 40 ...] [--param beta=0.5] [--vis]`.

## Package layout

| File | Content |
|---|---|
| `__init__.py` | Public API, the `HEURISTICS` registry (name → `(function, default params)`), `ALIASES`, `run_heuristic` (adds `meta["name"]` and `meta["runtime"]`) |
| `__main__.py` | Single CLI for all heuristics |
| `heuristic_utils.py` | Shared machinery for the SGC family (below) |
| `solution.py` | `HeuristicSolution`, `validate_solution`, `validate_warmstart`, `build_viz_handles` |
| `sgc.py` | SGC planner and loop, dispatching `Rule`s, `ALL_RULES`, AMA |
| `rdi.py` | RDI (regret insertion with the gap-filling planner) |
| `partitioned_rdi.py` | Order→station partitioning, `distribute_movecap`, partitioned RDI |
| `cfss.py` | Clustering plus interleaved SGC |
| `gbs.py` | Bin-centred GBS, with its own state classes |

## Registry

| Name | Call |
|---|---|
| `sgc` | `run_sgc()` |
| `lb_sgc` | `run_sgc(load_balanced=True)` |
| `sgc_singles` / `lb_sgc_singles` | `run_sgc(piggyback_singles=True)` / `run_sgc(load_balanced=True, piggyback_singles=True)` |
| `ama` / `ama_two_phase` | `run_ama(mode="full_grid" / "two_phase")` |
| `ama_singles` / `ama_two_phase_singles` / `lb_ama_two_phase_singles` | `run_ama(..., piggyback_singles=True[, load_balanced=True])` |
| `rdi` / `rdi_sum_rt` / `rdi_best_score` | `run_rdi(tiebreaker="sharing_degree" / "sum_rt_asc" / "best_score")` |
| `partitioned_rdi_{random,similarity,load_balanced}` | `run_partitioned_rdi(partition=...)`, horizon 50000 |
| `cfss` / `cfss_normalized` | `run_cfss(normalize_similarity=False / True)` |
| `gbs_{max_sharing,critical_path,readiness_weighted,adaptive}` | `run_gbs(rule=...)` |

Aliases such as `greedy`, `load_balanced_greedy` and `rdi_sgc` are accepted by `run_heuristic`
and by `precalculate_instances_for_lns.py`.

The registry replaces the old per-variant modules. For example, `heuristic_rdi_sgc_best_score.py`
became `rdi_best_score`, `heuristic_cfss_sgc_normalized.py` became `cfss_normalized`, and
`heuristic_gbs_critical_path.py` became `gbs_critical_path`.

---

## HeuristicSolution

```python
@dataclass
class HeuristicSolution:
    order_assignments: dict[o, (station, lane, start, end)]
    bin_events: dict[station, list[BinEvent]]
    makespan: int
    total_moves: int            # 2 * number of bin visits
    feasible: bool
    pick_events: dict[(o, station, sku), (pick_start, pick_end)]
    meta: dict                  # name, runtime, rule/tiebreaker, AMA runs, partition info, ...
```

Methods:
- `validate(inst, horizon)`: checks assignment, lanes, pickface, copies (N), MoveCap, timing, horizon, pick completeness and PickCap.
- `validate_warmstart(handles)`: checks consistency with the CP model's indexing.
- `to_warmstart`, `to_cpp_warmstart`, `to_viz`.
- `objective(alpha, beta, stations)`: `alpha * C_max + beta * Σ_s span_s`.
- `HeuristicSolution.stitch(sols)`: union of solutions over disjoint orders.

`BinEvent` is
`(sku, copy_id, fetch_start, fetch_end, presence_start, presence_end, return_start, return_end, orders_served, pick_intervals)`.
`copy_id` is the physical copy (0..N[k]-1).

---

## Shared machinery (`heuristic_utils.py`)

| Piece | Role |
|---|---|
| `HeuristicState` | Committed partial schedule: lane / pickface free times, `DifferenceArray` MoveCap profile, one `BinCopyPool` per SKU, bin events and pickface intervals per station. It also carries `inst`, `horizon` and `pick_cap`, so helpers take only `state`. |
| `init_state(inst, horizon, exo=None, ignore_pickcap=False)` | Initial state. `ExoState(lanes, blocks, moves, batch_start_time)` pre-occupies resources; it is used by decomposition. |
| `PlanScratch` | Tentative bookkeeping while planning one order at one station: pending and removed moves, pending copy commitments, the local pickface intervals, temporarily extended bins, new bins, shared picks and pick times. `apply_shared_pick` and `add_new_bin` are the two ways an order line gets served. |
| `find_shared_bin(cands, k, t_cursor, state, s, sc)` | Best existing or tentative visit of SKU k that the order can pick from. The order may wait at most `rt[k]`. It either fits inside the presence window (respecting PickCap via `find_pick_slot`) or extends the window, if the picker is ready by its end and the extended presence still respects pickface, copy, MoveCap and horizon limits (`_can_extend_bin_event`). |
| `earliest_feasible_fetch(k, t, state, sc)` | Earliest fetch start given copy availability (including tentative commitments) and MoveCap |
| `commit_plan(plan, state)` | Applies an `OrderPlan`: new bins, extended shared bins, lanes, pools, MoveCap profile |
| `record_plan` | Copies a committed plan into `order_assignments` and `pick_events` |

The **SGC planner** (`sgc.plan_order_at_station`) and the **RDI planner** (`rdi.plan_order_gap_filling`)
both build on these pieces and differ only in SKU sequencing, bin placement and score (see below).

---

## Algorithms

### SGC: Sequential Greedy Construction (`run_sgc`)
1. Sort orders by `order_rule`. The default is `Rule("sharing_degree", False)`, i.e. orders sharing SKUs with the fewest other orders go first.
2. For each order, plan it at every candidate station and commit the plan with the lowest
   `alpha*end + beta*(end - start)`.
3. Planner: take the earliest-free lane and visit SKUs in `sku_rule` order (default
   `Rule("demand", True)`). Each SKU is served from a shared bin if possible; otherwise a new copy is fetched so that
   its presence starts once both the picker and the pickface are free (delayed by copy availability and MoveCap).
4. `load_balanced=True` only considers stations with picking workload ≤ the average, falling back to the others if
   none of them yields a plan.

5. `piggyback_singles=True` (single-item piggybacking; works with or without `load_balanced`):
   - When order *o* is planned (at any station), the SKU of *o* requested by the most **unscheduled 1-item orders**
     is moved to the end of its SKU sequence. Ties go to the SKU already latest in the `sku_rule` order. If no SKU of *o*
     has such orders, nothing changes. Station choice then proceeds as normal.
   - After *o* is committed, those 1-item orders (in `order_rule` order) are scheduled one after another on the
     **lane *o* just vacated**. Each picks from *o*'s **last bin visit**, whose presence is extended pick by pick
     (`plan_single_from_bin`, using `find_shared_bin` restricted to that visit, so pickface, copy, MoveCap, PickCap and
     horizon limits all hold).
   - The chain stops at the first single that cannot draw from the visit. Remaining singles are scheduled normally
     later. SGC then continues in `order_rule` order, skipping orders already scheduled.
   - `meta["n_piggybacked"]` counts the piggybacked orders.
   - It only helps when multi-item orders are processed *before* the singles. With the default rule
     (sharing degree ascending), singles often come first and become anchors themselves, which still chains the other
     singles with the same SKU.

Parameters: `horizon, alpha, beta, order_rule, sku_rule, load_balanced, piggyback_singles, exo`.

**Dispatching rules.** `Rule(attr, descending)` sorts by `Instance.order_attrs[o][attr]` or `Instance.sku_attrs[k][attr]`.
- Order attributes: `sum_rt, order_size, sum_cycle, max_rt, sku_rarity, sku_contention, sharing_degree, min_copies`.
- SKU attributes: `rt, p, cycle, demand_ratio, copies, demand`.
- `ALL_RULES` holds the 192 combinations, in the column order of the AMA dataset.

### AMA: Adaptive Multi-Attribute SGC (`run_ama`)
Runs SGC for several `(order_rule, sku_rule)` pairs and returns the best by `objective(alpha, beta)`.
- `mode="full_grid"` runs all 192 pairs.
- `mode="two_phase"` runs 16 order rules with the SKU rule fixed to demand desc, then the 12 SKU rules for the best order rule.
- `meta["best_rule"]` holds the winner and `meta["runs"]` holds one record per SGC run.
- `load_balanced` and `piggyback_singles` are passed to every SGC run.

### RDI: Regret-based Dynamic Insertion (`run_rdi`)
Keeps an insertion plan of every unscheduled order at every station. It repeatedly commits the order with the largest key

```
(regret_k - 0.35 * best_score,  #shared picks,  sharing_degree,  tiebreak,  -best_score)
```

where `regret_k` is the sum of the score gaps of the k−1 next-best stations.
- `tiebreaker` is one of `sharing_degree`, `sum_rt_asc` (−Σrt) or `best_score`.
- After a commit, plans are refreshed only where they can change (`use_lazy=True`):
  - orders sharing an SKU with the committed order are re-planned at all stations;
  - every other order is re-planned at the committed station, plus any station whose cached plan overlaps the committed robot moves (MoveCap).

Gap-filling planner (`plan_order_gap_filling`):
- It repeatedly serves the remaining SKU that can be picked **earliest from a shared bin**. Otherwise it fetches the highest-demand
  remaining SKU into the **earliest pickface gap** (it does not only append after the last presence).
- Score: `alpha*end + beta*(end-start) - 0.5 * Σ_shared (rt+p)`.

Parameters: `horizon, alpha, beta, regret_k, use_lazy, tiebreaker, exo`.

### Partitioned RDI, the "decomposition RDI" (`run_partitioned_rdi`)
1. Assign orders to stations. `partition` is one of:
   - `"random"` (global RNG; `seed` sets it)
   - `"similarity"`
   - `"load_balanced"` (LPT on picking workload)
   - an explicit `{station_index: [orders]}`, e.g. from the RL policy
2. Split MoveCap with `distribute_movecap`; the remainder goes to the stations with the most orders.
3. Run RDI independently on each single-station subinstance (horizon 50000) and stitch the results.

`meta`: `station_feasible`, `assignment` (station → orders), `partition_time`.
This is the heuristic-only counterpart of `decomposition.solve_station_decomposition`, which solves each part by CP.

### CFSS: Cluster-First Schedule-Second (`run_cfss`)
1. Weighted Jaccard similarity between orders (`normalize_similarity` selects the normalised variant).
2. Agglomerative UPGMA clustering. Clusters merge while their similarity is ≥ `sim_threshold` and the merged cluster has ≤ `max_cluster_orders` orders.
3. Assign clusters to the station with the least rt mass. Multi-order clusters go first, then singletons, each group by mass descending.
4. Interleave: repeatedly take the station whose earliest lane frees first and commit its pending order with the best SGC plan.

Parameters: `horizon, alpha, beta, sim_threshold, max_cluster_orders, normalize_similarity`.

### GBS: Global Bin Scheduling (`run_gbs`)
A bin-centred heuristic.
1. Partition orders over stations (SKU-overlap gain minus load).
2. Per station, seed the lanes with overlapping orders.
3. Repeatedly schedule the best "bin task": one visit of SKU k serving every active order that needs it.

Rules:
- `max_sharing`: n_sharing·(rt+rt_ret)
- `critical_path`: the largest remaining work among the orders served
- `readiness_weighted`: max_sharing − pickface idle gap
- `adaptive`: the best feasible result of the three, by `alpha*C_max + beta*flow`

Parameters: `horizon, rule, alpha, beta`. **Ignores PickCap; see KNOWN_ISSUES.**

---

## Adding a heuristic

1. Create `heuristics/my_heuristic.py` with
   ```python
   def run_my_heuristic(inst: Instance, *, horizon: int = 10000, my_param: float = 1.0) -> HeuristicSolution:
       state = init_state(inst, horizon)                    # if you build on the SGC machinery
       ...
       plan = ...                                          # an OrderPlan (e.g. via sgc.plan_order_at_station)
       commit_plan(plan, state); record_plan(plan, order_assignments, pick_events)
       ...
       return HeuristicSolution(order_assignments, dict(state.station_bin_events), makespan,
                                total_moves, feasible, pick_events, meta={"heuristic": "my_heuristic"})
   ```
   Read problem data from `inst` (`inst.rt`, `inst.movecap`, `inst.demand_count`, `inst.order_attrs`, …).
   If you need a new per-instance precomputation, add it as a `cached_property` on `Instance` and list it in `Instance._CACHED`.
2. Register it in `heuristics/__init__.py`: add `"my_heuristic": (run_my_heuristic, {defaults})` to `HEURISTICS`, and export it.
3. It is then available in `python -m heuristics`, `run_heuristic`, `precalculate_instances_for_lns.py --heuristic`
   and `benchmark_models.py --methods my_heuristic`.
4. Add a case to `tests/test_heuristics.py`. The registry smoke test parametrises over `HEURISTICS` automatically.

---

## Verifying behaviour: the equivalence harness

`tests/regression/equivalence.py` checks the refactored code against a reference commit (default `f391800`, the last
commit before the refactor) **exactly**. It runs every case with both versions and compares:
- feasibility, makespan, moves;
- all order assignments, bin events (in list order, including copy ids and pick intervals) and pick events;
- `validate_solution` and `validate_warmstart` output;
- the docplex warm-start dictionary;
- benchmark metrics.

```bash
.venv/bin/python tests/regression/equivalence.py              # all cases (~3 min with 8 workers)
.venv/bin/python tests/regression/equivalence.py --filter pkl- --jobs 1   # subset, sequential timing
```

How it runs:
- The old side runs in a temporary `git worktree` of the reference commit; its results are cached in `tests/regression/.cache/`.
- The case grid lives in `tests/regression/cases.py`: 108 generated instances (stations × lanes × orders × MoveCap × {normal, tight} horizon), PickCap = 2 variants and 3 real precalculated instances. Together that is 2259 cases covering every heuristic and parameter variant.
- AMA has no old reference because it crashed at the reference commit. It is covered by `test_ama_is_best_of_sgc_runs` and by the default-rule SGC equality.

When you **intentionally** change heuristic behaviour (e.g. fix a known issue), the harness will report the affected
cases. Commit the change and use that commit as the new `--ref` baseline for later refactors.
