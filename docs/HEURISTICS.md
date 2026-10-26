# Heuristics (`heuristics/` package)

Every heuristic is a plain function with the same contract:

```python
run_x(inst: Instance, *, horizon: int = ..., **algorithm_params) -> HeuristicSolution
```

- MoveCap and PickCap are always read from the `Instance`, never passed separately. Both are mandatory integers
  (≥ 1); for an effectively uncapped instance, pass a large value.
- The default horizon is 10000 everywhere.
- No `*args` and no instance generation inside: the caller builds or loads the `Instance`.
- Parameters are keyword-only, explicit, and have defaults.

```python
from heuristics import run_heuristic, run_rdi, HEURISTICS
sol = run_heuristic("lb_sgc", inst, horizon=10000)          # by registry name
sol = run_rdi(inst, tiebreaker="sharing_degree")          # or directly
violations = sol.validate(inst, horizon=10000)
sp = sol.to_warmstart(mdl, handles)          # docplex model from cp_model.build_model (strict)
sol.to_cpp_warmstart(mdl, handles)           # C++ model from cpp_cp_model.build_model (strict)
cp_as_heur = HeuristicSolution.from_cp(cp_result, handles)   # CP solution -> HeuristicSolution
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

## Registry

| Name | Call |
|---|---|
| `sgc` | `run_sgc()` |
| `lb_sgc` | `run_sgc(load_balanced=True)` |
| `sgc_singles` / `lb_sgc_singles` | `run_sgc(piggyback_singles=True)` / `run_sgc(load_balanced=True, piggyback_singles=True)` |
| `sgc_share` / `lb_sgc_share` | `run_sgc(extend_share_window=True, share_reward=1.0[, load_balanced=True])` |
| `sgc_share_window` | `run_sgc(extend_share_window=True)` (ablation) |
| `ama` / `ama_two_phase` | `run_ama(mode="full_grid" / "two_phase")` |
| `ama_singles` / `ama_two_phase_singles` / `lb_ama_two_phase_singles` | `run_ama(..., piggyback_singles=True[, load_balanced=True])` |
| `ama_two_phase_share` | `run_ama(mode="two_phase", extend_share_window=True, share_reward=1.0)` |
| `rdi` / `rdi_sharing_degree` / `rdi_best_score` | `run_rdi(tiebreaker="sum_rt_asc" / "sharing_degree" / "best_score")` |
| `rdi_share` | `run_rdi(extend_share_window=True, share_reward=1.0, legacy_share_bonus=0.0)` |
| `rdi_share_window` / `rdi_share_reward` | ablations: window only (legacy bonus kept) / `share_reward=1.0, legacy_share_bonus=0.0` |
| `partitioned_rdi_{random,similarity,load_balanced}` | `run_partitioned_rdi(partition=...)` (seed 0) |
| `partitioned_rdi_load_balanced_share` | as above with the `rdi_share` options |
| `cfss` / `cfss_normalized` | `run_cfss(normalize_similarity=False / True)` |

Aliases such as `greedy`, `load_balanced_greedy` and `rdi_sgc` are accepted by `run_heuristic`
and by `precalculate_instances_for_lns.py`.

The registry replaces the old per-variant modules. For example, `heuristic_rdi_sgc_best_score.py`
became `rdi_best_score` and `heuristic_cfss_sgc_normalized.py` became `cfss_normalized`. GBS was removed.

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
- `validate_warmstart(handles)`: checks consistency with the CP model: indexing, copies, MoveCap, PickCap and horizon.
- `to_warmstart(mdl, handles)` and `to_cpp_warmstart(mdl, handles)` are **strict by default**. They raise
  `ValueError` if the solution is flagged infeasible or `validate_warmstart` reports any hard violation (everything
  except the "Symmetry" diagnostics, which `inject_warmstart` fixes by remapping lanes). An infeasible heuristic
  solution therefore can never silently become a CP warm start. Pass `strict=False` to override.
- `HeuristicSolution.from_cp(cp_result, handles)`: converts a docplex / C++ / CP-SAT-style CP solution on the
  `build_model` variables. It returns `None` if there is no solution. It is used by the LNS final validation and the decomposition.
- `to_viz`.
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
| `init_state(inst, horizon, exo=None)` | Initial state. `ExoState(lanes, blocks, moves, batch_start_time)` pre-occupies resources; it is used by decomposition. |
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

Parameters: `horizon, alpha, beta, order_rule, sku_rule, load_balanced, piggyback_singles,
extend_share_window, share_reward, exo`. The two sharing options are described under
[Bin-sharing options](#bin-sharing-options-sgc-and-rdi).

**Dispatching rules.** `Rule(attr, descending)` sorts by `Instance.order_attrs[o][attr]` or `Instance.sku_attrs[k][attr]`.
- Order attributes: `sum_rt, order_size, sum_cycle, max_rt, sku_rarity, sku_contention, sharing_degree, min_copies`.
- SKU attributes: `rt, p, cycle, demand_ratio, copies, demand`.
- `ALL_RULES` holds the 192 combinations, in the column order of the AMA dataset.

### AMA: Adaptive Multi-Attribute SGC (`run_ama`)
Runs SGC for several `(order_rule, sku_rule)` pairs and returns the best by `objective(alpha, beta)`.
- `mode="full_grid"` runs all 192 pairs.
- `mode="two_phase"` runs 16 order rules with the SKU rule fixed to demand desc, then the 12 SKU rules for the best order rule.
- `meta["best_rule"]` holds the winner and `meta["runs"]` holds one record per SGC run.
- `load_balanced`, `piggyback_singles`, `extend_share_window` and `share_reward` are passed to every SGC run.

### RDI: Regret-based Dynamic Insertion (`run_rdi`)
Keeps an insertion plan of every unscheduled order at every station. It repeatedly commits the order with the largest key

```
(regret_k - 0.35 * best_score,  #shared picks,  sharing_degree,  tiebreak,  -best_score)
```

where `regret_k` is the sum of the score gaps of the k−1 next-best stations.
- `tiebreaker` is one of `sum_rt_asc` (−Σrt, **default**), `sharing_degree` or `best_score`. The last two always give
  identical results (see KNOWN_ISSUES).
- Exact ties go to the first candidate in order-id order. Unscheduled orders are kept in an insertion-ordered dict, so
  results never depend on hash-table order.
- After a commit, plans are refreshed only where they can change (`use_lazy=True`):
  - orders sharing an SKU with the committed order are re-planned at all stations;
  - every other order is re-planned at the committed station, plus any station whose cached plan overlaps the committed robot moves (MoveCap).

Gap-filling planner (`plan_order_gap_filling`):
- It repeatedly serves the remaining SKU that can be picked **earliest from a shared bin**. Otherwise it fetches the highest-demand
  remaining SKU into the **earliest pickface gap** (it does not only append after the last presence).
- Score: `alpha*end + beta*(end-start) - 0.5 * Σ_shared (rt+p)`.

Parameters: `horizon, alpha, beta, regret_k, use_lazy, tiebreaker, extend_share_window, share_reward,
legacy_share_bonus (default 0.5), exo`. `run_partitioned_rdi` passes all RDI parameters through.

### Bin-sharing options (SGC and RDI)

With PickCap = 1, orders can share a bin visit only *sequentially*. The second order picks after the first, and the
bin's presence is extended by one pick. Two options make the planners share more. Both are off by default, which keeps
the original behaviour; the options live in `ShareOptions` on the `HeuristicState`.

- `extend_share_window` (bool). By default a shared pick may start at most `rt[k]` after the order's picker is
  ready, assuming a fresh bin would arrive within `rt[k]`. That assumption fails when robots are the bottleneck. With
  the option on, the limit becomes the start of the order's own best new-fetch pick for that SKU, computed by the
  side-effect-free `_new_fetch_sgc` / `_new_fetch_gap` helpers and never earlier than the default. In words: wait for
  an existing bin as long as that is no later than fetching a fresh one.
- `share_reward` = λ. The plan score (lower is better) becomes `alpha*end + beta*span − λ·Σ_shared (rt[k] + rt_ret[k])`,
  i.e. each shared pick is credited with the robot time it saves. For a fixed order this equals adding λ·(rt + rt_ret)
  per *new* fetch, up to a per-order constant. RDI additionally keeps its older bonus
  `legacy_share_bonus·Σ_shared (rt[k] + p[k])`, default 0.5 (added in `ff676ba`); the `*_share` variants set it to 0.

On 20 benchmark instances (small scale, robot-bound), compared with each heuristic's default:

| | makespan | pick/bin |
|---|---|---|
| SGC + wider window | −2.5% | 0.024 → 0.074 |
| SGC + wider window + λ = 1 (`sgc_share`) | −2.9% | → 0.084 |
| LB-SGC + wider window + λ = 1 (`lb_sgc_share`) | −1.8% | 0.019 → 0.061 |
| RDI + wider window (legacy bonus kept) | −2.1% | 0.049 → 0.109 |
| RDI + λ = 1 instead of the legacy bonus | −0.6% | → 0.069 |
| RDI + wider window + λ = 1 (`rdi_share`) | −2.5% | → 0.133 |

λ between 0.5 and 2 performs about the same; most of the gain comes from the wider window. See also KNOWN_ISSUES,
"Sharing under PickCap = 1".

### Partitioned RDI, the "decomposition RDI" (`run_partitioned_rdi`)
1. Assign orders to stations. `partition` is one of:
   - `"random"` (local `random.Random(seed)`, default `seed=0`)
   - `"similarity"` (random seed order, same RNG)
   - `"load_balanced"` (LPT on picking workload)
   - an explicit `{station_index: [orders]}`, e.g. from the RL policy
   Named methods only use the first `min(|S|, MoveCap)` stations (`active_station_count`), so every used station gets a robot.
2. Split MoveCap with `distribute_movecap`; the remainder goes to the stations with the most orders. With an explicit
   assignment, a station left with MoveCap 0 is not solved: a warning is issued, its orders stay unscheduled and the station is infeasible.
3. Run RDI independently on each single-station subinstance and stitch the results.
4. Check the cross-station copy limits `N[k]`. If any is exceeded, warn and set `feasible = False`.

`meta`: `station_feasible`, `assignment` (station → orders), `partition_time`, `copy_conflicts`.
This is the heuristic-only counterpart of `decomposition.solve_station_decomposition`, which solves each part by CP.

### CFSS: Cluster-First Schedule-Second (`run_cfss`)
1. Weighted Jaccard similarity between orders (`normalize_similarity` selects the normalised variant).
2. Agglomerative UPGMA clustering. Clusters merge while their similarity is ≥ `sim_threshold` and the merged cluster has ≤ `max_cluster_orders` orders.
3. Assign clusters to the station with the least rt mass. Multi-order clusters go first, then singletons, each group by mass descending.
4. Interleave: repeatedly take the station whose earliest lane frees first and commit its pending order with the best SGC plan.

Parameters: `horizon, alpha, beta (default 1.0), sim_threshold (0.1), max_cluster_orders (4), normalize_similarity`.
Plans use the SGC planner, so MoveCap and PickCap hold. Clusters and pending orders are visited in a fixed order, so the
result is deterministic.

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

`tests/regression/equivalence.py` checks the working tree against a reference commit **exactly**. The default
reference is `HEAD`; `f391800` is the last commit before the refactor. It runs every case with both versions and compares:
- feasibility, makespan, moves;
- all order assignments, bin events (in list order, including copy ids and pick intervals) and pick events;
- `validate_solution` and `validate_warmstart` output;
- the docplex warm-start dictionary;
- benchmark metrics.

```bash
.venv/bin/python tests/regression/equivalence.py              # working tree vs. HEAD (~3 min with 8 workers)
.venv/bin/python tests/regression/equivalence.py --ref f391800   # vs. the pre-refactor code (legacy adapter)
.venv/bin/python tests/regression/equivalence.py --filter pkl- --jobs 1   # subset, sequential timing
```

How it runs:
- The old side runs in a temporary `git worktree` of the reference commit; its results are cached in `tests/regression/.cache/`.
  For references that already contain `heuristics/`, both sides use the same adapter. For older references
  (≤ `f391800`), a legacy adapter calls the removed modules.
- The case grid lives in `tests/regression/cases.py`: 108 generated instances (stations × lanes × orders × MoveCap {3, 20, 1000} × {normal, tight} horizon), PickCap = 2 variants and 3 real precalculated instances. Together that is 1899 cases covering every heuristic and parameter variant.
- Cases pass their parameters explicitly, so a change of *defaults* only shows up in cases that rely on defaults
  (`cfss`, `rdi_beta`, `rdi_exo`, partitioned and benchmark cases).
- AMA has no old reference because it crashed at the reference commit. It is covered by `test_ama_is_best_of_sgc_runs` and by the default-rule SGC equality.

When you **intentionally** change heuristic behaviour (e.g. fix a known issue), the harness will report the affected
cases. Commit the change and use that commit as the new `--ref` baseline for later refactors.
