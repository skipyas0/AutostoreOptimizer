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
| `__main__.py` | Single CLI for all heuristics (`python -m heuristics benchmark ...` dispatches to `benchmark.py`) |
| `benchmark.py` | Benchmark of many heuristics (and RL checkpoints) over generated or pregenerated instances |
| `heuristic_utils.py` | Shared machinery for the SGC family (below) |
| `solution.py` | `HeuristicSolution`, `validate_solution`, `validate_warmstart`, `build_viz_handles` |
| `sgc.py` | SGC planner and loop, dispatching `Rule`s, `ALL_RULES`, AMA |
| `rdi.py` | RDI (regret insertion with the gap-filling planner) |
| `partitioned_rdi.py` | Order→station partitioning, `distribute_movecap`, partitioned RDI |
| `cfss.py` | Clustering plus interleaved SGC |
| `gbs.py` | GBS: bin-centred construction (one visit serves every active order needing the SKU); `run_gbs_sequence` decoder |
| `evolve.py` | `run_gbs_evolve`: evolutionary search over GBS order sequences |

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
| `rdi_lb` / `rdi_lb_share` | `run_rdi(load_balanced=True)` / with the `rdi_share` options |
| `rdi_share_window` / `rdi_share_reward` | ablations: window only (legacy bonus kept) / `share_reward=1.0, legacy_share_bonus=0.0` |
| `partitioned_rdi_{random,similarity,load_balanced}` | `run_partitioned_rdi(partition=...)` (seed 0) |
| `partitioned_rdi_load_balanced_share` | as above with the `rdi_share` options |
| `cfss` / `cfss_normalized` | `run_cfss(normalize_similarity=False / True)` |
| `cfss_share` | `run_cfss(consecutive=True, balance="work", extend_share_window=True, share_reward=1.0)` |
| `gbs_max_sharing` (alias `gbs`) / `gbs_critical_path` / `gbs_readiness_weighted` / `gbs_adaptive` | `run_gbs(rule=...)` (interleaved stations, strict SKU deferral, overlap partition) |
| `gbs_lpt` / `gbs_cluster_lpt` | `run_gbs(partition="lpt" / "cluster_lpt")` |
| `gbs_regret` / `gbs_regret_lc` | `run_gbs(partition="regret")` / `run_gbs(partition="regret", admit="last_chance")` |
| `gbs_no_defer` / `gbs_sequential` | `run_gbs(defer=None)` / `run_gbs(defer=None, interleave=False)` (original construction) |
| `gbs_evolve` / `gbs_evolve_work` | `run_gbs_evolve(method="ga", fitness="makespan")` / `(method="ls", fitness="work")`, both `window=3, max_evals=300` |
| `gbs_evolve_work_regret` | `gbs_evolve_work` with the regret partition as a fourth seed |

Aliases such as `greedy`, `load_balanced_greedy` and `rdi_sgc` are accepted by `run_heuristic`
and by `precalculate_instances_for_lns.py`.

The registry replaces the old per-variant modules. For example, `heuristic_rdi_sgc_best_score.py`
became `rdi_best_score` and `heuristic_cfss_sgc_normalized.py` became `cfss_normalized`. GBS was removed in
`ed5cadc` (invalid solutions) and reinstated in a corrected form (see GBS under Algorithms).

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
| `HeuristicState` | Committed partial schedule: lane / pickface free times, `DifferenceArray` MoveCap profile (updated in place per move, O(move length)), one `BinCopyPool` per SKU (created on first use), bin events and pickface intervals per station. It also carries `inst`, `horizon` and `pick_cap`, so helpers take only `state`. |
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

**Load balancing** (`load_balanced=True`, mirrors load-balanced SGC). Before each selection, stations whose
committed picking workload (Σ p over committed order lines) is above the current average are treated as
non-existent. Every order's best station, second score and regret are computed over the remaining stations only.
Plans at hidden stations stay cached and are kept up to date, so a station is usable again as soon as it falls back
to the average. If no order fits on the allowed stations, that selection uses all of them; `meta["n_fallbacks"]`
counts how often this happened.

On 10 benchmark instances, load balancing cuts the benchmark's pick-imbalance metric from 166 to 5 (LB-SGC: 8) at
+1.8% makespan, which is still 2.4% better than LB-SGC. Combined with the share options (`rdi_lb_share`), imbalance
is 7 and makespan is +1.4% vs `rdi_share`.

Parameters: `horizon, alpha, beta, regret_k, use_lazy, tiebreaker, extend_share_window, share_reward,
legacy_share_bonus (default 0.5), load_balanced, exo`. `run_partitioned_rdi` passes all RDI parameters through.

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

Parameters: `horizon, alpha, beta (default 1.0), sim_threshold (0.1), max_cluster_orders (4), normalize_similarity,
balance, consecutive, extend_share_window, share_reward`.
- `balance="work"` balances the clusters on picking workload Σ p instead of rt mass (`"rt_mass"`, default).
- `consecutive=True`: the earliest-free station only considers the orders of its current cluster (the first one in
  assignment order with pending orders); if none of them fits, all its pending orders. Cluster mates are then on the
  lanes together and can share visits. Without it, phase 2 takes the best-scored (usually shortest) pending order
  wherever it sits, so clusters are co-located but rarely share (~3 shared visits per instance).
- `extend_share_window`, `share_reward`: the SGC bin-sharing options.

On the 15 LNS benchmark instances, relative to `lb_sgc_share` (main, Nov 2026):

| CFSS variant | makespan | fetches | robot work | workload imb. | shared visits |
|---|---|---|---|---|---|
| `cfss` | 0.0% | +4.0% | +2.3% | 23.7% | 3 |
| + share options | −1.3% | +2.2% | +1.2% | 23.7% | 25 |
| consecutive | +2.2% | +2.7% | +1.5% | 23.7% | 17 |
| consecutive + share | +0.5% | −0.5% | −0.4% | 23.7% | 56 |
| consecutive + share + `balance="work"` (`cfss_share`) | −0.8% | −0.6% | −0.5% | 0.4% | 57 |

The scratch prototype in docs/FUTURE_IDEAS.md had reported −2.0% makespan and −1.9% fetches for "consecutive +
share". That code was lost, and the version here gets about the same number of shared visits but a smaller fetch
saving. CFSS stays order-centred; GBS saves −20% fetches on the same instances.
Plans use the SGC planner, so MoveCap and PickCap hold. Clusters and pending orders are visited in a fixed order, so the
result is deterministic.

### GBS: Global Bin Scheduling (`run_gbs`)
Bin-centred: instead of placing orders and borrowing bins, it schedules bin visits, each serving **every** active order
at the station that needs the SKU.

1. Partition the orders over the stations (`partition`):
   - `"overlap"` (default, original): largest `sum_rt` first, to the station maximising rt-overlap with its SKUs minus its rt load.
   - `"lpt"`: Longest Processing Time on picking workload. Largest `Σ p` first, to the station with the smallest `Σ p` so far.
     Ties go to the station sharing the most SKUs with the order, then the first station.
   - `"cluster_lpt"`: the same LPT over order *clusters* (CFSS clustering: UPGMA on normalised rt-weighted Jaccard,
     threshold 0.1, at most `cluster_max_orders = 4` orders), so cluster mates share a station.
2. Seed every station's lanes with mutually overlapping orders.
3. A *step* at a station schedules the best *bin task* (one per SKU still needed by an active order) that fits the
   horizon: fetch when the pickface frees up (copy, MoveCap), the served orders pick one after another (PickCap at a
   time), return. A finished order frees its lane for the unscheduled order with the largest overlap with the active
   orders' remaining SKUs.
4. SKU deferral (`defer`): a task is *deferred* if unscheduled orders of the same station also need its SKU (a per-station
   counter, updated on admission).
   - `"strict"` (default): deferred tasks come after all other tasks.
   - `"soft"`: their score drops by one visit's robot time `rt + rt_ret`.
   - `None`: no deferral (registry `gbs_no_defer`).
   An order may pick its SKUs in any order, so a deferred SKU tends to become its last pick, and the visit then also serves
   the next order admitted to a freed lane.
5. Which station steps next (`interleave`):
   - `True` (default): the station whose pickface frees first (ties: first station), so visits are committed roughly in
     time order over all stations.
   - `False` (original, registry `gbs_sequential`): each station runs to completion before the next starts.

Partition `"regret"` (`regret_eps`, default 0.05): regret-2 assignment on co-location value. `fit(o, s)` = robot
time `Σ (rt + rt_ret)` of o's SKUs that station s already has (updated incrementally as orders are assigned); s
is eligible while `load_s + Σp(o) <= (1 + ε) · Σp / |S|` (no eligible station: least loaded). The unassigned
order with the largest regret (best eligible fit − second best; ties: larger Σp, then order id) goes to its best
station (ties: least loaded). Overlaps are sparse, so regret ≈ "most co-location value first".

Admission `admit="last_chance"`: a freed lane takes the waiting order with the largest overlap weight plus the
robot time of the active orders' remaining SKUs for which it is the station's *last* waiting order
(`future[k] == 1`). Admitting it lets one visit serve all of them; strict deferral would otherwise keep holding the
SKU back. Default `"overlap"` (overlap weight only).

Rules (task score, higher first): `max_sharing` (n_served·(rt+rt_ret)), `critical_path` (largest remaining
Σ(rt+p+rt_ret) of a served order), `readiness_weighted` (max_sharing − pickface idle gap), `adaptive` (best feasible of
the three by `objective(alpha, beta)`).
Parameters: `horizon, alpha, beta, rule, defer, interleave, partition, cluster_max_orders, exo` (`exo.lanes` is ignored,
as in SGC). `defer=None` reproduces the GBS of `93fa0f9` exactly.

Fixed against the scrapped version: picks respect PickCap (previously all served orders picked at once during one
`p[k]`), and `feasible` is honest (previously pre-filled `(0, 0)` picks made it always `True`).

Why interleaving matters: a `BinCopyPool` copy is only free after its *last* return. Built station by station, a later
station cannot use a copy in an earlier station's idle gaps, so station end times grow with the station index.

Why deferral matters: a visit can only serve orders that are active (on a lane) at the time, at most |L| per station.
If every SKU were fetched once per station, the overlap partition would need 0.68 fetches per order line; without
deferral GBS uses 0.88 (37% of that saving), with strict deferral 0.77 (73%).

On the 15 LNS benchmark instances (β = 1), relative to `lb_sgc_share`:

| | makespan | fetches | robot work | robot util. | workload imb. | makespan imb. | runtime |
|---|---|---|---|---|---|---|---|
| `lb_sgc_share` | 0 | 0 | 0 | 88% | 6.9% | 9.3% | 1.0 s |
| `gbs_sequential` | +49.8% | −9.3% | −5.3% | 57% | 14.9% | 71.5% | 1.2 s |
| `gbs_no_defer` | −1.4% | −9.3% | −5.3% | 85% | 14.9% | 10.9% | 1.3 s |
| `gbs_max_sharing` | −8.8% | −20.4% | −12.4% | 85% | 14.9% | 14.1% | 1.1 s |
| `gbs_critical_path` | −7.6% | −18.1% | −11.0% | 85% | 14.9% | 12.3% | 1.1 s |
| `gbs_readiness_weighted` | −9.4% | −20.4% | −12.4% | 86% | 14.9% | 13.4% | 1.1 s |
| `gbs_adaptive` | −9.3% | −20.1% | −12.2% | 86% | 14.9% | 13.9% | 3.3 s |
| `gbs_lpt` | −6.7% | −18.4% | −10.9% | 85% | 0.2% | 19.8% | 1.1 s |
| `gbs_cluster_lpt` | −8.9% | −20.9% | −12.8% | 85% | 0.2% | 17.7% | 2.6 s |

Regret partition and last-chance admission, relative to `gbs_max_sharing` (same 15 instances; main, Nov 2026):

| | makespan | fetches | robot work | W better on |
|---|---|---|---|---|
| `admit="last_chance"` alone | −0.2% | −0.5% | −0.3% | 14/15 |
| `gbs_regret` | −0.6% | −2.8% | −1.8% | 15/15 |
| `gbs_regret_lc` | +0.9% | −3.3% | −2.1% | 15/15 |

These are the scratch-prototype numbers of docs/FUTURE_IDEAS.md, reproduced exactly (same partition on all 15
instances). LNS keeps the start's W, so `gbs_regret_lc` should give about −2% final LNS makespan; this has not been
validated yet.

Runtimes in this table predate the incremental MoveCap profile (see `DifferenceArray`), which made GBS about 50×
faster (3.0 s → 0.06 s on GEN1-10, bit-identical results); SGC got about 8× faster, RDI is unchanged.

For comparison, `rdi_lb_share` reaches −6.9% makespan and −2.9% robot work at ~100× the runtime. The clustering
(O(n³) in `agglomerative_cluster`) is what makes `gbs_cluster_lpt` slower.

LNS keeps the start's robot work almost unchanged (final makespan ≈ 1.03·W/MoveCap on these instances), so the robot-work
saving is what a GBS warm start should carry over; this has not been validated with LNS runs yet.

### GBS sequence evolution (`run_gbs_evolve`, `evolve.py`)
Searches over the inputs of GBS instead of over schedules; GBS completes every candidate.

- **Genome**: per station, the sequence in which its orders are admitted to the lanes. This fixes
  the partition and the admission order. Optionally (`evolve_bias=True`) there is also a per-SKU bias added to GBS's
  bin-task score, which reorders the bin visits at the stations.
- **Decoder** `run_gbs_sequence(inst, sequence, window=..., sku_bias=...)`: GBS with that partition; a freed lane
  admits, among the first `window` unscheduled orders of the sequence, the one with the largest overlap with the active
  orders (`window=1`: strictly in order). Every GBS run records its admission order in `meta["sequence"]`, and
  decoding it with `window=1` reproduces the run exactly (tested for all rules, partitions and deferral modes).
- **Seeds**: the GBS runs with the `overlap`, `lpt` and `cluster_lpt` partitions (rule `max_sharing`); their
  decodings are not counted in the budget. The result is never worse than the best seed.
- **Fitness**: `"makespan"` = (C, W) or `"work"` = (W, C), lexicographic; W = Σ (rt + rt_ret) over bin visits.
- **Search**: `"ls"`, a (1+1) EA that accepts children that are not worse; `"ga"`, a steady-state GA (binary tournament,
  station-group crossover with probability 0.5, replace-worst, no duplicate fitness values in the population of 24).
- **Operators**: swap of nearby orders, insert within a station, transfer to another station, exchange between
  stations, *pull* (an order right after a random order sharing one of its SKUs), and the SKU bias.
- Budget: `max_evals` decodings and/or `time_limit` seconds. `meta`: `history`, `seed_fitness`, `op_stats`,
  `sequence` (the best genome).

Results on the 15 LNS benchmark instances: see [Evolving GBS sequences](#evolving-gbs-sequences-results) below.

### Evolving GBS sequences: results

`scripts/evolve_benchmark/run.py` (then `analyze.py`): the 15 LNS benchmark instances (GEN1-5 × 5/10/15 stations), 2 seeds,
1500 decodings per run (~3 min with 10 parallel runs), `window=3`, all solutions valid. Changes are relative to the best of
the three GBS seeds *on that metric*, as the mean over instances and seeds; wins count instances (seed mean) and p is a one-sided Wilcoxon test.

| arm | ΔC | ΔW | C / (W/MoveCap) | C wins | W wins |
|---|---|---|---|---|---|
| best seed (GBS) | 0 | 0 | 1.19 | | |
| `ls`, makespan | −3.5% | +0.9% | 1.11 | 15/15 | 0/15 |
| `ga`, makespan | −3.8% (p < 0.001) | +0.9% | 1.11 | 15/15 | 0/15 |
| `ga`, makespan + SKU bias | −3.5% | +1.1% | 1.11 | 15/15 | 0/15 |
| `ls`, work | −1.4% | −0.7% (p < 0.001) | 1.16 | 12/15 | 15/15 |
| `ga`, work | +0.4% | −0.6% | 1.18 | 5/15 | 15/15 |
| `ls`, work, 10 000 evals (5/10 stations) | −3.1% | **−1.8%** | 1.19 | 10/10 | 10/10 |

- The makespan arms converge fast (90% of the gain by 750 evaluations). They gain by packing robot moves more tightly
  (C/(W/MoveCap) 1.19 → 1.11), and they do it with slightly *more* robot work. The gain shrinks with size: −6% at
  5 stations, −3% at 10 and −2% at 15.
- W decreases slowly and steadily: −0.7% after 1500 evaluations, −1.2% after 3000, −1.8% after 10 000, still improving at the end
  (~14 min per instance). It closes 26% of the gap between the best seed's W and the evolved partition's own
  bound (one visit per station and SKU). GBS seeds are 6% above that bound and 19% above one visit per SKU.
- The (1+1) LS is as good as the GA on makespan and better on W. Crossover adds nothing measurable. The SKU-bias
  gene (bin order) does not help: GBS's task order is already good, and the order sequence is what matters.
- Since LNS keeps W and ends at C ≈ 1.04–1.08 · W / MoveCap (warm-start benchmark), the makespan arms should *not*
  carry over to LNS, while the work arm predicts about −1.8% final makespan. This is not yet validated with LNS:
  `scripts/evolve_benchmark/prepare_lns.py` builds the two starts in the warm-start benchmark layout.

**Status and limits.**
- The LNS validation (see `scripts/evolve_benchmark/README.md`, step 4) has **not been run**. Until it has, the
  statement above is a prediction.
- The registry defaults (`max_evals=300`) give only a small gain, about −3% makespan (`gbs_evolve`) and −0.3% W (`gbs_evolve_work`). For a real warm start
  pass a larger budget, e.g. `run_heuristic("gbs_evolve_work", inst, max_evals=10000)` (~14 min on 400 orders), or
  `time_limit=`.
- Only 5- and 10-station instances were run with the 10 000-evaluation budget.

**Complexity.** One decode is O(V·(L·m + S + d) + n·(w·m + n/S) + H): with n orders, S stations, L lanes, m SKUs per order,
V ≈ 0.77·n·m bin visits, d the robot move length and w the window. For a fixed instance shape that is linear in n
(0.05, 0.06 and 0.10 s at 300, 400 and 600 orders). Variation and bookkeeping are O(n) per evaluation. A run costs
O(E · decode) for E evaluations, plus the one-off O(n³) `cluster_lpt` clustering of the seeds (4 s at 600 orders).

---

## Benchmarking

`python -m heuristics benchmark` runs a selection of registry heuristics on a set of instances and
prints two summary tables (`--output-json` keeps the per-instance details):

- **Quality**: makespan C, its mean gap to the best method on each instance, normalized makespan (RL
  reward), robot work W (fetch + return robot-seconds), its gap to the best, number of fetches,
  W relative to one fetch per order line (the sharing gain), validator violations, time.
- **Utilisation, balance and picker gaps**: robot utilisation W / (MoveCap · C); picker utilisation
  (time with at least one pick running / C, mean over stations); station-finish spread
  (C − min C_s) / C and its coefficient of variation; picking-workload imbalance; picks per bin
  presence; and *holeyness* of the picker schedule. That is idle time inside each station's
  [first pick, last pick] span, the share of it in pauses longer than `--gap-threshold`
  (default 60 s), the number of such pauses per station, and the longest pause.

The schedule metrics (`schedule_metrics`) cover feasible stations only. `solution_metrics` keeps the
penalised RL-environment view and is what the equivalence harness fingerprints.

**Selecting methods.** Methods are named by registry key, RL checkpoints by `rl:<stem>`.
`--methods` / `-m` and `--skip` / `-x` take any number of tokens (also comma-separated): `default`
(the 14-method comparison set, `DEFAULT_METHODS`), `all` (whole registry), an exact name or alias,
or a shell glob. Selection = union of `--methods` minus every match of `--skip`, in registry order.
A `--methods` token matching nothing is an error; `--list` prints the selection and exits.

```bash
python -m heuristics benchmark --methods all --skip '*rdi*'        # quote globs
python -m heuristics benchmark -m '*sgc*' -x '*singles*' 'gbs_*'
python -m heuristics benchmark -m 'gbs_*' lb_sgc_share precalculated_instances/
python -m heuristics benchmark --checkpoints rl_agent_weights_mlp_1200.pth -m default 'rl:*' --preset rl-small
```

**Instances.** Positional paths run on pregenerated instances (precalculated folders, `instance.pkl`
files, or a directory of folders; horizon from `config.json`). Otherwise `-n` instances are
generated from `--preset` with seeds `--seed-base + i`:

| Preset | Distribution |
|---|---|
| `tiny` | 2-3 stations, 10-20 orders/station |
| `small` (default) | 2-4 stations, 20-40 orders/station |
| `medium` | 4-8 stations, 30-50 orders/station |
| `large` | 8-15 stations, 40-60 orders/station, 2.5-5 robots/station |
| `lns` | cycles the precalculated LNS shapes 5/10/15 stations × 300/400/600 orders, MoveCap 40; `--seed-base 1` reproduces `GEN1-*` exactly |
| `rl-small`/`rl-medium`/`rl-large` | the old `benchmark_models.py` generator (RL training distribution: 400 orders and MoveCap 40 for any station count) |

Orders and MoveCap scale with the number of stations (`orders = stations × orders_per_station`,
`MoveCap = round(stations × robots_per_station)`, 3-6 robots/station by default), so instances of
one preset have comparable per-station load. Any preset parameter can be overridden with a fixed
value, a range or a choice list: `--stations 10 --orders-per-station 30-50 --movecap 40 --skus 5000,20000`.
`--save-instances DIR` writes the generated instances as precalculated-style folders, which can then be
passed back as positional paths.

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
   and `python -m heuristics benchmark --methods my_heuristic`.
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
