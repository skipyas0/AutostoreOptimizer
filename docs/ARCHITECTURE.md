# Project architecture

Joint **order and bin scheduling** in robotic compact storage / retrieval systems
(RCS/RS, AutoStore-like). The goal is to minimise makespan. The approach is:
- constructive heuristics,
- a monolithic CP model (IBM CP Optimizer),
- a CP-based LNS ("fix-and-optimize") matheuristic,
- decomposition schemes,
- ML-augmented order assignment.

This file maps the top-level code. Heuristic internals are in
[HEURISTICS.md](HEURISTICS.md). Known bugs and quirks are in [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

---

## 1. The problem in one screen

| Symbol | Meaning | `Instance` attribute |
|---|---|---|
| `S` | picking stations | `inst.S` |
| `L` | lanes (order slots) per station; one order per lane at a time | `inst.L` |
| `K` | SKUs, each stored in `N[k]` physical bins | `inst.K`, `inst.N` |
| `O` | orders, each a list of SKUs | `inst.O`, `inst.orders_req` |
| `rt[k]`, `rt_ret[k]` | robot fetch / return time of a bin of SKU k | `inst.rt`, `inst.rt_ret` |
| `p[k]` | pick time per order line | `inst.p` |
| MoveCap | max simultaneous robot moves (fetch + return), global | `inst.movecap` (required int ≥ 1; use a large value for "uncapped") |
| PickCap | max orders picking from one presented bin simultaneously | `inst.pickcap` (int ≥ 1, default 1) |

A **bin visit** at station s consists of three intervals:
- fetch `[fetch_start, fetch_end)`, length `rt[k]`;
- presence at the pickface `[presence_start, presence_end)`;
- return `[return_start, return_end)`, length `rt_ret[k]`.

Constraints:
- at most one bin is present at a station's pickface at a time;
- copies of SKU k in use are ≤ `N[k]` at all times;
- concurrent moves are ≤ MoveCap;
- concurrent picks on one bin are ≤ PickCap;
- each order occupies one lane from its first to its last pick.

Several orders at the same station can share one bin visit. That sharing is the main lever for reducing robot moves and makespan.

---

## 2. Pipeline

```
 datagen.generate_data ─► Instance ─► heuristics.run_heuristic(name, inst) ─► HeuristicSolution
        (instance.py)        │                                                   │
                             │                         .to_warmstart / .to_cpp_warmstart
                             ▼                                                   ▼
                cp_model.build_model / cpp_cp_model.build_model ──► CP model + warm start
                             │
              ┌──────────────┼─────────────────────────────┐
              ▼              ▼                             ▼
      monolithic CP     matheuristic.Solver (LNS)     decomposition.solve_*  (CP per sub-problem,
                        neighborhood_selection          RDI warm starts, stitched)
                        freezing_utils
                        matheuristic_plots (logging/plots)
```

LNS experiments start from a **precalculated instance folder**, written by
`precalculate_instances_for_lns.py`:

```
precalculated_instances/GEN<gen_seed>-<S>-<L>-<SKUs>-<orders>-<movecap>[-<heuristic>]/
  config.json                 generation + heuristic + CP config
  instance.pkl                pickled Instance
  instance_summary.txt
  heuristic_solution.pkl      pickled heuristics.HeuristicSolution (warm start)
  heuristic_summary.txt, heuristic_solution.html
  cp_solution_<seed>.pkl, cp_sol_<seed>.txt, cplog_<seed>.txt, cp_intermediate_records_<seed>.pkl
  experiments/                LNS runs (matheuristic.py)
```

`matheuristic.load_instance(folder)` loads `(instance, heur_sol, folder, config)`.

---

## 3. Module map (top level)

### Problem data
| Module | Purpose | Key entry points |
|---|---|---|
| `constants.py` | Reference config and parameter levels, used to normalise ML features | `REFERENCE_CONFIG`, `PARAM_LEVELS`, `PARAM_ORDER` |
| `instance.py` | `Instance`: problem data plus statistics (`stats`), normalised ML `features`, the order–SKU graph for GNNs, and **lazily cached heuristic attributes** (`demand_count`, `req_sets`, `sharing_degree`, `sku_order_index`, `order_attrs`, `sku_attrs`). `subinstance(orders, stations, movecap)` restricts an instance | `Instance`, `.subinstance`, `.to_pickle/.from_pickle`, `.create_order_graph` |
| `datagen.py` | Synthetic instances: Zipf SKU popularity, log-normal order sizes, depth-based retrieval times, popularity-correlated bin counts | `generate_data(...)` |
| `jaccard_similarity.py` | Weighted / unweighted Jaccard similarity between orders | `build_similarity_matrix` |

### Heuristics: `heuristics/` package
See [HEURISTICS.md](HEURISTICS.md).
- `run_heuristic(name, inst, **params) -> HeuristicSolution`
- registry `HEURISTICS`
- CLI: `python -m heuristics --list`
- Algorithms: SGC (± load balancing, ± single-item piggybacking), AMA, RDI, partitioned RDI, CFSS.

### Exact model
| Module | Purpose |
|---|---|
| `cp_model.py` | docplex CP Optimizer model. `build_model(inst, add_symmetry_breaking, horizon, exo_*, batch_start_time, objective_func)` returns `(mdl, handles)`, where `handles` are the interval-variable dictionaries `I_os, I_os_lane, F, B, R, P, C, Block, U, …`. `inject_warmstart(sol, pick_events, mdl, handles)` turns a HeuristicSolution into a starting point. |
| `cp_lns_core.cpp` + `setup.py` | The same model in C++ (CP Optimizer C++ API, pybind11). It avoids docplex overhead in the LNS loop. Build with `python setup.py mac` (macOS) or `python setup.py ubuntu` (Linux) to get the untracked `cp_lns_core*.so`. |
| `cpp_cp_model.py` | Python wrapper with the same `build_model` / `inject_warmstart` interface as `cp_model.py`, plus `export_model` |
| `cp_model_utils.py` | `CppSolveResult`, solution diffs (`build_solve_dict`, `Diff`), utilisation time series, `.cpo` canonicalisation and comparison |
| `verify_cp_models.py` | Checks that the docplex and C++ models export identical `.cpo`. The shift-length/wages section exists only in docplex and always shows as a difference. Fixed intervals (frozen LNS data) are compared by their bounds |

### LNS matheuristic ("CP-based LNS", fix-and-optimize)
| Module | Purpose |
|---|---|
| `matheuristic.py` | `Solver`: the LNS loop. Each iteration selects a neighbourhood, freezes the rest of the incumbent, re-solves with CP from a warm start, and applies acceptance, severity adaptation and objective switching. CLI: `python matheuristic.py <instance folder> --backend cpp …` |
| `neighborhood_selection.py` | Selection strategies (random orders/SKUs, time slices, lane and MoveCap balancing, similar orders, refetched SKUs, history-based) and `StrategyManager` (adaptive weights) |
| `freezing_utils.py` | Freezing policies: which variables of the incumbent are fixed, plus partial starting points |
| `subproblem.py` | Subproblem mode (`--lns-model subproblem`). Instead of freezing variables of the full model, each iteration builds a reduced model with only the active orders. Frozen orders enter as fixed intervals (`exo_lanes`, `exo_station_bins`, `exo_blocks`, `exo_moves`) and objective constants (`ObjectiveOffsets`), and the result is merged back into the full solution. The full model is never built, so extraction cost scales with the neighbourhood, not the instance. Works with both backends. `--validate-merges` checks every merged solution. Open issue: shared bin visits (KNOWN_ISSUES.md) |
| `matheuristic_plots.py` | `VisualLogger`: per-iteration logging, solution-delta "barcodes", KPI and convergence plots |
| `precalculate_instances_for_lns.py` | Builds precalculated instance folders. It generates the instance, runs the warm-start heuristic (`--heuristic`, any registry name or `load_balanced_rdi` / `random_rdi` / `similarity_rdi` for the CP-based decomposition) and runs monolithic CP baselines per `--cp-seeds` |
| `analyze_lns_experiments.py` | Aggregates LNS experiment folders into tables (LaTeX/PNG) |

### Decomposition
| Module | Purpose |
|---|---|
| `decomposition.py` | CP-based **spatial**, **temporal**, **2D** and **RL** decompositions. Sub-problems are solved by CP, warm-started by RDI with `ExoState` (resources already used by earlier sub-problems), and stitched into one `HeuristicSolution`. CLI: `python decomposition.py --decomposition-type {temporal,spatial,load_balanced,2d_spatial,2d_temporal,rl}`. The heuristic-only variant is `heuristics.run_partitioned_rdi`. |

### ML
| Module | Purpose |
|---|---|
| `trainset_generator.py` | AMA data-mining dataset: instance features plus the objective of all 192 SGC dispatching-rule pairs (`heuristics.sgc.ALL_RULES`) |
| `ama_sgc_dataset_analysis.ipynb` | Analysis of that dataset (see KNOWN_ISSUES: produced with an older planner) |
| `rl_order_assignment.py` | RL order→station assignment: a GNN PPO agent and a permutation-invariant `StationOrderScorer` trained with self-critical REINFORCE. Reward = partitioned RDI makespan (`evaluate_assignment_reward`) |
| `benchmark_models.py` | Standardised benchmark of heuristics and RL checkpoints on the RL training distribution. Every method is `Instance -> HeuristicSolution`, and RL-environment metrics come from `solution_metrics`. `--methods`, `--skip-rl` |

### Visualisation and misc
| Module | Purpose |
|---|---|
| `schedule_visualizer.py` | Plotly Gantt chart of a CP solution or of `HeuristicSolution.to_viz(inst)` |
| `manim_visualizer.py` | Manim animation of the LNS search |
| `test_skew.py` | Sanity check of the datagen Pareto skew |

---

## 4. Import graph (top level)

```
constants ◄── instance ◄── datagen
jaccard_similarity ◄── instance, heuristics.cfss, heuristics.partitioned_rdi
heuristics ◄── cp_model_utils, matheuristic, decomposition, precalculate_instances_for_lns,
               benchmark_models, rl_order_assignment, trainset_generator
cp_model ◄── cpp_cp_model, matheuristic, decomposition, subproblem (lazy), heuristics.solution (lazy)
subproblem ◄── matheuristic
cp_model_utils ◄── cpp_cp_model, matheuristic, matheuristic_plots, neighborhood_selection
matheuristic ◄── decomposition (load_instance)
neighborhood_selection ◄── matheuristic, heuristics.partitioned_rdi (lazy)
```

`heuristics` itself imports only `instance` (plus `jaccard_similarity`, `cp_model`
and `neighborhood_selection` lazily where they are needed).

---

## 5. Common workflows

```bash
# list / run a heuristic on a generated instance
python -m heuristics --list
python -m heuristics rdi --stations 4 --lanes 4 --orders 160 --skus 20000 --movecap 20 --vis

# precalculate an LNS instance folder (heuristic warm start + CP baselines)
python precalculate_instances_for_lns.py --stations 4 --lanes 4 --orders 200 --skus 5000 \
       --movecaps 20 --gen-seeds 42 --cp-seeds 42 --heuristic lb_sgc --time-limit 600

# run the LNS matheuristic on it
python matheuristic.py GEN42-4-4-5000-200-20 --backend cpp --runs 5
# ... solving reduced subproblems instead of the frozen full model (much faster on large instances)
python matheuristic.py GEN42-4-4-5000-200-20 --backend cpp --runs 5 --lns-model subproblem

# benchmark heuristics (+ RL checkpoints found as rl_agent_weights*.pth)
python benchmark_models.py --num-instances 10 --methods all --skip-rl

# verify the heuristics still match the reference commit exactly
python tests/regression/equivalence.py              # vs. HEAD; --ref <commit> for another baseline
```

---

## 6. Sub-folders (not part of the maintained code)

| Folder | Content |
|---|---|
| `tests/` | `test_heuristics.py` (new API), `test_instance.py`, other module tests, `regression/` (equivalence harness) |
| `scripts/` | One-off tools. `migrate_heuristic_pickles.py` is current; `run_custom_warmstart*.py` are legacy and import removed modules |
| `analyze_heuristics/`, `benchmarking/` | Legacy analysis and benchmark scripts built on the removed `solve_heuristic_instance` APIs (broken) |
| `learning_ama_archive/` | Archived AMA dataset, plots and decision trees |
| `precalculated_instances/`, `param_exp/`, `decomposition_experiments/`, `results/`, `logs/`, `media/` | Data and experiment outputs (git-ignored) |
| `build/` | C++ extension build artefacts |
