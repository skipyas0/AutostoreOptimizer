# Experiment evaluation suite

One package for turning experiment folders into thesis tables and figures. It
replaces `analyze_experiment1.py`, `analyze_lns_experiments.py` and
`analyze_lns_scaling.py`.

```bash
python -m evaluation compare precalculated_instances/temp_lns --out results/lns_config_comparison
python -m evaluation compare <root> --configs LNS-2 LNS-6 --baselines CP "RDI-LB share"
python -m evaluation compare <root> --pattern 'GEN*-10-*' --times 30 120 900 --png-tables
python -m evaluation scaling results/lns_scaling
```

## `compare`: LNS configurations vs. CP vs. heuristics

Input: a root folder of precalculated instance folders (`GEN{seed}-{stations}-...`), each with

| File | Read as |
|---|---|
| `config.json` | instance size, generator seed, which heuristic is the LNS start |
| `heuristic_summary.txt` | the LNS start heuristic |
| `heuristic_<name>_summary.txt` | every other heuristic (makespan, runtime) |
| `cplog_<k>.txt` | CP run k (incumbents with timestamps) |
| `experiments/<date>/` | LNS experiment: `experiment_config.json`, `experiment_dataframe.pkl`, `run_<k>.log` |

LNS experiments are grouped by their config, not by folder name. Configs that
differ only by keys added later (`CONFIG_DEFAULTS` in `data.py`) are treated as
the same config. Ids `LNS-1..n` follow the date each config first ran, and labels
name only the parameters that differ (`configs.py`). If an instance has the same
config twice, the latest run is used and a warning is printed.

Output (`--out`, default `results/compare_<root>`):

| | Full (`_full`) | Reduced (`_reduced`) |
|---|---|---|
| Methods | all LNS configs, CP, all heuristics | best LNS config + `--baselines` (default CP, RDI) |
| t02_overall | gap at each time point, improvement over start, mean rank, wins | same |
| t03_by_size | the same per size class | same |
| t04_per_instance | mean final makespan per instance | same |
| t05_significance | best LNS vs. each method (Wilcoxon, Holm) | same |
| fig01_convergence | mean gap over time (log time; heuristics as red stars, best LNS final value labelled) | with ± std bands |
| fig02_final_gap | final gap per run (box + dots) | same |
| fig03_mean_rank | mean rank (all + per size) | same |
| fig04_per_instance | heatmap of the mean final gap | same |
| t12_solution_structure | final solutions per size: fetch/pick ratio, fetches and robot work vs. start, refetches, robot/picker utilisation, workload/makespan imbalance, unused stations | same |
| fig11_solution_structure | the same metrics, one marker per size class | same |
| fig12_robot_work_vs_utilisation | robot work vs. robot utilisation, both relative to the start; diagonals = equal makespan change | same |

Other outputs (no full/reduced pair):

| Output | Content |
|---|---|
| t01_lns_configs (+ `_common.txt`) | config ids, differing and shared parameters |
| t06_heuristics, fig05 | heuristics: makespan, gap, runtime |
| t07_lns_iterations, fig06 | iteration statistics per config, size and **objective phase**; time composition |
| t08_strategies_makespan, fig07 | strategies in **makespan-phase** iterations: improves incumbent / new best |
| t09_makespan_improvement_share, fig08 | share of each config's total best-makespan reduction by strategy, split by phase |
| t10_strategies_other_objectives, fig09, fig10 | configs with objective cycling, **non-makespan phases**: rate of improving the phase objective, share of its total improvement, acceptance, makespan cost |
| t11_cp_startup | CP extraction time, first incumbent time, warm-started runs |
| fig13_lns_bin_fetches_over_time | bin fetches of the LNS best solution over time vs. the final CP/heuristic fetch counts |
| data/ | `runs.csv`, `lns_iterations.csv.gz`, `solution_metrics.csv` (one row per final solution; also the cache) |

"Best LNS" is the config with the lowest mean final gap over instances.

Strategy metrics per iteration (phase = the objective the CP optimised, `cp_obj`):

- **Improves phase objective**: some solution the CP found beats the incumbent's
  value of the phase objective (from `discovered_solutions` and the previous
  iteration's `current_<obj>`).
- **New best**: the best makespan dropped. Makespan improvement shares use this
  reduction.
- **Accepted / makespan change**: the incumbent changed, and by how much its makespan
  moved (acceptance is always by makespan, within the RRT threshold).

### Definitions

- **Best known (BKS)**: the best makespan any loaded method found on the instance.
  It is computed before `--configs` filtering, so gaps are comparable between reports
  on the same root.
- **Gap**: (makespan − BKS) / BKS · 100.
- **Improvement**: (start − makespan) / start · 100, where start is the LNS start heuristic.
- **Clocks**: LNS and CP both start at t = 0 holding the precomputed start heuristic.
  For LNS, model setup (`run_<k>.log`) counts as solution time; a solution found
  s seconds into an iteration's CP solve is placed at that iteration's start +
  pre-solve time + s. CP is warm-started from the start heuristic (detected because its
  first incumbent equals it), so model extraction and the wait for the first incumbent
  count as solution time. A CP run without that warm start has no solution before its
  first incumbent. A heuristic is one point at its own runtime.
- At a time point, runs without a solution yet (heuristics before their runtime) are
  excluded from the mean and counted in the table ("k/n w/o sol.").

### Solution structure

Computed from the final solution of every run (`solutions.py`): heuristics
`heuristic_<name>_solution.pkl`, CP `cp_solution_<k>.pkl`, LNS
`experiments/<date>/run<k>-solution.pkl`. A solution whose makespan differs from the
logged final value triggers a warning. Results are cached in `data/solution_metrics.csv`
by file path, mtime and size, so reruns only read new or changed files.

- **Fetch / pick ratio**: bin visits / order lines (1 = no bin serves two lines).
- **Refetches**: visits of a SKU that was already fetched before, in % of visits
  (any station; *same-station*: back to the same station).
- **Robot utilisation**: Σ(rt + rt_ret) over visits / (MoveCap · makespan). Fetches and
  returns have fixed durations and share MoveCap robots, so
  makespan = robot work / (MoveCap · utilisation), which fig12 plots.
- **Picker utilisation**: Σ pick times / (stations · pickcap · makespan).
- **Workload imbalance**: (max − mean) / mean of the pick time per station.
- **Makespan imbalance**: (makespan − earliest station end) / makespan. A station
  without picks ends at 0, so unused stations give 100 % (also counted separately).

## `scaling`: full model vs. subproblem overhead

Reads `lns_iterations.csv`, `lns_runs.csv` and `nbh_sweep.csv` from
`benchmark_lns_scaling.py` and writes `table_scaling`, `table_breakeven` and three
figures (default output `<dir>/analysis`).

## Extending

Every method is loaded into one long *runs* table (`data.py`): `method, family,
instance, size, run, t, v, final`, where `(t, v)` is the anytime trajectory. Metrics
(`metrics.py`) and figures (`plots.py`) only use that table, so:

- **New method** (e.g. a decomposition): write a loader that returns rows in this
  format, append them in `load_root`, and give the method a look in
  `Comparison._methods`. Every table and figure then includes it.
- **New heuristic**: drop `heuristic_<name>_summary.txt` into the instance folders.
  Optionally add a display name to `HEURISTIC_NAMES`.
- **New config parameter**: if it was added after older runs, put its old behaviour in
  `CONFIG_DEFAULTS`. For a short label, add a formatter to `configs.FORMATTERS`.
- **New table or figure**: add a method to `Comparison` (`table_*` writes through
  `_write`, which produces CSV/MD/TeX) and call it from `run`. Plot functions take the
  method list, so iterating over `self.variants()` gives the full and reduced versions.
- **New report** (different data): add `reports/<name>.py` and a subcommand in
  `__main__.py`, reusing `style` and `tables`.

Colors (`style.py`): LNS configs take the validated categorical palette in config order
(the color stays the same in full and reduced plots). CP is near-black and heuristics are
gray dashed reference lines with direct labels.
