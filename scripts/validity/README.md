# Validity check: does the static model rank schedules the way the simulator does?

The warehouse simulator (`warehouse-sim/`, Rust core `wsim`) is the reference. Each simulator
instance is turned into a static `Instance`, solved with the heuristics (and LNS), and every static
schedule is executed in the simulator as a station plan (`wsim --plan`, see
`warehouse-sim/core/README.md`, *Station plans*). The question is how well static measures
predict the simulated makespan, especially their **ranking** of schedules.

```bash
cd warehouse-sim/core && cargo build --release && cd ../..
.venv/bin/python scripts/validity/prepare.py            # instances, calibration, heuristics, plans, LNS roots (~15 min)
scripts/validity/run_lns.sh results/validity 100 1 4      # LNS, 4 at a time (~35-40 min, ~3 GB each)
.venv/bin/python scripts/validity/simulate.py           # every plan in every execution mode (~2 min)
.venv/bin/python scripts/validity/analyze.py            # tables and figures in results/validity/analysis/
```

## Setup

- Layout `--site 40,40,16,5`: 40 × 40 columns, 16 deep, 80 % fill (about 20 000 bins), 5 stations,
  3 lanes. 300 orders (lognormal size, about 960 lines), 10 000 SKUs (Zipf 1). Seeds 1-10.
- Regimes: **ports** (40 robots: robots mostly idle, ports the bottleneck) and **robots** (12 robots).
- Static instance (`bridge.static_instance`): stations, lanes, orders, SKUs 1:1 (SKUs renumbered
  0..K-1, `sku_map.json`), `p[k]` = the simulator's pick time per line, `N[k]` = bins of the SKU,
  MoveCap = robots, PickCap 1. `rt[k]` = calibrated drive to the stack + dig + lift for the SKU's
  least-covered bin (fitted on the simulator's own run: about 11 s + 11.5 s per bin above),
  plus the drive to the ports (mean over stations) and the lowering. `rt_ret` = the calibration
  run's mean return (about 17 s). Port queueing is deliberately not in `rt`: the static model has
  no port resource.
- Methods: 18 heuristics (SGC, AMA, CFSS, RDI, GBS families), LNS from `gbs_regret_lc`
  (subproblem LNS, the campaign's settings, 100 iterations). Partitioned RDI is infeasible here.
- Execution modes: `full_b3` (the whole plan, 3 bins requested ahead per station), `full_b6`,
  `assign_b3` (only order → station and lane order). References: the simulator's own FCFS and
  sharing assignment.
- Port-aware bound `C_port` = max over stations of (visits × τ + picking time), with τ the port time
  per visit (lowering, lifting, changeover; about 30 s) measured in the calibration run.

## Results

See `results/validity/analysis/` (`t1_rank_correlation`, `t2_selection_regret`, `t3_per_method`,
`t4_vs_native`, `t5_lns_transfer`, `fig1_static_vs_sim`).
