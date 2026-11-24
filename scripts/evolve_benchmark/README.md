# Evolve benchmark (GBS sequence evolution)

Algorithm: `docs/HEURISTICS.md`, "GBS sequence evolution". Run everything from the repo root with the repo's Python.

| Step | Command | Cost |
|---|---|---|
| 1. Search benchmark | `python scripts/evolve_benchmark/run.py --out results/evolve_benchmark` | 150 jobs (15 instances x 5 arms x 2 seeds x 1500 evaluations), ~45 min on 10 cores |
| 1b. Long-budget check | `python scripts/evolve_benchmark/run.py --out results/evolve_benchmark_long --arms ls_work --seeds 1 --evals 10000 --instances <5- and 10-station GEN names>` | 10 jobs, ~16 min on 10 cores |
| 2. Analysis | `python scripts/evolve_benchmark/analyze.py results/evolve_benchmark` | seconds; writes `analysis/summary.md`, `convergence.png` |
| 3. LNS roots | `python scripts/evolve_benchmark/prepare_lns.py --out results/evolve_lns --start gbs_evolve_work=results/evolve_benchmark_long:ls_work:0 --start gbs_evolve_makespan=results/evolve_benchmark:ga_makespan:0` | seconds; decodes the stored best sequences and checks C and W against the recorded run |
| 4. LNS validation (**not run yet**) | `OUT=<path>/evolve_lns SRC=<frozen LNS code> scripts/warmstart_benchmark/run_all.sh` | ~4 h per start at P=3 (`RUNS=2` halves it); the two roots are 5/10-station instances, as in the warm-start benchmark |
| 5. LNS analysis | copy `results/warmstart_benchmark_copied/{lb_sgc,lb_sgc_share,gbs_max_sharing,gbs_cluster_lpt}` and their `warm_starts.csv` rows next to the new roots, then `python scripts/warmstart_benchmark/analyze.py <root>` | |

`run.py` appends to `runs.jsonl` and skips finished (instance, arm, seed) jobs, so an interrupted run can be restarted.
`results/` is git-ignored: the numbers that matter are copied into the docs.

Arms (`run.py: ARMS`, all `window=3`): `ls_makespan`, `ga_makespan`, `ls_work`, `ga_work`, `ga_makespan_bias`.
`prepare_lns.py` cannot use `evolve_bias` arms because the bias genome is not stored in `runs.jsonl`.

**What step 4 should show.** LNS keeps the start's robot work W and ends at C ~ 1.04-1.08 W/MoveCap (see
`warmstart_benchmark`). So `gbs_evolve_work` (W -2.3% vs `gbs_max_sharing`, -1.8% vs the best seed) should end ~2% below GBS, while
`gbs_evolve_makespan` (C -6.8%, W +0.4% vs `gbs_max_sharing` on these 10 instances) should end no better than GBS. If both
beat GBS equally, or the makespan start wins, the "work carries over" explanation is wrong.
