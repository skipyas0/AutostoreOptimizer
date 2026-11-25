# Experiment campaign (Nov–Dec 2026): testing the new heuristics and LNS features

One server: AMD Ryzen 7 7700X (8 cores / 16 threads), 64 GB RAM, NixOS. About one week of compute.
Main LNS model: **subproblem** (graph neighbourhood, anchored picks), **1 h per run**. The full model only
appears in H5 (equal-time comparison).

## What is already established (local results, Nov 2026)

| Claim | Evidence | Open |
|---|---|---|
| GBS (strict deferral) beats SGC/RDI as a heuristic: −12% W, −9% C | 15 eval instances | other regimes, other seeds |
| GBS start gives −10% final LNS C vs LB-SGC | 10/10 instances, full model, 150 iterations | 15/20 stations, subproblem model, 1 h |
| Final C follows start W (r = 0.98); LNS keeps W | same | is it the LNS or the makespan-only acceptance? |
| Graph neighbourhood +2.6 pp, no symmetry breaking +2.0 pp fetch reduction; full = subproblem per iteration | fetch benchmark | anchored subproblem only timed (3.6 s/iter), no quality run |
| Evolution: −1.8% W (work fitness), −3.8% C / +0.9% W (makespan fitness) | heuristic only | never run through LNS |
| Regret partition −1.8% W, + last-chance −2.1% W; CFSS consecutive + share | heuristic only (now on main) | never run through LNS |
| LNS-6 (`mmmf/4`) best LNS configuration | old subproblem rule, symmetry breaking, LB-SGC start | stale |

## Hypotheses

- **H1 (dose-response).** Final LNS C is linear in the start's W across a graded set of starts; the start's C
  adds nothing once W is controlled. Falsifier: `gbs_evolve_makespan` (low C, W ≈ seeds) ends > 1% below
  `gbs_max_sharing`.
- **H2 (scale).** The GBS advantage holds at 15 and 20 stations (it was larger at 10 than at 5 stations).
- **H3 (W recovery).** At 1 h a makespan-driven LNS does not recover a poor start's W; W-targeted phases
  (`b`, `s`) recover part of it.
- **H4 (time accounting).** 15–30 min of evolution beats the same time spent in LNS; the regret partition gets
  most of evolution's W gain in milliseconds. Read off the H1 anytime curves (no extra jobs).
- **H5 (model).** At equal wall time the subproblem LNS beats the full model, increasingly with size.
- **H6 (objectives)**, **H7 (strategies)**, **H8 (neighbourhood size × search limit)**: screened in S2 on the
  tuning instances.
- **H10 (regimes).** GBS's edge shrinks where robots are not the bottleneck (PickCap 2–3, MoveCap 80–160) and
  may shrink where copies bind (1–2 copies per SKU). Heuristics on 8 regimes; LNS on 3.
- **H11/H12 (heuristics).** CFSS fixes help but stay far from GBS; the GBS ablation holds on new seeds and at
  20/800. From `starts.csv` (no LNS).
- **H14 (bounds).** Gap of final W to the lane bound W_lb, and of C to C_lb, from `bounds.csv`.
- **H16 (headline).** GBS-family start + LNS beats the warm-started monolithic CP and every heuristic at 1 h.

## Instance sets (`instances.py`)

| Set | Instances | Use |
|---|---|---|
| `eval` | GEN1–5 × 5/300, 10/400, 15/600 (the existing 15) | reporting only |
| `tune` | GEN6–10 × the same shapes | S2 screening (avoids tuning on the eval set) |
| `scale` | GEN1–3 × 20/800 | H2, H5, H16 |
| `regime` | GEN1–5 × 10/400 × {pc2, pc3, mc20, mc80, mc160, b1, b2, b4} | H10 heuristics |
| `regime_lns` | the pc2, mc80, b2 part of `regime` | H10 LNS |

## Stages and commands

All commands run from the repo root on the server; `ROOT=results/campaign`, `PY=.venv/bin/python`.
Every stage prints `[k/N] ... ETA` lines to `$ROOT/progress.txt`. The runner skips finished jobs, so a stopped
stage can just be restarted with the same command.

```bash
ROOT=results/campaign; PY=.venv/bin/python
scripts/campaign/freeze.sh $ROOT/code                       # frozen code + engine (refuses a dirty tree)
$PY scripts/campaign/instances.py --out $ROOT/instances      # ~5 min
lscpu -e                                                    # check cpus 0-7 are distinct physical cores
```

**Stage 0: pilots (~2 h wall).** Starts for the pilots, then resources at every size and the concurrency
slowdown. Run the pilots alone on the machine.
```bash
$PY scripts/campaign/starts.py --root $ROOT --sets eval scale tune --starts gbs_cluster_lpt -j 14
$PY scripts/campaign/run.py --root $ROOT --src $ROOT/code --exp P0_lns P0_cp --slots 1 --cores 0 --mem-gb 62
$PY scripts/campaign/run.py --root $ROOT --src $ROOT/code --exp P0_long --slots 2 --cores 0-1
$PY scripts/campaign/concurrency.py --root $ROOT --src $ROOT/code --levels 1 4 7 8 --iters 30
$PY scripts/campaign/collect.py --root $ROOT --exp P0_lns P0_cp P0_long
```
Afterwards: update `MEM_GB`, `SETUP_S` and `ITER_S` in `experiments.py` from `collected/P0_*_runs.csv`
(`peak_rss_gb`, `setup_s`, `iter_s`) and choose `--slots` from the concurrency summary (7 unless the slowdown
at 7 is large). Then commit and freeze again for S2 (`freeze.sh $ROOT/code_S2`).

**Stage 1: starts, bounds, heuristic tables (~6 h, CPU only, can overlap S2 on the spare cores).**
```bash
$PY scripts/campaign/starts.py --root $ROOT -j 8 --starts lb_sgc lb_sgc_share rdi_lb_share cfss_share \
    gbs_no_defer gbs_max_sharing gbs_cluster_lpt gbs_regret gbs_regret_lc       # all sets
$PY scripts/campaign/starts.py --root $ROOT -j 8 --sets eval scale --starts gbs_evolve_work gbs_evolve_makespan
$PY scripts/campaign/starts.py --root $ROOT --bounds -j 8 --sets eval scale regime_lns
```
The evolution starts take ~15–30 min each (10 000 decodings). Starts are not timed against each other, so
they can share the machine with S2 if S2 runs on cores 0–6 and the starts get ≤ 1–2 processes. Otherwise run
Stage 1 first.

**Stage 2: screening (~91 slot-h ≈ 13 h wall).** 22 arms × 10 tuning instances (10/400, 15/600) × 2 runs ×
10 min, start `gbs_cluster_lpt`.
```bash
$PY scripts/campaign/run.py --root $ROOT --src $ROOT/code_S2 --exp S2_screen --slots 7 --cores 0-6 --mem-gb 56
$PY scripts/campaign/collect.py --root $ROOT --exp S2_screen
```
Then: analyse, set `FINAL` in `experiments.py` (and the H3 objective arms), commit, `freeze.sh $ROOT/code_S3`.

**Stage 3: main experiments (~594 slot-h ≈ 85 h wall at 7 slots).** 1 h runs, 2 runs per job.

| Experiment | Jobs | Slot-h | Content |
|---|---|---|---|
| H1_starts | 150 | 310 | 10 starts × 15 eval instances (headline and dose-response) |
| H2_scale | 12 | 27 | 4 starts × 3 scale instances |
| H3_objective | 40 | 83 | {lb_sgc, gbs_regret_lc} × 2 W-targeted schedules × 10 eval (10/400, 15/600) |
| H5_full | 36 | 75 | full model, {lb_sgc, gbs_regret_lc} × 18 instances (memory-heavy) |
| H10_regime | 30 | 62 | {lb_sgc_share, gbs_regret_lc} × 15 regime instances |
| H16_cp | 36 | 38 | monolithic CP, 1 h, {lb_sgc, gbs_regret_lc} × 18 instances (memory-heavy) |

```bash
nohup $PY scripts/campaign/run.py --root $ROOT --src $ROOT/code_S3 --slots 7 --cores 0-6 --mem-gb 56 \
    --exp H1_starts H2_scale H3_objective H5_full H10_regime H16_cp > $ROOT/nohup_S3.out 2>&1 &
```
The memory-heavy jobs (full model and CP at 600/800 orders) start first (longest first) and run alongside
light ones within the 56 GB budget. Run all of S3 in one invocation, so every job sees the same load mix.

**Buffer (~1 day):** reruns and one follow-up chosen from the S3 results.

## Measurement rules

- A job is one process with `runs` LNS runs (or one CP solve per seed), pinned to one physical core with
  `OMP_NUM_THREADS=1`. CP Optimizer runs with 1 worker everywhere.
- Time budgets are wall-clock per run (`--time-limit`), including model setup. Comparisons at equal time use
  the anytime trajectories (`*_traj.csv.gz`).
- Statistics: per-instance means over runs, paired by instance (Wilcoxon signed-rank, Holm correction; TOST
  for equivalence claims), as in `scripts/warmstart_benchmark/analyze.py`.
- Disk: an LNS experiment folder is ~10–60 MB in the compact format; ~300 jobs → ≤ 20 GB.
