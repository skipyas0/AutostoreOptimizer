# Sequencing vs. scheduling

Research question: much of the RCS/RS literature solves an **assignment + sequencing**
problem (orders to stations, order and bin sequences per station) and minimises the
number of **bin fetches**. This repo solves the **scheduling** problem and minimises
**makespan**. How far apart are the two optima?

* **Q1:** How many fetches does a makespan-optimal schedule use, and how similar is its
  sequence to a fetch-optimal sequence?
* **Q2:** If the scheduling model must follow a fetch-optimal sequence, how much makespan
  is lost?

This package only *imports* the main code (`cp_model`, `datagen`, `heuristics`). It does
not change it, so CP+LNS development is unaffected.

## Files

| File | Content |
|---|---|
| `sequence.py` | `SequenceSolution` (order -> station, ordered `Visit(sku, orders)` per station), validation, lane check, JSON. **Deliverable 3:** `extract_sequence(cp_sol, handles)` and `sequence_from_schedule(HeuristicSolution)` |
| `seq_model.py` | **Deliverable 1:** sequencing CP model, `solve_sequence_model(inst, secondary)` |
| `fix_schedule.py` | **Deliverable 2:** `constrain_schedule_to_sequence(mdl, handles, seq, level)`; `greedy_schedule_from_sequence` (feasible warm start) |
| `metrics.py` | Structural similarity of two sequences (`compare`), per-solution stats (`describe`) |
| `run_experiment.py` | **Deliverable 4:** benchmark grid, per-instance JSON, `summary.csv`, `summary.md` |
| `tests/` | `python -m pytest seq_vs_sched/tests -q` |

## The sequencing model

Time is replaced by visit positions: each bin visit at a station takes one unit-length
slot. Optional intervals: `X[o,s]` (order window), `V[s,k,e]` (e-th visit of SKU k at s),
`Q[o,s,k,e]` (order o picks from that visit, starts with it).

* each order is assigned to exactly one station, and each of its lines is served by one
  visit of that SKU at that station;
* one visit per position (`no_overlap`), and at most |L| open orders per station
  (cumulative over `X`);
* objective: minimise the number of visits. `secondary="balance"` adds a lexicographic
  tie-breaker that minimises the busiest station's visit count. Without it, the pure
  objective has no reason to spread work over stations.

This model does not include times, MoveCap, PickCap, bin copies N[k] or cross-station
timing, which is the usual abstraction in the sequencing papers.

**Lane semantics.** In the sequencing model, an order is open from its first to its last
visit, inclusive. A schedule can be slightly denser: within one bin visit, order A can
take its last pick and order B its first pick in the same lane. So a schedule's sequence
can break the sequencing model's lane limit (`seq_lane_feasible = False`), and the timed
fetch optimum can be lower than the sequencing optimum. The reverse direction always
works: any sequencing-feasible sequence can be realised by the scheduling model.

## Restricting the schedule (levels)

| level | fixed | free |
|---|---|---|
| `assignment` | order -> station | visits, sequence, lanes, timing |
| `visits` | + which visits exist and which visit serves each order line | visit order, lanes, timing |
| `sequence` | + visit order per station | lanes, timing |

`greedy_schedule_from_sequence(inst, seq)` builds a feasible timed schedule that follows a
sequence exactly (earliest-start list scheduling under MoveCap, N[k] and the station port).
The levels are nested relaxations. The benchmark warm-starts `sequence` from the greedy
schedule, then uses each solution to warm-start the next level, so M(assignment) <= M(visits) <= M(sequence).

## Benchmark protocol

For each instance (generated with `datagen.generate_data`):

1. `SEQ[none]` and `SEQ[balance]`: sequencing model (`--seq-time`).
2. `<arm>|<level>`: scheduling model restricted to each arm, at every level
   (`--fixed-time`).
3. `cold`: makespan CP warm-started from `lb_sgc` only (`--sched-time`). This is the
   unbiased structure for Q1.
4. `ref`: makespan CP warm-started from the best schedule found so far. Its makespan is
   the best known **M\*** (the monolithic CP rarely proves optimality, even here).
5. `fetch_opt` (`--sched-fetch`): scheduling model minimising bin fetches, warm-started
   from the fewest-fetch schedule.

Main outputs (`summary.md`; one row per instance in `summary.csv`):

* `M_ratio_<arm>|<level>` = makespan / M\* (Q2)
* `f_ratio_ref`, `f_ratio_cold` = fetches of the makespan schedule / fetch optimum (Q1)
* `sim_<cold|ref>_vs_<arm>_*`: `assign_match`, `assign_rand`, `order_prec_agree`,
  `visit_seq_sim` and `visit_set_jaccard` (see `metrics.py`; station labels are matched
  first, and all are 1 when identical)
* `f_seq_lb`: the proven lower bound on fetches (equal to `f_seq_best` means optimal)

```bash
python -m seq_vs_sched.run_experiment                       # S in {2,3}, L=4, O=20, K=100, MoveCap in {4,1000}, seeds 1-5
python -m seq_vs_sched.run_experiment --jobs 3 --workers 3 --sched-fetch --out seq_vs_sched/results/toy
python -m seq_vs_sched.run_experiment --summarize seq_vs_sched/results/toy   # re-aggregate
```

Runs resume: instances whose JSON already exists are skipped. `results/` is git-ignored.

## Caveats

* "Optimal" makespan means best known, with the CP lower bound reported (`M_lb`).
  `ref` is warm-started from fetch-optimal schedules, so its structure can be biased
  toward them; read the `cold` similarity columns alongside it.
* Fetch-optimal sequences are not unique. The similarity scores compare against one
  optimum, so they are lower bounds on "the closest fetch-optimal sequence".
