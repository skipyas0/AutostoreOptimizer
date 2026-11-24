# Future ideas

Parked ideas with the evidence we have so far. Every entry says what was measured, on which data, and what is
still open. Numbers marked "15 eval instances" are on `precalculated_instances/` (5 / 10 / 15 stations, 3 lanes,
PickCap 1, MoveCap 40, β = 1) relative to `lb_sgc_share`, unless stated otherwise.

## GBS: work stealing at the end of a station's queue

**Idea.** With interleaved stations (`interleave=True`), a station whose own queue is empty leaves lanes idle while
other stations still have unstarted orders. Let it take one of those orders instead.

**Rule (as tested).**
- *When*: right before and right after a station's step, for each of its lanes that is free while its own
  `unscheduled` list is empty.
- *Victim*: the other station with the largest unstarted picking work `Σ p` (ties: first station), provided it still
  has at least `q` unstarted orders (`steal_min_queue`).
- *Which order*: the victim's unstarted order whose SKUs the victim's *other* unstarted orders need least, i.e. the
  smallest `Σ (rt + rt_ret)` over its SKUs with `victim.future[k] > 1` (ties: earlier in the victim's queue).
  The alternatives "victim's next order" and "best overlap with the thief's active SKUs" performed about the same.
- *Bookkeeping*: remove the order from `victim.unscheduled`, subtract its SKUs from `victim.future`, admit it to the
  thief's free lane, record its new station in the assignment. A station stays in the interleaved loop while it can
  still steal.

**Evidence (15 eval instances, strict deferral, scratch prototype).**

| | makespan vs no stealing | instances better / worse | makespan imb. | workload imb. |
|---|---|---|---|---|
| `cluster_lpt`, q = 1 | −1.35% (−10.1% vs `lb_sgc_share`) | 12 / 3 | 17.7% → 6.0% | 0.2% → 9.7% |
| `cluster_lpt`, q = 2 | −0.8% | 10 / 4 | → 8.0% | → 8.0% |
| `cluster_lpt`, q = 4 | −0.7% | 9 / 3 | → 11.3% | → 5.9% |
| `overlap`, q = 1 | −0.8% | 7 / 7 | 14.1% → 6.0% | 14.9% → 18.9% |

Robot work and fetches are unchanged (only unstarted orders move). Valid on the regression grid (438 runs incl.
PickCap 2–3, MoveCap 3, tight horizon; 125 runs actually stole).

**Trade-off.** Stealing trades workload balance (Σ p per station) for finish-time balance: stations that share
more need fewer visits and finish early, so they take extra picking work.

**Verdict for LNS warm starts: not worth it** (warm-start benchmark, 2026-11-24, `scripts/warmstart_benchmark/`,
10 instances × 4 runs × 150 iterations, full-model LNS):
- LNS evens finishing times on its own from every start (makespan imbalance 8–16% → 0.6–2.5%), which is the only
  thing stealing improves.
- LNS does not keep the start's workload balance either: `gbs_cluster_lpt` 0.2% → 10.1%, `gbs_max_sharing`
  14.8% → 17.4%, LB-SGC 6.6% → 8.3%.
- The two GBS starts have the same robot work but very different balance; their final makespans differ by
  −0.2 pp (95% CI −0.6 to +0.2, 7/3 instances, p = 0.28). Final makespan follows the start's robot work (r = 0.98),
  which stealing does not change.

Stealing would only matter where the GBS schedule itself is used (no LNS afterwards). The option would be
`steal_min_queue: int | None = None`, off by default.

## GBS: cheaper clustering for `cluster_lpt`

`agglomerative_cluster` rescans all pair similarities on every merge (O(n³)); it makes `gbs_cluster_lpt` take ~2.6 s
instead of ~1.1 s. Only ~11% of order pairs share any SKU, so a sparse similarity (SKU → orders index) plus a heap of
candidate merges would make it near-linear in the number of overlapping pairs. Results must stay identical.

## CFSS: let the clusters reach the schedule

Diagnosis (15 eval instances): clusters are co-located at a station, but phase 2 commits the lowest-score (shortest)
pending order wherever it sits, and the SGC planner runs without the share options. With PickCap 1 a bin can only be
shared while it is still at the pickface, so co-location alone gives ~3 shared visits per instance and +2.4% fetches.

| CFSS variant | makespan | fetches | robot work | shared visits |
|---|---|---|---|---|
| as is | −0.3% | +2.4% | +1.4% | 3 |
| + share options (window, λ = 1) | −1.6% | +0.7% | +0.4% | 25 |
| clusters scheduled consecutively | −1.0% | +0.6% | +0.4% | 21 |
| consecutive + share options | −2.0% | −1.9% | −1.2% | 54 |

Further fixes: balance clusters on Σ p instead of rt mass (workload imbalance 12%); `normalize_similarity=False`
makes the "similarity" the raw overlap weight (17–88), so `sim_threshold=0.1` never binds; sparse similarity as
above. CFSS stays an order-centred planner, so its ceiling is low; `gbs_cluster_lpt` is where clustering pays off.

## SGC warm starts: tune for robot work and balance, not makespan

LNS keeps the warm start's robot work W almost unchanged (final makespan ≈ 1.03 · W / MoveCap on the 15 instances),
so a warm start should be judged by W and balance.

- Pilot (LB-SGC share, 192 rule pairs): fetches vary only −0.5% to +2.6% across all rules, but workload imbalance
  0.5–7.5%. Descending order rules (largest first) balance well; the makespan-best rule (`sum_rt` desc / `rt` asc)
  costs +1.2% W. λ = 10 gives −0.7% W; β = 0 −0.8% W; LB tolerance and shared-first SKU order did not help.
- Proposed experiment: full factorial of rule pairs × λ ∈ {0, 1, 3, 10} × β ∈ {0, 1} × piggybacking on a tuning set
  (seeds 6–20), select by W under balance constraints, validate with LNS on the eval set.
- AMA keeps the best run by makespan; selecting by W under balance constraints would pick better warm starts at no
  extra runtime.
