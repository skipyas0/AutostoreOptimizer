"""Evolutionary search over GBS order sequences (``run_gbs_evolve``).

Genome: for every station, the sequence in which its orders are admitted to the lanes. This
encodes the partition (which station gets an order) *and* the admission order. Optionally, a
per-SKU bias that is added to GBS's bin-task score (``evolve_bias``), i.e. it reorders the bin
visits at the stations.

Decoder: ``gbs.run_gbs_sequence`` builds the full schedule from a genome (GBS with the partition
and admission order fixed, ~0.05-0.1 s on 300-600 orders). GBS's own runs record their admission
order (``meta["sequence"]``), and decoding that sequence reproduces them exactly, so the search
starts from the GBS solutions and can only improve on the best of them.

Fitness (lower is better, compared lexicographically):

- ``"makespan"``: (C_max, W)
- ``"work"``: (W, C_max), W = robot work Σ (rt + rt_ret) over bin visits. LNS keeps the
  warm start's W almost unchanged and ends at C ≈ 1.03-1.08 · W / MoveCap, so W is what a warm
  start carries over.

Search (``method``):

- ``"ls"``: (1+1) EA. Mutate the incumbent with 1 + Poisson(0.5) operators and accept the child
  if it is not worse (neutral moves let it drift across makespan plateaus).
- ``"ga"``: steady-state GA. Binary tournament, station-group crossover (probability
  ``crossover_rate``), mutation; the child replaces the worst member if it is better and its
  fitness is not already in the population.

Mutation operators (uniformly chosen): ``swap`` (two nearby orders of one station), ``insert``
(move an order within its station), ``transfer`` (move an order to another station), ``exchange``
(swap two orders between stations), ``pull`` (put an order right after a random order sharing one
of its SKUs, possibly at another station, so the two are active together and can share visits)
and, with ``evolve_bias``, ``bias`` (perturb one SKU's task-score bias).
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

from heuristics.gbs import GBS_RULES, run_gbs, run_gbs_sequence
from heuristics.solution import HeuristicSolution
from instance import Instance

EVOLVE_METHODS = ("ls", "ga")
EVOLVE_FITNESS = ("makespan", "work")
SEED_PARTITIONS = ("overlap", "lpt", "cluster_lpt")
OPS = ("swap", "insert", "transfer", "exchange", "pull")


@dataclass
class Genome:
    seq: list[list[int]]  # seq[i]: admission sequence of station inst.S[i]
    bias: dict[int, float] = field(default_factory=dict)  # SKU -> task score bias

    def copy(self) -> Genome:
        return Genome([list(x) for x in self.seq], dict(self.bias))


def robot_work(sol: HeuristicSolution, inst: Instance) -> int:
    return sum(inst.rt[ev.sku] + inst.rt_ret[ev.sku] for evs in sol.bin_events.values() for ev in evs)


class _Problem:
    def __init__(self, inst: Instance, horizon: int, rule: str, defer, window: int, fitness: str):
        self.inst, self.horizon, self.rule, self.defer, self.window = inst, horizon, rule, defer, window
        self.fitness = fitness
        self.n_evals = 0
        req = inst.orders_req
        self.work_p = {o: sum(inst.p[k] for k in req[o]) for o in inst.O}
        self.sharers = {o: [o2 for k in req[o] for o2 in inst.sku_order_index[k] if o2 != o] for o in inst.O}
        self.visit_w = {k: inst.rt[k] + inst.rt_ret[k] for o in inst.O for k in req[o]}

    def decode(self, g: Genome) -> HeuristicSolution:
        seq = {s: g.seq[i] for i, s in enumerate(self.inst.S)}
        return run_gbs_sequence(self.inst, seq, horizon=self.horizon, rule=self.rule, defer=self.defer,
                                window=self.window, sku_bias=g.bias or None)

    def evaluate(self, g: Genome) -> tuple[tuple, HeuristicSolution]:
        self.n_evals += 1
        sol = self.decode(g)
        if not sol.feasible:
            return (math.inf, math.inf), sol
        w = robot_work(sol, self.inst)
        return ((sol.makespan, w) if self.fitness == "makespan" else (w, sol.makespan)), sol


# ======================================================================
# Variation
# ======================================================================


def _pick_station(g: Genome, rng: random.Random, min_len: int = 1) -> int | None:
    idx = [i for i, x in enumerate(g.seq) if len(x) >= min_len]
    if not idx:
        return None
    return rng.choices(idx, weights=[len(g.seq[i]) for i in idx])[0]


def _locate(g: Genome) -> dict[int, tuple[int, int]]:
    return {o: (i, j) for i, x in enumerate(g.seq) for j, o in enumerate(x)}


def mutate(g: Genome, pb: _Problem, rng: random.Random, evolve_bias: bool, reach: int = 8) -> str:
    """Apply one random operator to g in place; returns its name."""
    ops = OPS + (("bias",) if evolve_bias else ())
    op = rng.choice(ops)
    n_st = len(g.seq)
    if op in ("transfer", "exchange") and n_st < 2:
        op = "insert"
    if op == "swap":
        i = _pick_station(g, rng, 2)
        if i is None:
            return op
        x = g.seq[i]
        a = rng.randrange(len(x))
        b = min(len(x) - 1, max(0, a + rng.choice((-1, 1)) * rng.randint(1, reach)))
        x[a], x[b] = x[b], x[a]
    elif op == "insert":
        i = _pick_station(g, rng, 2)
        if i is None:
            return op
        x = g.seq[i]
        o = x.pop(rng.randrange(len(x)))
        x.insert(rng.randrange(len(x) + 1), o)
    elif op == "transfer":
        i = _pick_station(g, rng, 2)
        if i is None:
            return op
        j = rng.choice([j for j in range(n_st) if j != i])
        a = rng.randrange(len(g.seq[i]))
        o = g.seq[i].pop(a)
        pos = round(a / max(1, len(g.seq[i])) * len(g.seq[j]))
        g.seq[j].insert(min(len(g.seq[j]), pos), o)
    elif op == "exchange":
        i = _pick_station(g, rng)
        j = rng.choice([j for j in range(n_st) if j != i and g.seq[j]] or [i])
        if i == j:
            return op
        a, b = rng.randrange(len(g.seq[i])), rng.randrange(len(g.seq[j]))
        g.seq[i][a], g.seq[j][b] = g.seq[j][b], g.seq[i][a]
    elif op == "pull":
        i = _pick_station(g, rng)
        if i is None:
            return op
        o = rng.choice(g.seq[i])
        if not pb.sharers[o]:
            return op
        o2 = rng.choice(pb.sharers[o])
        g.seq[i].remove(o)
        j, b = _locate(g)[o2]
        g.seq[j].insert(b + 1, o)
    elif op == "bias":
        i = _pick_station(g, rng)
        if i is None:
            return op
        k = rng.choice(pb.inst.orders_req[rng.choice(g.seq[i])])
        if k in g.bias and rng.random() < 0.25:
            del g.bias[k]
        else:
            g.bias[k] = g.bias.get(k, 0.0) + rng.gauss(0.0, pb.visit_w[k])
    return op


def crossover(a: Genome, b: Genome, pb: _Problem, rng: random.Random) -> Genome:
    """Station-group crossover: keep a random half of A's stations (sequences included), fill the
    other stations with B's sequences minus the orders already placed; orders left over go to the
    least-loaded of those stations, at their relative position in B."""
    n = len(a.seq)
    if n < 2:
        return a.copy()
    keep = set(rng.sample(range(n), rng.randint(1, n - 1)))
    child = [list(a.seq[i]) if i in keep else [] for i in range(n)]
    placed = {o for i in keep for o in a.seq[i]}
    for i in range(n):
        if i not in keep:
            child[i] = [o for o in b.seq[i] if o not in placed]
    placed.update(o for i in range(n) if i not in keep for o in child[i])
    load = {i: sum(pb.work_p[o] for o in child[i]) for i in range(n) if i not in keep}
    for i in keep:  # orders B has at A's kept stations but A has elsewhere (outside `keep`)
        for pos, o in enumerate(b.seq[i]):
            if o in placed:
                continue
            j = min(load, key=lambda j: (load[j], j))
            rel = pos / max(1, len(b.seq[i]))
            child[j].insert(round(rel * len(child[j])), o)
            load[j] += pb.work_p[o]
            placed.add(o)
    bias = {k: v for k, v in (a.bias | b.bias).items()}
    for k in set(a.bias) & set(b.bias):
        bias[k] = (a.bias if rng.random() < 0.5 else b.bias)[k]
    return Genome(child, bias)


# ======================================================================
# Search
# ======================================================================


def _seeds(inst: Instance, horizon: int, rule: str, defer, partitions) -> list[tuple[str, Genome]]:
    out = []
    for part in partitions:
        sol = run_gbs(inst, horizon=horizon, rule=rule, defer=defer, partition=part)
        out.append((part, Genome([list(sol.meta["sequence"].get(s, [])) for s in inst.S])))
    return out


def run_gbs_evolve(
    inst: Instance,
    *,
    horizon: int = 10000,
    method: str = "ga",
    fitness: str = "makespan",
    max_evals: int = 300,
    time_limit: float | None = None,
    seed: int = 0,
    rule: str = "max_sharing",
    defer: str | None = "strict",
    window: int = 1,
    evolve_bias: bool = False,
    seed_partitions: tuple[str, ...] = SEED_PARTITIONS,
    pop_size: int = 24,
    crossover_rate: float = 0.5,
    reach: int = 8,
) -> HeuristicSolution:
    """Evolve GBS order sequences (see the module docstring).

    Budget: ``max_evals`` decodings (the seeds' GBS runs not counted) and/or ``time_limit`` seconds,
    whichever comes first. ``meta["history"]``: (evals, seconds, best fitness) at every improvement.
    """
    if method not in EVOLVE_METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {EVOLVE_METHODS}")
    if fitness not in EVOLVE_FITNESS:
        raise ValueError(f"Unknown fitness {fitness!r}; choose from {EVOLVE_FITNESS}")
    if rule not in GBS_RULES:
        raise ValueError(f"Unknown GBS rule {rule!r}; choose from {GBS_RULES}")
    t_start = time.perf_counter()
    rng = random.Random(seed)
    pb = _Problem(inst, horizon, rule, defer, window, fitness)

    def out_of_budget() -> bool:
        if pb.n_evals >= max_evals:
            return True
        return time_limit is not None and time.perf_counter() - t_start >= time_limit

    seeds = _seeds(inst, horizon, rule, defer, seed_partitions)
    pb_evals0 = pb.n_evals
    pop: list[tuple[tuple, Genome, HeuristicSolution]] = []
    seed_fit = {}
    for name, g in seeds:
        f, sol = pb.evaluate(g)
        seed_fit[name] = f
        pop.append((f, g, sol))
    pb.n_evals = pb_evals0  # seed decodings are free: they equal the GBS runs
    best = min(pop, key=lambda x: x[0])
    history = [(0, time.perf_counter() - t_start, best[0])]
    op_stats: dict[str, list[int]] = {}  # op -> [tried, improved-or-equal accepted]

    def child_of(g: Genome) -> tuple[Genome, list[str]]:
        c = g.copy()
        ops = [mutate(c, pb, rng, evolve_bias, reach) for _ in range(1 + _poisson(rng, 0.5))]
        return c, ops

    def note(ops, success):
        for op in ops:
            st = op_stats.setdefault(op, [0, 0])
            st[0] += 1
            st[1] += int(success)

    if method == "ls":
        cur = best
        while not out_of_budget():
            c, ops = child_of(cur[1])
            f, sol = pb.evaluate(c)
            note(ops, f < cur[0])
            if f <= cur[0]:
                cur = (f, c, sol)
                if f < best[0]:
                    best = cur
                    history.append((pb.n_evals, time.perf_counter() - t_start, f))
    else:  # ga
        fits = {x[0] for x in pop}
        while len(pop) < pop_size and not out_of_budget():  # fill with mutants of the seeds
            c = rng.choice(seeds)[1].copy()
            for _ in range(rng.randint(3, 15)):
                mutate(c, pb, rng, evolve_bias, reach)
            f, sol = pb.evaluate(c)
            pop.append((f, c, sol))
            fits.add(f)
            if f < best[0]:
                best = (f, c, sol)
                history.append((pb.n_evals, time.perf_counter() - t_start, f))

        def tournament():
            a, b = rng.sample(pop, 2)
            return a if a[0] <= b[0] else b

        while not out_of_budget():
            p1 = tournament()
            if rng.random() < crossover_rate:
                c = crossover(p1[1], tournament()[1], pb, rng)
                ops = ["crossover"]
                ops += [mutate(c, pb, rng, evolve_bias, reach) for _ in range(_poisson(rng, 0.5))]
            else:
                c, ops = child_of(p1[1])
            f, sol = pb.evaluate(c)
            worst = max(range(len(pop)), key=lambda i: pop[i][0])
            ok = f < pop[worst][0] and f not in fits
            note(ops, ok and f < p1[0])
            if ok:
                fits.discard(pop[worst][0])
                pop[worst] = (f, c, sol)
                fits.add(f)
                if f < best[0]:
                    best = (f, c, sol)
                    history.append((pb.n_evals, time.perf_counter() - t_start, f))

    f, g, sol = best
    sol.meta.update({
        "heuristic": "gbs_evolve", "method": method, "fitness": fitness, "fitness_value": f,
        "evals": pb.n_evals, "runtime_search": time.perf_counter() - t_start, "seed": seed,
        "window": window, "evolve_bias": evolve_bias, "seed_fitness": seed_fit,
        "history": history, "op_stats": op_stats, "n_bias": len(g.bias),
        "sequence": {s: list(g.seq[i]) for i, s in enumerate(inst.S)},
    })
    return sol


def _poisson(rng: random.Random, lam: float) -> int:
    k, p, L = 0, 1.0, math.exp(-lam)
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1
