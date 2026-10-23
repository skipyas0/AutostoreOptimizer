"""Run every equivalence case with either the OLD (pre-refactor) or NEW code
and write {case_id: {"fp": fingerprint, "time": seconds}} to a JSON file.

Invoked by equivalence.py; the OLD side runs with PYTHONPATH pointing at a git
worktree of the reference commit.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_config")

import cases as C  # noqa: E402


# ---------------------------------------------------------------------------
# OLD adapter (pre-refactor API)
# ---------------------------------------------------------------------------


def _old_partition_lists(inst, method, seed):
    from decomposition import (
        create_load_balanced_partitions,
        create_random_partitions,
        create_similarity_partitions,
    )

    if method == "explicit":
        return C.explicit_assignment(inst)
    random.seed(seed)
    fn = {
        "random": create_random_partitions,
        "similarity": create_similarity_partitions,
        "load_balanced": create_load_balanced_partitions,
    }[method]
    return {i: list(p) for i, p in enumerate(fn(inst, len(inst.S)))}


def old_run(heur, inst, params):
    hz = params.get("horizon", 10000)
    a = params.get("alpha", 1.0)
    b = params.get("beta", 0.0)
    if heur == "sgc":
        from autostore_heuristic import run_sgc

        return run_sgc(
            inst, horizon=hz, ALPHA=a, BETA=b,
            restrict_stations_by_workload=params.get("load_balanced", False),
        )
    if heur == "rdi":
        from heuristic_rdi_sgc import run_rdi_sgc

        kw = {}
        exo = params.get("exo")
        if exo:
            kw = dict(
                exo_lanes=exo["lanes"], exo_blocks=exo["blocks"],
                exo_moves=exo["moves"], batch_start_time=exo["batch_start_time"],
            )
        return run_rdi_sgc(
            inst, horizon=hz, ALPHA=a, BETA=b,
            regret_k=params.get("regret_k", 2),
            use_lazy=params.get("use_lazy", True),
            tiebreaker=params.get("tiebreaker", "sharing_degree"),
            **kw,
        )
    if heur == "cfss":
        if params.get("normalize_similarity", False):
            from heuristic_cfss_sgc_normalized import run_cfss_sgc
        else:
            from heuristic_cfss_sgc import run_cfss_sgc
        return run_cfss_sgc(
            inst, horizon=hz, move_cap=inst.movecap, ALPHA=a, BETA=b,
            sim_threshold=params.get("sim_threshold", 0.1),
            max_cluster_orders=params.get("max_cluster_orders", 8),
        )
    if heur == "gbs":
        from heuristic_gbs import run_gbs

        rule = params.get("rule", "max_sharing")
        if rule != "adaptive":
            return run_gbs(inst, horizon=hz, move_cap=inst.movecap, ALPHA=a, BETA=b, scoring_rule=rule)
        # run_gbs_adaptive crashes at the reference commit (bad Instance(...) call);
        # emulate its intended selection logic on top of the old run_gbs.
        best_sol, best_obj, best_moves = None, float("inf"), float("inf")
        for r in ["max_sharing", "critical_path", "readiness_weighted"]:
            sol = run_gbs(inst, horizon=hz, move_cap=inst.movecap, ALPHA=a, BETA=b, scoring_rule=r)
            if not sol.feasible:
                continue
            flow = sum(sol.order_assignments[o][3] - sol.order_assignments[o][2] for o in inst.O)
            obj = a * sol.makespan + b * flow
            if obj < best_obj or (obj == best_obj and sol.total_moves < best_moves):
                best_sol, best_obj, best_moves = sol, obj, sol.total_moves
        return best_sol
    if heur == "partitioned_rdi":
        from decomposition import distribute_movecap, stitch_heuristic_solutions
        from heuristic_rdi_sgc import run_rdi_sgc
        from instance import Instance

        assignment = _old_partition_lists(inst, params["partition"], params.get("seed"))
        n = len(inst.S)
        workloads = [len(assignment.get(i, [])) for i in range(n)]
        caps = distribute_movecap(inst.movecap, n, workloads)
        subs = []
        for i, s_idx in enumerate(inst.S):
            orders = assignment.get(i, [])
            if not orders:
                continue
            sub = Instance(
                [s_idx], inst.L, inst.K, {o: inst.orders_req[o] for o in orders},
                inst.rt, inst.p, inst.N, caps[i], inst.seed, inst.rt_ret, inst.pickcap,
            )
            subs.append(run_rdi_sgc(sub, horizon=50000, batch_start_time=0))
        return stitch_heuristic_solutions(subs)
    raise KeyError(heur)


def old_bench(inst, method, seed):
    import benchmark_models as bm

    random.seed(seed)
    fn = {
        "random": bm.run_random_partition_rdi,
        "similarity": bm.run_similarity_partition_rdi,
        "load_balanced": bm.run_load_balanced_partition_rdi,
        "full_rdi": bm.run_full_rdi_no_decomposition,
        "greedy": lambda i: bm.run_full_greedy_heuristic(i, restrict_stations_by_workload=False),
        "lb_greedy": lambda i: bm.run_full_greedy_heuristic(i, restrict_stations_by_workload=True),
    }[method]
    return fn(inst)


def old_validate(sol, inst, hz):
    from autostore_heuristic import validate_solution

    return validate_solution(sol, inst, horizon=hz, move_cap=inst.movecap)


def old_validate_warmstart(sol, pick_events, handles):
    from autostore_heuristic import validate_warmstart

    return validate_warmstart(sol, pick_events, handles)


# ---------------------------------------------------------------------------
# NEW adapter (refactored API)
# ---------------------------------------------------------------------------


def new_run(heur, inst, params):
    params = dict(params)
    if heur == "sgc":
        from heuristics import run_sgc

        return run_sgc(inst, **params)
    if heur == "rdi":
        from heuristics import ExoState, run_rdi

        if params.get("exo"):
            params["exo"] = ExoState(**params["exo"])
        return run_rdi(inst, **params)
    if heur == "cfss":
        from heuristics import run_cfss

        return run_cfss(inst, **params)
    if heur == "gbs":
        from heuristics import run_gbs

        return run_gbs(inst, **params)
    if heur == "partitioned_rdi":
        from heuristics import run_partitioned_rdi

        if params["partition"] == "explicit":
            params["partition"] = C.explicit_assignment(inst)
        params.pop("horizon", None)
        return run_partitioned_rdi(inst, **params)
    raise KeyError(heur)


def new_bench(inst, method, seed):
    import benchmark_models as bm

    name = {
        "random": "1. Random-partitioned RDI",
        "similarity": "2. Similarity-partitioned RDI",
        "load_balanced": "3. Load-balanced RDI",
        "full_rdi": "4. Full RDI (No Decomposition)",
        "greedy": "5. Standard Full Greedy (SGC)",
        "lb_greedy": "6. Load-balanced Greedy (SGC)",
    }[method]
    random.seed(seed)
    return bm.run_method(bm.DEFAULT_METHODS[name], inst)


def new_validate(sol, inst, hz):
    from heuristics import validate_solution

    return validate_solution(sol, inst, horizon=hz)


def new_validate_warmstart(sol, pick_events, handles):
    from heuristics import validate_warmstart

    return validate_warmstart(sol, pick_events, handles)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _work(side, spec):
    run = old_run if side == "old" else new_run
    bench = old_bench if side == "old" else new_bench
    validate = old_validate if side == "old" else new_validate
    validate_ws = old_validate_warmstart if side == "old" else new_validate_warmstart

    out = {}
    with contextlib.redirect_stdout(io.StringIO()):
        inst = C.build_instance(spec)
        for case_id, heur, params, extras in C.cases_for(spec, inst):
            try:
                t0 = time.perf_counter()
                if heur == "bench_metrics":
                    res = C.strip_times(bench(inst, params["method"], params["seed"]))
                    out[case_id] = {"fp": res, "time": time.perf_counter() - t0}
                    continue
                sol = run(heur, inst, params)
                el = time.perf_counter() - t0
                viol = validate(sol, inst, params["horizon"]) if sol is not None else None
                fp = C.fingerprint(sol, viol)
                if fp is not None and extras.get("warmstart"):
                    fp["warmstart"] = C.warmstart_fingerprint(sol, inst, params["horizon"], validate_ws)
                out[case_id] = {"fp": fp, "time": el}
            except Exception as e:  # record crashes as part of the fingerprint
                out[case_id] = {
                    "fp": {"error": f"{type(e).__name__}: {e}"},
                    "time": 0.0,
                    "tb": traceback.format_exc(),
                }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["old", "new"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--filter", default=None, help="substring filter on instance ids")
    args = ap.parse_args()

    specs = C.instance_specs()
    if args.filter:
        specs = [s for s in specs if args.filter in s["id"]]
    results = {}
    t0 = time.perf_counter()
    if args.jobs == 1:
        for sp in specs:
            results.update(_work(args.side, sp))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(_work, args.side, sp) for sp in specs]
            for i, f in enumerate(as_completed(futs), 1):
                results.update(f.result())
                print(f"[{args.side}] {i}/{len(specs)} instances done", file=sys.stderr, flush=True)
    with open(args.out, "w") as f:
        json.dump(results, f)
    print(f"[{args.side}] {len(results)} cases in {time.perf_counter() - t0:.1f}s -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
