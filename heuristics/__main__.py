"""Run any registered heuristic on a generated (or pickled) instance, or on a batch of
precalculated instances.

    python -m heuristics rdi --stations 4 --lanes 4 --orders 160 --skus 20000 --movecap 20
    python -m heuristics ama_two_phase --orders 40 --vis
    python -m heuristics cfss --instance precalculated_instances/GEN1-1-1-100-10-2/instance.pkl --save
    python -m heuristics rdi precalculated_instances/* --save
    python -m heuristics --list

Batch mode (positional instance paths, each a precalculated folder or its instance.pkl)
takes horizon/alpha/beta from each folder's config.json; --param and the generation /
horizon flags are rejected there.

--save writes heuristic_<name>_solution.pkl, heuristic_<name>_solution.html and
heuristic_<name>_summary.txt next to the instance (or into --out-dir). If any of them
already exists, all three get a _2, _3, ... suffix, so nothing is overwritten.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from heuristics import HEURISTICS, resolve, run_heuristic

# Generation / horizon flags and their defaults. They default to None in argparse so that
# batch mode can tell whether the user set them.
GEN_DEFAULTS = {
    "stations": 4,
    "lanes": 4,
    "orders": 160,
    "skus": 20000,
    "seed": 42,
    "pick": 4,
    "movecap": 20,
    "horizon": 10000,
}

# Fallbacks for config.json keys, matching precalculate_instances_for_lns.solve_warmstart_heuristic.
CONFIG_PARAM_DEFAULTS = {"horizon": 10000, "alpha": 1.0, "beta": 0.0}


def _parse_value(v: str):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def _resolve_instance_path(p: str) -> Path | None:
    """Folder -> folder/instance.pkl; a .pkl file is taken as is. None if there is no instance."""
    path = Path(p)
    if path.is_dir():
        path = path / "instance.pkl"
    return path if path.is_file() and path.suffix == ".pkl" else None


def _config_params(instance_pkl: Path) -> dict:
    cfg_file = instance_pkl.parent / "config.json"
    cfg = {}
    if cfg_file.is_file():
        with open(cfg_file) as f:
            cfg = json.load(f)
    else:
        print(f"[!] No config.json next to {instance_pkl}, using defaults {CONFIG_PARAM_DEFAULTS}")
    return {k: cfg.get(k, d) for k, d in CONFIG_PARAM_DEFAULTS.items()}


def _free_stem(out_dir: Path, name: str) -> str:
    """First stem heuristic_<name>[_k] for which none of the output files exist yet."""
    k = 1
    while True:
        stem = f"heuristic_{name}" + ("" if k == 1 else f"_{k}")
        files = (f"{stem}_solution.pkl", f"{stem}_solution.html", f"{stem}_summary.txt")
        if not any((out_dir / f).exists() for f in files):
            return stem
        k += 1


def _report(sol, violations) -> None:
    print(f"=== {sol.meta['name']} ===")
    print(f"Feasible:   {sol.feasible}")
    print(f"Makespan:   {sol.makespan}")
    print(f"Bin visits: {sol.total_moves // 2}")
    print(f"Time:       {sol.meta['runtime']:.3f}s")
    if "best_rule" in sol.meta:
        print(f"Best rule:  order={sol.meta['best_rule'][0]}, sku={sol.meta['best_rule'][1]}")
    if violations:
        print(f"VALIDATION FAILED ({len(violations)} violations)")
        for v in violations[:10]:
            print(f"  - {v}")
    else:
        print("Validation PASSED")


def _save(sol, inst, violations, params: dict, out_dir: Path, instance_src: str) -> Path:
    """Write pickle, summary and HTML plot (same content as precalculate_instances_for_lns)."""
    from schedule_visualizer import plot_schedule, write_html

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _free_stem(out_dir, sol.meta["name"])

    with open(out_dir / f"{stem}_solution.pkl", "wb") as f:
        pickle.dump(sol, f)

    with open(out_dir / f"{stem}_summary.txt", "w") as f:
        # first line in the same format as heuristic_summary.txt
        f.write(
            f"{sol.meta['name']} Result | Feasible: {sol.feasible} | Makespan: {sol.makespan} | "
            f"BinEvents: {sol.total_moves // 2} | Time: {sol.meta['runtime']:.3f}s\n"
        )
        f.write(f"Instance: {instance_src}\n")
        f.write(f"Params: {json.dumps(params, default=str)}\n")
        f.write(f"Validation: {'PASSED' if not violations else f'FAILED ({len(violations)} violations)'}\n")
        for v in violations[:50]:
            f.write(f"  - {v}\n")

    fig = plot_schedule(*sol.to_viz(inst), show=False)
    write_html(fig, str(out_dir / f"{stem}_solution.html"))
    print(f"Saved {stem}_solution.pkl/.html and {stem}_summary.txt to {out_dir}")
    return out_dir / f"{stem}_solution.pkl"


def _run_batch(args, ap: argparse.ArgumentParser) -> None:
    from instance import Instance

    set_flags = [f"--{k}" for k in GEN_DEFAULTS if getattr(args, k) is not None]
    if args.param or set_flags or args.instance:
        ap.error(
            "with positional instance paths the heuristic params come from each instance's config.json; "
            f"drop {', '.join((['--param'] if args.param else []) + set_flags + (['--instance'] if args.instance else []))}"
        )
    if args.vis:
        ap.error("--vis is not supported for batch runs; use --save to write the HTML plots")

    results, skipped, failed = [], [], []
    for p in args.instances:
        pkl = _resolve_instance_path(p)
        if pkl is None:
            skipped.append(p)
            continue
        print(f"\n### {pkl.parent.name}")
        try:
            inst = Instance.from_pickle(str(pkl))
            params = _config_params(pkl)
            print(f"Stations={len(inst.S)} Lanes={len(inst.L)} Orders={len(inst.O)} "
                  f"MoveCap={inst.movecap} PickCap={inst.pickcap} Params={params}")
            sol = run_heuristic(args.name, inst, **params)
            if sol is None:
                print("No feasible solution.")
                failed.append((p, "no solution"))
                continue
            violations = sol.validate(inst, params["horizon"])
            _report(sol, violations)
            if args.save:
                out_dir = Path(args.out_dir) / pkl.parent.name if args.out_dir else pkl.parent
                _save(sol, inst, violations, params, out_dir, str(pkl))
            results.append((pkl.parent.name, sol, len(violations)))
        except Exception as e:  # keep going over the rest of the batch
            print(f"[!] {type(e).__name__}: {e}")
            failed.append((p, f"{type(e).__name__}: {e}"))

    print(f"\n=== Batch summary: {resolve(args.name)} on {len(results)} instance(s) ===")
    for name, sol, nviol in results:
        status = "OK" if nviol == 0 else f"{nviol} violations"
        print(f"{name:<45} makespan={sol.makespan:<8} bins={sol.total_moves // 2:<7} "
              f"time={sol.meta['runtime']:.3f}s  {status}")
    if skipped:
        print(f"Skipped (no instance.pkl): {', '.join(skipped)}")
    for p, why in failed:
        print(f"FAILED {p}: {why}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a heuristic from the heuristics registry.")
    ap.add_argument("name", nargs="?", help="registry name or alias (see --list)")
    ap.add_argument(
        "instances", nargs="*",
        help="precalculated instance folders (or instance.pkl files) for a batch run; "
             "params are taken from each folder's config.json",
    )
    ap.add_argument("--list", action="store_true", help="list registered heuristics and exit")
    ap.add_argument("--instance", default=None, help="path to a pickled Instance (overrides generation)")
    ap.add_argument("--stations", type=int, default=None, help=f"default {GEN_DEFAULTS['stations']}")
    ap.add_argument("--lanes", type=int, default=None, help=f"default {GEN_DEFAULTS['lanes']}")
    ap.add_argument("--orders", type=int, default=None, help=f"default {GEN_DEFAULTS['orders']}")
    ap.add_argument("--skus", type=int, default=None, help=f"default {GEN_DEFAULTS['skus']}")
    ap.add_argument("--seed", type=int, default=None, help=f"default {GEN_DEFAULTS['seed']}")
    ap.add_argument("--pick", type=int, default=None, help=f"default {GEN_DEFAULTS['pick']}")
    ap.add_argument("--movecap", type=int, default=None, help=f"default {GEN_DEFAULTS['movecap']}")
    ap.add_argument("--horizon", type=int, default=None, help=f"default {GEN_DEFAULTS['horizon']}")
    ap.add_argument(
        "--param", action="append", default=[], metavar="KEY=VALUE",
        help="extra heuristic parameter (JSON value), e.g. --param beta=0.5 --param tiebreaker=sum_rt_asc",
    )
    ap.add_argument("--vis", action="store_true", help="open an HTML Gantt chart of the schedule")
    ap.add_argument(
        "--save", action="store_true",
        help="save solution pickle, HTML plot and text summary (default: next to the instance)",
    )
    ap.add_argument(
        "--out-dir", default=None,
        help="where --save writes; in batch mode a subfolder per instance is created here",
    )
    args = ap.parse_args()

    if args.list or not args.name:
        for name, (fn, defaults) in HEURISTICS.items():
            print(f"{name:<32} {fn.__name__}({', '.join(f'{k}={v!r}' for k, v in defaults.items())})")
        return

    try:
        resolve(args.name)
    except ValueError as e:
        ap.error(str(e))

    if args.instances:
        _run_batch(args, ap)
        return

    for k, v in GEN_DEFAULTS.items():
        if getattr(args, k) is None:
            setattr(args, k, v)

    if args.save and not args.instance and not args.out_dir:
        ap.error("--save on a generated instance needs --out-dir")

    if args.instance:
        from instance import Instance

        inst = Instance.from_pickle(args.instance)
    else:
        from datagen import generate_data

        inst = generate_data(
            num_stations=args.stations,
            lanes_per_station=args.lanes,
            num_orders=args.orders,
            num_skus=args.skus,
            seed=args.seed,
            pick_touch_time=args.pick,
            movecap=args.movecap,
        )
    params = dict(kv.split("=", 1) for kv in args.param)
    params = {k: _parse_value(v) for k, v in params.items()}
    params.setdefault("horizon", args.horizon)

    print(f"Stations={len(inst.S)} Lanes={len(inst.L)} Orders={len(inst.O)} "
          f"MoveCap={inst.movecap} PickCap={inst.pickcap}")
    sol = run_heuristic(args.name, inst, **params)
    if sol is None:
        print("No feasible solution.")
        return
    violations = sol.validate(inst, params["horizon"])
    print()
    _report(sol, violations)

    if args.save:
        out_dir = Path(args.out_dir) if args.out_dir else Path(args.instance).parent
        _save(sol, inst, violations, params, out_dir, args.instance or "generated")

    if args.vis:
        from schedule_visualizer import plot_schedule

        plot_schedule(*sol.to_viz(inst), show=True)


if __name__ == "__main__":
    main()
