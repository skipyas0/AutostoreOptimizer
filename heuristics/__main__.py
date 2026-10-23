"""Run any registered heuristic on a generated (or pickled) instance.

    python -m heuristics rdi --stations 4 --lanes 4 --orders 160 --skus 20000 --movecap 20
    python -m heuristics ama_two_phase --orders 40 --vis
    python -m heuristics cfss --instance precalculated_instances/GEN1-1-1-100-10-2/instance.pkl
    python -m heuristics --list
"""

from __future__ import annotations

import argparse
import json

from heuristics import HEURISTICS, resolve, run_heuristic


def _parse_value(v: str):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a heuristic from the heuristics registry.")
    ap.add_argument("name", nargs="?", help="registry name or alias (see --list)")
    ap.add_argument("--list", action="store_true", help="list registered heuristics and exit")
    ap.add_argument("--instance", default=None, help="path to a pickled Instance (overrides generation)")
    ap.add_argument("--stations", type=int, default=4)
    ap.add_argument("--lanes", type=int, default=4)
    ap.add_argument("--orders", type=int, default=160)
    ap.add_argument("--skus", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pick", type=int, default=4)
    ap.add_argument("--movecap", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=10000)
    ap.add_argument(
        "--param", action="append", default=[], metavar="KEY=VALUE",
        help="extra heuristic parameter (JSON value), e.g. --param beta=0.5 --param tiebreaker=sum_rt_asc",
    )
    ap.add_argument("--vis", action="store_true", help="open an HTML Gantt chart of the schedule")
    args = ap.parse_args()

    if args.list or not args.name:
        for name, (fn, defaults) in HEURISTICS.items():
            print(f"{name:<32} {fn.__name__}({', '.join(f'{k}={v!r}' for k, v in defaults.items())})")
        return

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
    if "partitioned_rdi" not in resolve(args.name):
        params.setdefault("horizon", args.horizon)

    print(f"Stations={len(inst.S)} Lanes={len(inst.L)} Orders={len(inst.O)} "
          f"MoveCap={inst.movecap} PickCap={inst.pickcap}")
    sol = run_heuristic(args.name, inst, **params)
    if sol is None:
        print("No feasible solution.")
        return
    violations = sol.validate(inst, params.get("horizon", args.horizon))
    print(f"\n=== {sol.meta['name']} ===")
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

    if args.vis:
        from schedule_visualizer import plot_schedule

        plot_schedule(*sol.to_viz(inst), show=True)


if __name__ == "__main__":
    main()
