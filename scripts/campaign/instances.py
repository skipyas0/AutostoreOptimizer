"""Campaign step 1: instance sets as self-contained folders (instance.pkl, instance_summary.txt, config.json).

    .venv/bin/python scripts/campaign/instances.py --out results/campaign/instances            # all sets
    .venv/bin/python scripts/campaign/instances.py --out results/campaign/instances --sets tune

Sets (folder names follow GEN<seed>-<stations>-<lanes>-<skus>-<orders>-<movecap>[-<variant>]):

- ``eval``: the 15 LNS benchmark instances of ``precalculated_instances/`` (GEN1-5 x 5/300, 10/400, 15/600),
  copied. Used only for reporting.
- ``tune``: the same three shapes with generator seeds 6-10. Used for configuration screening, so the
  eval instances never influence the chosen configuration.
- ``scale``: 20 stations / 800 orders, seeds 1-3 (MoveCap 40 like every other set).
- ``regime``: 10/400, seeds 1-5, each regime differing from the eval instances in one knob:
  ``pc2`` / ``pc3`` PickCap 2 / 3, ``mc20`` / ``mc80`` / ``mc160`` MoveCap 20 / 80 / 160 (eval: 40),
  ``b1`` / ``b2`` / ``b4`` at most 1 / 2 / 4 bin copies per SKU (eval: 8; ``b1`` uses the uniform bin model, the
  popularity-correlated one needs at least 2). The heuristics run on all of
  them; LNS confirms ``pc2``, ``mc80`` and ``b2`` (``experiments.py``).

Generation uses the generator settings of the eval instances (``precalculate_instances_for_lns``: no CP
runs, no warm start; starts come from ``starts.py``). Before writing anything, the script regenerates
one eval instance and checks it is identical to the stored pickle, so the generator has not drifted.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pickle
import shutil
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from datagen import generate_data  # noqa: E402
from instance import Instance  # noqa: E402

SHAPES = ((5, 300), (10, 400), (15, 600))
BASE = {"lanes": 3, "symmetry_breaking": True, "skus": 28750, "movecap": 40, "heuristic": "load_balanced_greedy",
        "order_dist": "lognormal", "skew": 1.0, "rt_model": "depth_based", "pick_model": "variable",
        "pick_touch_time": 4, "return_model": "balancing", "bin_model": "popularity_correlated", "max_bins": 8,
        "grid_depth": 16, "cp_seeds": [], "horizon": 10000, "alpha": 1.0, "beta": 1.0, "time_limit": 900,
        "workers": 1, "pickcap": 1}
LNS_REGIMES = ("pc2", "mc80", "b2")
REGIMES = {"pc2": {"pickcap": 2}, "pc3": {"pickcap": 3}, "mc20": {"movecap": 20}, "mc80": {"movecap": 80},
           "mc160": {"movecap": 160}, "b1": {"max_bins": 1, "bin_model": "uniform"}, "b2": {"max_bins": 2}, "b4": {"max_bins": 4}}


def name_of(cfg: dict, variant: str = "") -> str:
    base = f"GEN{cfg['gen_seed']}-{cfg['stations']}-{cfg['lanes']}-{cfg['skus']}-{cfg['orders']}-{cfg['movecap']}"
    return f"{base}-{variant}" if variant else base


def generate(cfg: dict) -> Instance:
    return generate_data(
        num_stations=cfg["stations"], lanes_per_station=cfg["lanes"], num_orders=cfg["orders"],
        num_skus=cfg["skus"], seed=cfg["gen_seed"], movecap=cfg["movecap"], pickcap=cfg["pickcap"],
        pick_touch_time=cfg["pick_touch_time"], order_size_dist=cfg["order_dist"],
        return_model=cfg["return_model"], max_bins_per_sku=cfg["max_bins"], sku_popularity_skew=cfg["skew"],
        retrieval_time_model=cfg["rt_model"], pick_time_model=cfg["pick_model"],
        bin_count_model=cfg["bin_model"], grid_depth=cfg["grid_depth"],
    )


def same_instance(a: Instance, b: Instance) -> bool:
    return pickle.dumps(vars(a), protocol=4) == pickle.dumps(vars(b), protocol=4)


def check_generator() -> None:
    src = os.path.join(REPO, "precalculated_instances", "GEN2-5-3-28750-300-40")
    with open(os.path.join(src, "config.json")) as f:
        stored_cfg = json.load(f)
    cfg = {**BASE, **stored_cfg, "pickcap": 1}
    if not same_instance(generate(cfg), Instance.from_pickle(os.path.join(src, "instance.pkl"))):
        raise SystemExit("generator drift: GEN2-5-3-28750-300-40 regenerated differs from the stored instance")
    print("generator check OK (GEN2-5-3-28750-300-40 reproduced)")


def write(out: str, name: str, inst: Instance, cfg: dict) -> None:
    dst = os.path.join(out, name)
    os.makedirs(dst, exist_ok=True)
    inst.to_pickle(os.path.join(dst, "instance.pkl"))
    with open(os.path.join(dst, "instance_summary.txt"), "w") as f, contextlib.redirect_stdout(f):
        inst.print_summary()
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(cfg, f)


def configs(set_name: str):
    if set_name == "tune":
        for seed in range(6, 11):
            for st, o in SHAPES:
                yield {**BASE, "stations": st, "orders": o, "gen_seed": seed}, ""
    elif set_name == "scale":
        for seed in (1, 2, 3):
            yield {**BASE, "stations": 20, "orders": 800, "gen_seed": seed}, ""
    elif set_name == "regime":
        for variant, knobs in REGIMES.items():
            for seed in range(1, 6):
                yield {**BASE, "stations": 10, "orders": 400, "gen_seed": seed, **knobs}, variant
    else:
        raise ValueError(set_name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(REPO, "results", "campaign", "instances"))
    ap.add_argument("--sets", nargs="+", default=["eval", "tune", "scale", "regime"])
    a = ap.parse_args()
    check_generator()
    os.makedirs(a.out, exist_ok=True)
    index = {}
    for set_name in a.sets:
        names = []
        if set_name == "eval":
            for seed in range(1, 6):
                for st, o in SHAPES:
                    name = f"GEN{seed}-{st}-3-28750-{o}-40"
                    src, dst = os.path.join(REPO, "precalculated_instances", name), os.path.join(a.out, name)
                    os.makedirs(dst, exist_ok=True)
                    for fn in ("instance.pkl", "instance_summary.txt", "config.json"):
                        shutil.copyfile(os.path.join(src, fn), os.path.join(dst, fn))
                    names.append(name)
        else:
            for cfg, variant in configs(set_name):
                name = name_of(cfg, variant)
                if not os.path.exists(os.path.join(a.out, name, "instance.pkl")):
                    write(a.out, name, generate(cfg), cfg)
                names.append(name)
        index[set_name] = names
        print(f"{set_name}: {len(names)} instances")
        if set_name == "regime":  # the subset LNS confirms
            index["regime_lns"] = [n for n in names if n.rsplit("-", 1)[1] in LNS_REGIMES]
    path = os.path.join(a.out, "sets.json")
    if os.path.exists(path):
        with open(path) as f:
            index = {**json.load(f), **index}
    with open(path, "w") as f:
        json.dump(index, f, indent=2)


if __name__ == "__main__":
    main()
