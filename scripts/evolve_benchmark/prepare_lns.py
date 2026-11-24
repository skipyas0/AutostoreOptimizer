"""LNS roots from evolved GBS sequences, in the layout of scripts/warmstart_benchmark (run_all.sh, analyze.py).

    .venv/bin/python scripts/evolve_benchmark/prepare_lns.py --out results/evolve_lns \\
        --start gbs_evolve_work=results/evolve_benchmark_long:ls_work:0 \\
        --start gbs_evolve_makespan=results/evolve_benchmark:ga_makespan:0

Each ``--start NAME=RUNS_DIR:ARM:SEED`` decodes the best sequence stored in ``RUNS_DIR/runs.jsonl``
(``run_gbs_sequence`` with that arm's settings; it checks that C and W match the recorded run) for
every instance of ``--instances`` and writes ``<out>/NAME/precalculated_instances/<instance>/``.
Instance files are *copied* (not symlinked), so the folder can be moved to another machine.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd  # noqa: E402

from evaluation.solutions import metrics  # noqa: E402
from heuristics import validate_solution  # noqa: E402
from heuristics.evolve import robot_work  # noqa: E402
from heuristics.gbs import run_gbs_sequence  # noqa: E402
from instance import Instance  # noqa: E402
from run import ARMS  # noqa: E402

INSTANCES = [f"GEN{s}-{st}-3-28750-{o}-40" for s in range(1, 6) for st, o in ((5, 300), (10, 400))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(REPO, "results", "evolve_lns"))
    ap.add_argument("--source", default=os.path.join(REPO, "precalculated_instances"))
    ap.add_argument("--instances", nargs="+", default=INSTANCES)
    ap.add_argument("--start", action="append", required=True, help="NAME=RUNS_DIR:ARM:SEED")
    a = ap.parse_args()
    rows = []
    for spec in a.start:
        name, rest = spec.split("=", 1)
        runs_dir, arm, seed = rest.rsplit(":", 2)
        kw = {k: v for k, v in ARMS[arm].items() if k in ("window",)}
        if ARMS[arm].get("evolve_bias"):
            raise ValueError("bias genomes are not stored in runs.jsonl; pick an arm without evolve_bias")
        runs = {}
        with open(os.path.join(runs_dir, "runs.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                if r["arm"] == arm and r["seed"] == int(seed):
                    runs[r["instance"]] = r
        for inst_name in a.instances:
            r = runs[inst_name]
            src = os.path.join(a.source, inst_name)
            dst = os.path.join(a.out, name, "precalculated_instances", inst_name)
            os.makedirs(os.path.join(dst, "experiments"), exist_ok=True)
            for fn in ("instance.pkl", "instance_summary.txt"):
                shutil.copyfile(os.path.join(src, fn), os.path.join(dst, fn))
            with open(os.path.join(src, "config.json")) as f:
                config = json.load(f)
            config["heuristic"] = name
            with open(os.path.join(dst, "config.json"), "w") as f:
                json.dump(config, f)
            inst = Instance.from_pickle(os.path.join(src, "instance.pkl"))
            seq = {int(s): v for s, v in r["sequence"].items()}
            sol = run_gbs_sequence(inst, seq, horizon=config["horizon"], **kw)
            if (sol.makespan, robot_work(sol, inst)) != (r["makespan"], r["work"]):
                raise RuntimeError(f"{name} {inst_name}: decoded C/W {sol.makespan}/{robot_work(sol, inst)} "
                                   f"!= recorded {r['makespan']}/{r['work']}")
            viol = validate_solution(sol, inst, horizon=config["horizon"])
            if not sol.feasible or viol:
                raise RuntimeError(f"{name} {inst_name}: invalid warm start {viol[:3]}")
            sol.meta.update({"heuristic": name, "source": spec, "evals": r["evals"]})
            with open(os.path.join(dst, "heuristic_solution.pkl"), "wb") as f:
                pickle.dump(sol, f)
            with open(os.path.join(dst, "heuristic_summary.txt"), "w") as f:
                f.write(f"{name} Result | Feasible: {sol.feasible} | Makespan: {sol.makespan} | "
                        f"BinEvents: {sol.total_moves // 2} | Time: {r['runtime']:.3f}s\n")
            visits = [(s, ev.sku) for s, evs in sol.bin_events.items() for ev in evs]
            picks = [(s, k, x, y) for (o, s, k), (x, y) in sol.pick_events.items()]
            rows.append(dict(start=name, instance=inst_name, runtime=r["runtime"], **metrics(visits, picks, inst)))
            print(f"{name:20s} {inst_name}: C={sol.makespan} W={robot_work(sol, inst)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "warm_starts.csv"), index=False)
    commit = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump({"warm_starts_commit": commit, "starts": [s.split("=")[0] for s in a.start], "sources": a.start,
                   "instances": a.instances, "created": time.strftime("%Y-%m-%d %H:%M")}, f, indent=2)
    print(df.groupby("start", sort=False)[["makespan", "fetches", "robot_work"]].mean().round(1).to_string())


if __name__ == "__main__":
    main()
