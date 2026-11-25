"""Bridge between the warehouse simulator (warehouse-sim, `wsim`) and the static model.

The simulator is the reference ("reality"); the static model is the abstraction under test.

- `dump`: generate simulator instances (`wsim --dump`).
- `calibrate`: run the simulator's own heuristic on an instance with cycle logging and fit
  the robot time before a lift (drive to the stack, dig, lift) on the number of bins above.
- `static_instance`: the static `Instance` of a simulator instance:
    S = the simulator's stations (in `Ops::stations` order), L = its lanes, orders and SKUs 1:1,
    p[k] = the simulator's pick time per line, N[k] = bins of SKU k, MoveCap = robots, PickCap 1,
    rt[k] = fitted (approach + dig + lift) for the SKU's least-covered bin
            + drive from that bin to the ports (mean over stations, the robot motion model)
            + lowering into the port,
    rt_ret[k] = mean robot time of a return in the calibration run (lift at the port, carry, drop).
  Port queueing is left out on purpose: the static model has no port resource, so a fetch
  in it ends exactly when the bin is presented. That gap is part of what is measured.
- `plan_from_solution`: a static schedule as a simulator station plan (`instance::Plan`).
- `simulate`: run a plan (or the simulator's own heuristic) and collect its metrics.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from instance import Instance  # noqa: E402

WSIM = os.environ.get("WSIM", os.path.join(REPO, "warehouse-sim", "core", "target", "release", "wsim"))

# Red Line robot (warehouse-sim/core/src/motion.rs, Spec::RED_LINE)
V_MAX, ACCEL, SWITCH = 3.1, 0.8, 1.25
HOIST, GRIP, RELEASE, BIN_H = 1.6, 0.75, 0.5, 0.33


def wsim(args: list[str], **kw) -> subprocess.CompletedProcess:
    r = subprocess.run([WSIM, "run", *args], capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"wsim {' '.join(args)} failed ({r.returncode}):\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r


def sets(params: dict) -> list[str]:
    out = []
    for k, v in params.items():
        out += ["--set", f"{k}={json.dumps(v) if not isinstance(v, str) else v}"]
    return out


# ---------------------------------------------------------------------------------------- instances

def dump(out: str, layout_args: list[str], params: dict, seeds: list[int]) -> list[str]:
    """Generate simulator instances into `out`; returns the instance file paths."""
    os.makedirs(out, exist_ok=True)
    seed_arg = ",".join(str(s) for s in seeds)
    wsim([*layout_args, *sets(params), "--seeds", seed_arg, "--dump", out])
    return [os.path.join(out, f"instance-seed{s}.json") for s in seeds]


def run_time(cells: int, pitch: float) -> float:
    """Rest-to-rest trapezoidal run along one axis (motion.rs)."""
    d = cells * pitch
    if d <= 0:
        return 0.0
    if d <= V_MAX * V_MAX / ACCEL:
        return 2.0 * math.sqrt(d / ACCEL)
    return d / V_MAX + V_MAX / ACCEL


def drive_time(a: tuple[int, int], b: tuple[int, int], geo: dict) -> float:
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return run_time(dx, geo["cellW"]) + run_time(dy, geo["cellL"]) + (SWITCH if dx and dy else 0.0)


def port_op_time(geo: dict, grip: bool) -> float:
    """Hoist time at a port (level 0 of the grid): lowering (release) or lifting (grip) a bin."""
    mv = geo["gridHeight"] * BIN_H / HOIST
    return 2.0 * mv + (GRIP if grip else RELEASE)


def bin_positions(sim: dict, geo: dict) -> dict[int, tuple[tuple[int, int], int]]:
    """bin -> ((x, y) of its column, bins above it) at the start."""
    w = geo["width"]
    pos = {}
    for col, stack in enumerate(sim["stacks"]):
        for lvl, b in enumerate(stack):
            pos[b] = ((col % w, col // w), len(stack) - 1 - lvl)
    return pos


def calibrate(inst_path: str, layout_args: list[str], out: str, params: dict | None = None) -> dict:
    """Run the simulator's own heuristic on the instance and fit the robot time before a lift."""
    os.makedirs(out, exist_ok=True)
    r = wsim([*layout_args, "--instance", inst_path, *sets(params or {}), "--until-done",
              "--cycles", out, "--json", os.path.join(out, "metrics.json")])
    m = json.load(open(os.path.join(out, "metrics.json")))[0]
    seed = m["seed"]
    d = pd.read_csv(os.path.join(out, f"cycles-seed{seed}.csv"))
    d = d[(d.at_port > 0) & (d.lifted > 0)]
    to_lift = d.lifted - d.assigned
    X = np.c_[np.ones(len(d)), d.above.values.astype(float)]
    with np.errstate(all="ignore"):  # spurious matmul warnings of numpy on macOS Accelerate
        coef, *_ = np.linalg.lstsq(X, to_lift.values.astype(float), rcond=None)
        res = to_lift.values - X @ coef
    ret = d[(d.returned > 0) & (d.ret_lift > 0)]
    ret_busy = (ret.returned - ret.ret_lift).mean()
    to_port = d.at_port - d.lifted
    return dict(lift_a=float(coef[0]), lift_b=float(coef[1]), lift_r2=float(1 - res.var() / to_lift.var()),
                ret_busy=float(ret_busy), to_port_mean=float(to_port.mean()), fetch_busy_mean=float((d.at_port - d.assigned).mean()),
                cal_makespan=m["ops"]["makespan"], cal_stdout=r.stdout.strip())


def static_instance(sim: dict, geo: dict, cal: dict, seed: int) -> tuple[Instance, list[int], dict]:
    """Static Instance of a simulator instance (see the module docstring), the simulator SKU of each
    static SKU (the static model numbers the ordered SKUs 0..K-1), and the per-SKU estimates."""
    p_ = sim["params"]
    sim_skus = sorted({k for skus in sim["orders"] for k in skus})
    to_static = {k: i for i, k in enumerate(sim_skus)}
    orders = {o: [to_static[k] for k in skus] for o, skus in enumerate(sim["orders"])}
    K = list(range(len(sim_skus)))
    ports = [tuple(s["port"]) for s in geo["stations"]]
    pos = bin_positions(sim, geo)
    sku_bins: dict[int, list[int]] = {}
    for b, k in enumerate(sim["binSku"]):
        sku_bins.setdefault(k, []).append(b)
    lower = port_op_time(geo, grip=False)
    rt, rt_ret, p, N, est = {}, {}, {}, {}, {}
    for k in K:
        bins = sku_bins[sim_skus[k]]
        # the bin the simulator would take: fewest above, then nearest the ports (on average)
        def score(b):
            (xy, above) = pos[b]
            return above, np.mean([drive_time(xy, q, geo) for q in ports])
        best = min(bins, key=score)
        above, drive = score(best)
        t = cal["lift_a"] + cal["lift_b"] * above + drive + lower
        rt[k] = max(1, round(t))
        rt_ret[k] = max(1, round(cal["ret_busy"]))
        p[k] = max(1, round(sim["pickTime"][sim_skus[k]]))
        N[k] = len(bins)
        est[k] = dict(above=above, drive=drive)
    S = list(range(len(geo["stations"])))
    L = list(range(p_["lanes"]))
    inst = Instance(S, L, K, orders, rt, p, N, movecap=p_["robots"], seed=seed, rt_ret=rt_ret, pickcap=1)
    return inst, sim_skus, est


# ---------------------------------------------------------------------------------------- plans

def plan_from_solution(sol, name: str, sim_skus: list[int]) -> dict:
    """Station plan of a static schedule: a HeuristicSolution, or a CP variable dict {name: {present, start, end}}.
    `sim_skus` maps static SKUs back to the simulator's (see `static_instance`)."""
    if hasattr(sol, "bin_events"):
        stations = sorted(set(sol.bin_events) | {a[0] for a in sol.order_assignments.values()})
        orders = {s: sorted((a[2], o) for o, a in sol.order_assignments.items() if a[0] == s) for s in stations}
        visits = {s: [] for s in stations}
        for s, evs in sol.bin_events.items():
            for e in evs:
                served = [o for o in e.orders_served if (o, s, e.sku) in sol.pick_events]
                first = min((sol.pick_events[(o, s, e.sku)][0] for o in served), default=e.presence_start)
                visits[s].append((e.presence_start, first, e.sku, sorted(served, key=lambda o: sol.pick_events[(o, s, e.sku)][0])))
    else:
        vs = sol
        parse = lambda n: tuple(int(x) for x in n[n.index("[") + 1:-1].split(","))  # noqa: E731
        orders, visits, picks = {}, {}, {}
        for n, v in vs.items():
            if not v.get("present"):
                continue
            if n.startswith("I_os["):
                o, s = parse(n)
                orders.setdefault(s, []).append((v["start"], o))
            elif n.startswith("P["):
                o, s, k, e = parse(n)
                picks.setdefault((s, k, e), []).append((v["start"], o))
        for n, v in vs.items():
            if n.startswith("B[") and v.get("present"):
                s, k, e = parse(n)
                served = sorted(picks.get((s, k, e), []))
                if not served:
                    continue  # an empty visit (no picks): nothing to fetch
                visits.setdefault(s, []).append((v["start"], served[0][0], k, [o for _, o in served]))
        stations = sorted(set(orders) | set(visits))
        for s in stations:
            orders[s] = sorted(orders.get(s, []))
    n_st = max(stations) + 1 if stations else 0
    out = {"name": name, "stations": []}
    for s in range(n_st):
        vis = sorted(visits.get(s, []), key=lambda x: (x[0], x[1], x[2]))
        out["stations"].append({
            "orders": [o for _, o in sorted(orders.get(s, []))],
            "visits": [{"sku": int(sim_skus[k]), "orders": [int(o) for o in os_]} for _, _, k, os_ in vis if os_],
        })
    return out


def pad_plan(plan: dict, n_stations: int) -> dict:
    while len(plan["stations"]) < n_stations:
        plan["stations"].append({"orders": [], "visits": []})
    return plan


# ---------------------------------------------------------------------------------------- simulation

def simulate(inst_path: str, layout_args: list[str], plan: dict | None = None, params: dict | None = None,
             cycles_dir: str | None = None, max_time: float = 2e5) -> dict:
    """Run the simulator on an instance (following `plan` if given). Returns its metrics."""
    with tempfile.TemporaryDirectory() as tmp:
        args = [*layout_args, "--instance", inst_path, *sets(params or {}), "--until-done", "--time", str(max_time),
                "--json", os.path.join(tmp, "m.json")]
        if plan is not None:
            pf = os.path.join(tmp, "plan.json")
            json.dump(plan, open(pf, "w"))
            args += ["--plan", pf]
        if cycles_dir:
            args += ["--cycles", cycles_dir]
        wsim(args)
        m = json.load(open(os.path.join(tmp, "m.json")))[0]
    ops, rt = m["ops"], m["ops"]["robotTime"]
    busy = sum(rt[c] for c in ("drive", "traffic", "hoist", "port", "queue"))
    return dict(
        sim_done=m["done"], sim_makespan=ops["makespan"] if m["done"] else float("nan"), sim_time=m["simTime"],
        sim_lines=m["lines"], sim_lines_h=m["linesPerHour"], sim_fetches=ops["fetches"],
        sim_robot_busy=busy, sim_robot_queue=rt["queue"], sim_robot_traffic=rt["traffic"], sim_robot_drive=rt["drive"],
        sim_robot_idle_pct=100 * m["robotIdle"], sim_picker_idle_pct=100 * m["pickerIdle"],
        sim_port_busy_pct=100 * m["portBusy"], sim_port_change_pct=100 * m["portChange"],
        sim_dug_per_fetch=m["dugPerFetch"], sim_errors=m["errors"], sim_problems=" | ".join(m["firstProblems"][:3]),
        sim_missed_shares=ops["missedShares"],
    )
