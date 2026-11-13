"""Structure of final solutions: utilisation, imbalance, bin fetches.

Every method's final schedule is reduced to two lists:
    visits  (station, sku)            one per bin visit (fetch -> presence -> return)
    picks   (station, sku, start, end) one per order line

Sources (the `solution` column of the runs table, see data.py):
    heuristics  heuristic_<name>_solution.pkl   HeuristicSolution
    CP          cp_solution_<k>.pkl             CppSolveResult (var_sols by name)
    LNS         <exp>/run<k>-solution.pkl       dict {var name: {present, start, end}}

Metrics of one solution (C = makespan, S = stations, W = robot work):
    fetches            number of bin visits
    fetch_pick_ratio   fetches / order lines (1 = every line has its own bin visit)
    refetch_pct        visits of a SKU that was already fetched before, any station,
                       in % of fetches: 100 (fetches - distinct SKUs) / fetches
    same_station_refetch_pct
                       the same, but only repeat visits of a SKU to the same station
    robot_work         W = sum over visits of (rt + rt_ret) of the SKU [ticks]
    robot_util_pct     100 W / (MoveCap C): share of robot capacity busy over the makespan
    picker_util_pct    100 sum(pick times) / (S pickcap C)
    workload_imb_pct   100 (max_s - mean_s) / mean_s of the pick time per station
    makespan_imb_pct   100 (C - min_s C_s) / C, C_s = end of the last pick at station s
    station_idle_pct   100 mean_s (C - C_s) / C: average idle tail of a station
    unused_stations    stations without any pick (they count as C_s = 0 above)
"""

import os
import pickle
import warnings


import numpy as np
import pandas as pd

METRICS = {  # column -> (label, unit)
    "fetch_pick_ratio": ("Fetch / pick ratio", ""),
    "refetch_pct": ("Refetches", "%"),
    "same_station_refetch_pct": ("Same-station refetches", "%"),
    "robot_util_pct": ("Robot utilisation", "%"),
    "picker_util_pct": ("Picker utilisation", "%"),
    "workload_imb_pct": ("Workload imbalance", "%"),
    "makespan_imb_pct": ("Makespan imbalance", "%"),
    "station_idle_pct": ("Station idle tail", "%"),
    "unused_stations": ("Unused stations", ""),
}
CACHE_COLS = ["solution", "mtime", "size_bytes"]


def _parse_name(name):
    i = name.index("[")
    return name[:i], tuple(int(x) for x in name[i + 1:-1].split(","))


def _from_var_dict(vs):
    visits, picks = [], []
    for name, v in vs.items():
        if not v.get("present") or name[0] not in "BP" or name.startswith("Block"):
            continue
        kind, idx = _parse_name(name)
        if kind == "B":
            visits.append((idx[0], idx[1]))
        elif kind == "P":
            picks.append((idx[1], idx[2], v["start"], v["end"]))
    return visits, picks


def _from_heuristic(sol):
    visits = [(s, e.sku) for s, evs in sol.bin_events.items() for e in evs]
    picks = [(s, k, a, b) for (o, s, k), (a, b) in sol.pick_events.items()]
    return visits, picks


def read_solution(path):
    """(visits, picks) of a solution pickle of any supported kind."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if hasattr(obj, "bin_events"):
        return _from_heuristic(obj)
    if hasattr(obj, "var_sols"):
        return _from_var_dict(obj.var_sols)
    if isinstance(obj, dict):
        return _from_var_dict(obj)
    raise TypeError(f"{path}: unknown solution type {type(obj).__name__}")


def metrics(visits, picks, inst):
    if not picks:
        return {}
    S = list(inst.S)
    C = max(b for _, _, _, b in picks)
    rt2 = {k: inst.rt[k] + inst.rt_ret[k] for k in {k for _, k in visits}}
    work = sum(rt2[k] for _, k in visits)
    n = len(visits)
    station_pick = {s: 0.0 for s in S}
    station_end = {s: 0.0 for s in S}
    for s, _, a, b in picks:
        station_pick[s] += b - a
        station_end[s] = max(station_end[s], b)
    load = np.array([station_pick[s] for s in S])
    ends = np.array([station_end[s] for s in S])
    pickcap = getattr(inst, "pickcap", 1) or 1
    return dict(
        makespan=C, fetches=n, lines=len(picks),
        fetch_pick_ratio=n / len(picks),
        refetch_pct=100 * (n - len({k for _, k in visits})) / n,
        same_station_refetch_pct=100 * (n - len(set(visits))) / n,
        robot_work=work,
        robot_util_pct=100 * work / (inst.movecap * C),
        picker_util_pct=100 * load.sum() / (len(S) * pickcap * C),
        workload_imb_pct=100 * (load.max() - load.mean()) / load.mean(),
        makespan_imb_pct=100 * (C - ends.min()) / C,
        station_idle_pct=100 * np.mean(C - ends) / C,
        unused_stations=int((load == 0).sum()),
    )


def _stamp(path):
    st = os.stat(path)
    return st.st_mtime, st.st_size


def solution_metrics(runs, instance_dirs, cache=None):
    """Metrics for every run with an existing solution file, one row per run.

    runs: the runs table (needs method, family, instance, run, final, solution).
    instance_dirs: {instance name: folder} for loading instance.pkl.
    cache: CSV path; rows whose file is unchanged (mtime, size) are reused.
    """
    old = pd.DataFrame()
    if cache and os.path.isfile(cache):
        old = pd.read_csv(cache).set_index("solution")
        if not set(METRICS).issubset(old.columns):  # written by an older version
            old = pd.DataFrame()
    rows, insts, n_new = [], {}, 0
    keep = ["method", "family", "instance", "size", "run", "final", "solution"]
    for r in runs[[c for c in keep if c in runs.columns]].itertuples(index=False):
        path = getattr(r, "solution", None)
        if not isinstance(path, str) or not os.path.isfile(path):
            continue
        mtime, size = _stamp(path)
        base = r._asdict()
        if path in old.index and old.loc[path, "mtime"] == mtime and \
                old.loc[path, "size_bytes"] == size:
            m = old.loc[path, ["makespan", "fetches", "lines", "robot_work", *METRICS]].to_dict()
        else:
            if r.instance not in insts:
                with open(os.path.join(instance_dirs[r.instance], "instance.pkl"), "rb") as f:
                    insts[r.instance] = pickle.load(f)
            m = metrics(*read_solution(path), insts[r.instance])
            n_new += 1
            if not m:
                continue
        if abs(m["makespan"] - r.final) > 0.5:
            warnings.warn(f"{path}: makespan {m['makespan']:.0f} != logged final "
                          f"{r.final:.0f}; the solution file is not the final one")
        rows.append({**base, "mtime": mtime, "size_bytes": size, **m})
    df = pd.DataFrame(rows)
    if cache and len(df):
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        df.drop(columns=["size"], errors="ignore").to_csv(cache, index=False)
    if n_new:
        print(f"solution metrics: computed {n_new}, reused {len(df) - n_new}")
    return df


