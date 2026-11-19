"""Export an LNS experiment as a self-contained interactive replay (HTML).

Reads what VisualLogger already writes to an experiment folder (solutions.npz,
experiment_dataframe.pkl, barcode_run_<k>.npz, run_<k>.log) and bakes a compact
trace into a copy of lns_replay_viewer.html. Works on any existing experiment:

    python lns_replay.py precalculated_instances/<inst>/experiments/<timestamp>
    python lns_replay.py <exp_dir> --runs 1 3 -o replay.html

Frames: frame 0 is the initial solution, frame f is the state after LNS iteration
f-1 (dataframe row f-1, barcode row f-1, deltas[f-1]).

The trace also carries KPIs recomputed from the replayed state (makespan = max end
of I_os, bin fetches = #present F, flow = sum of I_os ends). A mismatch against the
logged KPIs means the delta log missed some variable changes, so the replay (and
anything else built on solutions.npz) diverges from the real search state.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import result_store

VIEWER_TEMPLATE = Path(__file__).with_name("lns_replay_viewer.html")
TRACE_PLACEHOLDER = "/*__LNS_TRACE__*/null"

# Variable types drawn by the viewer. I_os (duplicate of I_os_lane) and Block
# (span of F..R) are only used for the KPI replay check.
TYPE_CODES = {"I_os_lane": 0, "C": 1, "P": 2, "B": 3, "F": 4, "R": 5}

STATUS_NAMES = {
    1: "Unknown",
    2: "Optimal_No_Improve",
    3: "Optimal_New_Best",
    4: "Optimal_Improve",
    5: "Feasible_No_Improve",
    6: "Feasible_New_Best",
    7: "Feasible_Improve",
    8: "Feasible_Degradation",
    9: "Infeasible",
}

_NAME_RE = re.compile(r"^([A-Za-z_0-9]+)\[(.*)\]$")


def _parse(name):
    m = _NAME_RE.match(name)
    if not m:
        return name, ()
    return m.group(1), tuple(int(x) for x in m.group(2).split(","))


def _fields(vtype, ix):
    """(station, order, sku, copy, lane) for a drawn variable; -1 = n/a."""
    if vtype == "I_os_lane":  # [o, s, lane]
        return ix[1], ix[0], -1, -1, ix[2]
    if vtype == "C":  # [o, k, s]
        return ix[2], ix[0], ix[1], -1, -1
    if vtype == "P":  # [o, s, k, e]
        return ix[1], ix[0], ix[2], ix[3], -1
    return ix[0], -1, ix[1], ix[2], -1  # B / F / R: [s, k, e]


def _barcode_order(initial_solution):
    """Column order of the barcode: cp_model_utils.sort_variables (start, name)."""
    def key(n):
        v = initial_solution[n]
        pres = isinstance(v, dict) and v.get("present") and v.get("start") is not None
        return (v["start"] if pres else math.inf, n)

    return sorted(initial_solution, key=key)


def _ranges(sorted_idx):
    """[a0, len0, a1, len1, ...] run-length encoding of a sorted index array."""
    if len(sorted_idx) == 0:
        return []
    idx = np.asarray(sorted_idx, dtype=np.int64)
    breaks = np.flatnonzero(np.diff(idx) != 1) + 1
    starts = np.concatenate([[0], breaks])
    ends = np.concatenate([breaks, [len(idx)]])
    out = np.empty(2 * len(starts), dtype=np.int64)
    out[0::2] = idx[starts]
    out[1::2] = ends - starts
    return out.tolist()


def _clean(arr, ndigits=None):
    out = []
    for x in arr:
        if x is None:
            out.append(None)
            continue
        x = float(x)
        if math.isnan(x) or x < 0:
            out.append(None)
        else:
            out.append(round(x, ndigits) if ndigits is not None else (int(x) if x.is_integer() else x))
    return out


def _strategy_names_from_logs(exp_dir):
    names = {}
    pat = re.compile(r"Strategy: (\S+)\s*\((\d+)\)")
    for log in Path(exp_dir).glob("run_*.log"):
        with open(log, errors="replace") as f:
            for line in f:
                m = pat.search(line)
                if m:
                    names[int(m.group(2))] = m.group(1)
    return names


def _instance_meta(instance_dir):
    """Scalar entries of the instance config.json (the file also holds big arrays)."""
    path = Path(instance_dir) / "config.json"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool))}


def _cp_reference(instance_dir, time_ticks):
    """Mean CP best-so-far (over seeds) sampled at the LNS cumulative wall time."""
    files = result_store.find_all(os.path.join(str(instance_dir), "cp_intermediate_records_*"))
    trajs = []
    for fpath in files:
        try:
            recs = result_store.load(fpath)
        except (OSError, KeyError, ValueError, IndexError, pickle.UnpicklingError):
            continue
        if not recs:
            continue
        times = np.array([r["time"] for r in recs], dtype=float)
        bests = np.array([r["best"] for r in recs], dtype=float)
        pos = np.searchsorted(times, time_ticks, side="right") - 1
        trajs.append(np.where(pos >= 0, bests[np.clip(pos, 0, None)], np.nan))
    if not trajs:
        return None
    with warnings.catch_warnings():  # ticks before any seed's first solution -> NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(np.vstack(trajs), axis=0)
    return _clean(mean, 1)


def build_run_trace(initial_solution, deltas, barcode, df_run, instance_dir=None,
                    cp_objective="makespan"):
    """Compact, JSON-ready trace of a single LNS run."""
    init = {
        k: v for k, v in initial_solution.items() if isinstance(v, dict)
    }
    n_iters = len(deltas)

    # --- variables that are ever present and drawn ---
    ever = {k for k, v in init.items() if v.get("present")}
    for d in deltas:
        ever.update(k for k, v in d.items() if isinstance(v, dict) and v.get("present"))

    parsed = {}
    for name in ever:
        vtype, ix = _parse(name)
        if vtype in TYPE_CODES and ix:
            parsed[name] = (vtype, ix)

    def sort_key(name):
        vtype, ix = parsed[name]
        st, o, k, e, ln = _fields(vtype, ix)
        return (TYPE_CODES[vtype], st, o, k, e, ln)

    names = sorted(parsed, key=sort_key)
    vidx = {n: i for i, n in enumerate(names)}
    cols = {"t": [], "s": [], "o": [], "k": [], "e": [], "l": []}
    for n in names:
        vtype, ix = parsed[n]
        st, o, k, e, ln = _fields(vtype, ix)
        cols["t"].append(TYPE_CODES[vtype])
        cols["s"].append(st)
        cols["o"].append(o)
        cols["k"].append(k)
        cols["e"].append(e)
        cols["l"].append(ln)

    init_se = []
    for n in names:
        v = init.get(n, {})
        if v.get("present"):
            init_se += [int(v["start"]), int(v["end"])]
        else:
            init_se += [-1, -1]

    delta_flat = []
    for d in deltas:
        row = []
        for n, v in d.items():
            i = vidx.get(n)
            if i is None or not isinstance(v, dict):
                continue
            if v.get("present"):
                row += [i, int(v["start"]), int(v["end"])]
            else:
                row += [i, -1, -1]
        delta_flat.append(row)

    # --- neighbourhood (freed set) per iteration, in drawn-variable indices ---
    freed = [[] for _ in range(n_iters)]
    if barcode is not None:
        order = _barcode_order(init)
        if len(order) == barcode.shape[1]:
            col_of = np.array([vidx.get(n, -1) for n in order], dtype=np.int64)
            keep = col_of >= 0
            sub = barcode[:, keep]
            mapped = col_of[keep]
            for it in range(min(n_iters, sub.shape[0])):
                cols_freed = np.flatnonzero(sub[it] != -1)
                freed[it] = _ranges(np.sort(mapped[cols_freed]))

    # --- KPI replay check (uses I_os / F, including vars not drawn) ---
    os_end = {k: v["end"] for k, v in init.items() if k.startswith("I_os[") and v.get("present")}
    fetch_present = {k for k, v in init.items() if k.startswith("F[") and v.get("present")}
    rep_ms, rep_fe, rep_fl = [], [], []

    def snap():
        rep_ms.append(max(os_end.values()) if os_end else None)
        rep_fe.append(len(fetch_present))
        rep_fl.append(sum(os_end.values()) if os_end else None)

    snap()
    for d in deltas:
        for k, v in d.items():
            if k.startswith("I_os["):
                if v.get("present"):
                    os_end[k] = v["end"]
                else:
                    os_end.pop(k, None)
            elif k.startswith("F["):
                if v.get("present"):
                    fetch_present.add(k)
                else:
                    fetch_present.discard(k)
        snap()

    df_run = df_run.sort_values("iteration").reset_index(drop=True).iloc[:n_iters]
    iter_time = df_run["total_iter_time"].to_numpy(dtype=float)
    t_cum = np.cumsum(np.where(iter_time > 0, iter_time, 0.0))

    def col(name, nd=None):
        return _clean(df_run[name].tolist(), nd) if name in df_run else None

    kpi = {
        "makespan": col("current_makespan"),
        "best_makespan": col("best_makespan"),
        "fetches": col("current_bin_fetches"),
        "best_fetches": col("best_bin_fetches"),
        "flow": col("current_total_flow_time"),
        "best_flow": col("best_total_flow_time"),
    }

    mismatch = {}
    for key, rep in (("makespan", rep_ms), ("fetches", rep_fe), ("flow", rep_fl)):
        logged = kpi[key] or []
        bad = [
            f for f in range(1, n_iters + 1)
            if f - 1 < len(logged) and logged[f - 1] is not None
            and rep[f] is not None and abs(rep[f] - logged[f - 1]) > 0.5
        ]
        mismatch[key] = bad

    cp_ref = None
    if instance_dir is not None and cp_objective == "makespan" and n_iters:
        cp_ref = _cp_reference(instance_dir, t_cum)

    return {
        "nIters": n_iters,
        "vars": cols,
        "init": init_se,
        "deltas": delta_flat,
        "freed": freed,
        "numOptimized": [int(x) for x in df_run["num_optimized"].tolist()],
        "statuses": [int(x) for x in df_run["statuses"].tolist()],
        "strategies": [int(x) for x in df_run["strategy"].tolist()],
        "cpObj": [str(x) if x else "" for x in df_run["cp_obj"].tolist()],
        "iterTime": _clean(iter_time.tolist(), 3),
        "timeCum": [round(float(x), 2) for x in t_cum],
        "kpi": kpi,
        "replay": {"makespan": rep_ms, "fetches": rep_fe, "flow": rep_fl},
        "mismatch": mismatch,
        "cpRef": cp_ref,
    }


def build_trace(exp_dir, runs=None, strategy_names=None):
    exp_dir = Path(exp_dir)
    instance_dir = exp_dir.parent.parent
    payload = result_store.load(str(exp_dir / "solutions.npz"))
    run_list = payload if isinstance(payload, list) else [payload]
    df = pd.read_pickle(exp_dir / "experiment_dataframe.pkl")
    cfg = {}
    if (exp_dir / "experiment_config.json").exists():
        with open(exp_dir / "experiment_config.json") as f:
            cfg = json.load(f)
    strat = dict(_strategy_names_from_logs(exp_dir))
    if strategy_names:
        strat.update({int(k): v for k, v in strategy_names.items()})

    out_runs = []
    for r_pos, run in enumerate(run_list):
        run_id = int(run.get("run_id", r_pos + 1))
        if runs and run_id not in runs:
            continue
        bc_path = exp_dir / f"barcode_run_{run_id}.npz"
        barcode = result_store.load_barcode(str(bc_path)) if bc_path.exists() else None
        trace = build_run_trace(
            run["initial_solution"], run.get("deltas", []), barcode,
            df[df["run_id"] == run_id], instance_dir,
            cfg.get("cp_objective", "makespan"),
        )
        trace["runId"] = run_id
        out_runs.append(trace)

    inst = _instance_meta(instance_dir)
    return {
        "version": 1,
        "experiment": exp_dir.name,
        "instance": instance_dir.name,
        "instanceMeta": inst,
        "config": {k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool)) or v is None},
        "strategyNames": {str(k): v for k, v in sorted(strat.items())},
        "statusNames": {str(k): v for k, v in STATUS_NAMES.items()},
        "runs": out_runs,
    }


def write_viewer(trace, out_path):
    html = VIEWER_TEMPLATE.read_text()
    if TRACE_PLACEHOLDER not in html:
        raise RuntimeError(f"Placeholder {TRACE_PLACEHOLDER!r} missing from {VIEWER_TEMPLATE}")
    data = json.dumps(trace, separators=(",", ":")).replace("</", "<\\/")
    Path(out_path).write_text(html.replace(TRACE_PLACEHOLDER, data, 1))
    return str(out_path)


def export_experiment(exp_dir, out_path=None, runs=None, strategy_names=None):
    """Build the trace for an experiment folder and write the replay HTML."""
    trace = build_trace(exp_dir, runs=runs, strategy_names=strategy_names)
    out_path = out_path or Path(exp_dir) / "lns_replay.html"
    write_viewer(trace, out_path)
    return str(out_path), trace


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp_dir", help="experiment folder (…/experiments/<timestamp>)")
    ap.add_argument("--runs", type=int, nargs="*", help="run ids to include (default: all)")
    ap.add_argument("-o", "--out", help="output HTML (default: <exp_dir>/lns_replay.html)")
    ap.add_argument("--json", help="also write the raw trace JSON here")
    args = ap.parse_args()

    out, trace = export_experiment(args.exp_dir, args.out, args.runs)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(trace, f, separators=(",", ":"))
    size = os.path.getsize(out) / 1e6
    print(f"Wrote {out} ({size:.1f} MB, {len(trace['runs'])} run(s))")
    for r in trace["runs"]:
        bad = {k: len(v) for k, v in r["mismatch"].items() if v}
        msg = ", ".join(f"{k}: {n} iters (first @ iter {r['mismatch'][k][0] - 1})" for k, n in bad.items())
        print(f"  run {r['runId']}: {r['nIters']} iters, {len(r['vars']['t'])} drawn vars"
              + (f" | replay ≠ logged KPI — {msg}" if bad else " | replay matches logged KPIs"))


if __name__ == "__main__":
    main()
