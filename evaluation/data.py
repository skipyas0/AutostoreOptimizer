"""Loaders for an experiment root (a folder of precalculated instance folders).

Every method is loaded into one long table of *runs* with the same columns, so
metrics and plots never need to know where a number came from:

    method   unique id ("LNS-1", "CP", "RDI", ...)
    family   "lns" | "cp" | "heuristic"
    instance, size, run
    t, v     anytime trajectory: times [s] and best makespan found by then
             (np arrays, t ascending, v non-increasing)
    final    best makespan at the end of the run

To add a method, write a loader that returns rows in this format and append them.

Clock conventions (t = 0 is the start of the method's own process):
  LNS        starts from the precomputed start heuristic (value at t = 0); the model
             setup before the first iteration is counted (parsed from run_<k>.log)
  CP         incumbents from cplog_<k>.txt; the clock includes model extraction
  heuristic  a single point at its own runtime
"""

import glob
import json
import os
import re
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

EXP_DATE_FMT = "%d-%m-%Y_%H-%M-%S"

# Config keys introduced after some experiments had run, with the value that
# reproduces the behaviour before the key existed.
CONFIG_DEFAULTS = {
    "combine_strat_with_tail": "all",
}
# Keys that do not change the algorithm.
CONFIG_IGNORE = {"runs", "render_video"}

HEURISTIC_NAMES = {
    "load_balanced_greedy": "LB-SGC",
    "lb_sgc_share": "LB-SGC share",
    "rdi": "RDI",
    "rdi_lb": "RDI-LB",
    "rdi_lb_share": "RDI-LB share",
    "ama_two_phase_share": "AMA two-phase share",
}


# --------------------------------------------------------------------------- #
# Trajectories
# --------------------------------------------------------------------------- #
def make_traj(points):
    """[(t, value), ...] -> (t, running-min v) arrays sorted by time."""
    if not points:
        return np.array([]), np.array([])
    pts = sorted(points)
    t = np.array([p[0] for p in pts], float)
    v = np.minimum.accumulate(np.array([p[1] for p in pts], float))
    return t, v


def values_at(t, v, times):
    """Best value known at each of `times` (NaN before the first solution)."""
    times = np.atleast_1d(np.asarray(times, float))
    if len(t) == 0:
        return np.full(times.shape, np.nan)
    idx = np.searchsorted(t, times, side="right") - 1
    out = np.where(idx >= 0, v[np.clip(idx, 0, None)], np.nan)
    return out


# --------------------------------------------------------------------------- #
# Instances
# --------------------------------------------------------------------------- #
NAME_RE = re.compile(r"GEN(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)")


def instance_dirs(root, pattern="GEN*"):
    return sorted(d for d in glob.glob(os.path.join(root, pattern)) if os.path.isdir(d))


def load_instance_meta(d):
    name = os.path.basename(d)
    cfg_path = os.path.join(d, "config.json")
    cfg = json.load(open(cfg_path)) if os.path.isfile(cfg_path) else {}
    m = NAME_RE.match(name)
    seed, stations, lanes, skus, orders, movecap = map(int, m.groups()) if m else [np.nan] * 6
    return dict(instance=name, gen_seed=cfg.get("gen_seed", seed),
                stations=cfg.get("stations", stations), orders=cfg.get("orders", orders),
                lanes=cfg.get("lanes", lanes), start_heuristic=cfg.get("heuristic"),
                time_limit=cfg.get("time_limit"))


def size_labels(instances):
    """Size class per (stations, orders); Small/Medium/Large when there are three."""
    keys = sorted({(r.stations, r.orders) for r in instances.itertuples()},
                  key=lambda k: (k[1], k[0]))
    if len(keys) == 3:
        names = ["Small", "Medium", "Large"]
    else:
        names = [f"{s} st. / {o} ord." for s, o in keys]
    return {k: n for k, n in zip(keys, names)}, names


# --------------------------------------------------------------------------- #
# Heuristics
# --------------------------------------------------------------------------- #
SUMMARY_RE = re.compile(r"Feasible:\s*(\w+)\s*\|\s*Makespan:\s*([\d.]+).*?Time:\s*([\d.]+)s")


def parse_heuristic_summary(path):
    """(feasible, makespan, runtime, validation_failed)."""
    with open(path) as f:
        text = f.read()
    m = SUMMARY_RE.search(text)
    if not m:
        raise ValueError(f"cannot parse {path}")
    return (m.group(1) == "True", float(m.group(2)), float(m.group(3)),
            "Validation: FAILED" in text)


def heuristic_tag(path, start_heuristic):
    """heuristic_<tag>_summary.txt -> tag; heuristic_summary.txt is the LNS start."""
    base = os.path.basename(path)
    if base == "heuristic_summary.txt":
        return start_heuristic or "start"
    return base[len("heuristic_"):-len("_summary.txt")]


def load_heuristics(d, meta):
    rows = []
    for path in sorted(glob.glob(os.path.join(d, "heuristic*summary.txt"))):
        tag = heuristic_tag(path, meta["start_heuristic"])
        feasible, ms, rt, failed = parse_heuristic_summary(path)
        if not feasible or failed:
            warnings.warn(f"{path}: infeasible or failed validation, skipped")
            continue
        t, v = make_traj([(rt, ms)])
        rows.append(dict(method=HEURISTIC_NAMES.get(tag, tag), family="heuristic",
                         instance=meta["instance"], run=1, t=t, v=v, final=ms,
                         runtime=rt, is_start=os.path.basename(path) == "heuristic_summary.txt"))
    return rows


# --------------------------------------------------------------------------- #
# CP
# --------------------------------------------------------------------------- #
def parse_cp_log(path):
    """(incumbents [(t, obj)], extraction_time, final_obj) from a CP Optimizer log."""
    incumbents, extraction, final = [], np.nan, np.nan
    with open(path) as f:
        for line in f:
            m = re.match(r"\s*\*\s+(\d+)\s+\S+\s+([\d.]+)s", line)
            if m:
                incumbents.append((float(m.group(2)), float(m.group(1))))
                continue
            m = re.search(r"Initial process time.*\(([\d.]+)s extraction", line)
            if m:
                extraction = float(m.group(1))
            m = re.search(r"Best objective\s*:\s*(\d+)", line)
            if m:
                final = float(m.group(1))
    if incumbents and not np.isnan(final) and min(o for _, o in incumbents) != final:
        warnings.warn(f"{path}: best incumbent differs from final objective")
    return incumbents, extraction, final


def load_cp(d, meta, start_value=np.nan):
    """CP runs. A run warm-started from the start heuristic (its first incumbent equals
    the start value) holds that solution from t = 0, like LNS: model extraction and the
    wait for the first incumbent count as solution time, not as "no solution yet"."""
    rows = []
    for log in sorted(glob.glob(os.path.join(d, "cplog_*.txt"))):
        k = int(re.search(r"cplog_(\d+)", log).group(1))
        inc, extraction, final = parse_cp_log(log)
        first_t = inc[0][0] if inc else np.nan
        warm = bool(inc) and inc[0][1] == start_value
        t, v = make_traj(([(0.0, start_value)] if warm else []) + inc)
        rows.append(dict(method="CP", family="cp", instance=meta["instance"], run=k,
                         t=t, v=v, final=v[-1] if len(v) else np.nan,
                         extraction=extraction, first_sol_time=first_t, warm_start=warm))
    return rows


# --------------------------------------------------------------------------- #
# LNS
# --------------------------------------------------------------------------- #
def parse_exp_date(path):
    try:
        return datetime.strptime(os.path.basename(path), EXP_DATE_FMT)
    except ValueError:
        return None


def finished_experiments(d):
    """Timestamp-named experiment folders with a saved config and dataframe."""
    out = []
    for e in glob.glob(os.path.join(d, "experiments", "*")):
        if (parse_exp_date(e) and os.path.isfile(os.path.join(e, "experiment_config.json"))
                and os.path.isfile(os.path.join(e, "experiment_dataframe.pkl"))):
            out.append(e)
    return sorted(out, key=parse_exp_date)


def normalize_config(cfg):
    full = {**CONFIG_DEFAULTS, **cfg}
    return {k: full[k] for k in sorted(full) if k not in CONFIG_IGNORE}


def config_key(cfg):
    return json.dumps(normalize_config(cfg), sort_keys=True)


def lns_trajectory(run_df, start_value, setup_time=0.0):
    """Incumbent trajectory of one LNS run (makespan), starting at the start heuristic.

    Iteration i starts at setup + sum of earlier iteration times; a solution found
    by the CP solver at time s within the solve is placed at that start plus the
    pre-solve part of the iteration plus s.
    """
    pts = [(0.0, start_value)]
    best = start_value
    t_start = setup_time
    for r in run_df.sort_values("iteration").itertuples():
        offset = t_start + (r.total_iter_time - r.solve_time)
        for s in r.discovered_solutions or []:
            ms = s["kpis"]["makespan"]
            if ms < best:
                best = ms
                pts.append((offset + s["time"], ms))
        if r.best < best:  # improvement without a logged discovery (e.g. after merge)
            best = r.best
            pts.append((t_start + r.total_iter_time, best))
        t_start += r.total_iter_time
    return make_traj(pts), t_start


def _run_log_info(exp, run_id):
    """(setup time, total variables, {strategy id: name}) from run_<k>.log."""
    path = os.path.join(exp, f"run_{run_id}.log")
    if not os.path.isfile(path):
        return 0.0, np.nan, {}
    with open(path) as f:
        log = f.read()
    m = re.search(r"Setup time before first iteration:\s*([\d.]+)s", log)
    setup = float(m.group(1)) if m else 0.0
    m = re.search(r"Optimized \d+/(\d+)", log)
    total_vars = int(m.group(1)) if m else np.nan
    names = {int(i): n for n, i in re.findall(r"Strategy: (\w+)\s*\((\d+)\)", log)}
    return setup, total_vars, names


ITER_DROP = ["unfrozen_orders", "unfrozen_skus", "discovered_solutions"]


def _phase_objective_columns(rdf, start_value):
    """Per iteration, the objective the CP optimised in that iteration (cp_obj):

    phase_prev   the incumbent's value of that objective before the iteration
    phase_found  the best value of that objective among the solutions the CP found
                 (NaN if it found none)
    makespan_change  change of the incumbent's makespan in the iteration (acceptance)
    accepted     the incumbent changed

    rdf is one run sorted by iteration. The start heuristic's KPIs other than the
    makespan are unknown, so phase_prev is NaN there for iteration 0.
    """
    objs = rdf["cp_obj"].astype(str).values if "cp_obj" in rdf else np.full(len(rdf), "makespan")
    prev_cols = {}
    for o in set(objs):
        col = f"current_{o}"
        if col in rdf:
            first = start_value if o == "makespan" else np.nan
            prev_cols[o] = np.concatenate([[first], rdf[col].values[:-1]])
    idx = np.arange(len(rdf))
    phase_prev = np.array([prev_cols[o][i] if o in prev_cols else np.nan
                           for i, o in zip(idx, objs)], float)
    phase_found = np.array([
        min((s["kpis"].get(o, np.nan) for s in (sols or [])), default=np.nan)
        for sols, o in zip(rdf["discovered_solutions"], objs)], float)
    # older dataframes may lack the current_* KPIs; fall back to the best makespan
    ms_col = next((c for c in ("current_makespan", "current", "best") if c in rdf), None)
    cur_ms = rdf[ms_col].values
    prev_ms = np.concatenate([[start_value], cur_ms[:-1]])
    kpi_cols = [c for c in rdf.columns if c.startswith("current_")] or [ms_col]
    cur = rdf[kpi_cols].values
    changed = np.concatenate([[True], (cur[1:] != cur[:-1]).any(axis=1)]) if len(rdf) else []
    changed = np.asarray(changed, bool)
    changed[0] = cur_ms[0] != start_value if len(rdf) else False
    return {"phase_obj": objs, "phase_prev": phase_prev, "phase_found": phase_found,
            "makespan_change": cur_ms - prev_ms, "accepted": changed}


def load_lns(d, meta, start_value):
    """Runs and iterations of every finished experiment, tagged with its config key.

    If an instance has several experiments with the same config, the latest is used.
    """
    by_key = {}
    for e in finished_experiments(d):
        cfg = json.load(open(os.path.join(e, "experiment_config.json")))
        key = config_key(cfg)
        if key in by_key:
            warnings.warn(f"{meta['instance']}: duplicate config, using "
                          f"{os.path.basename(e)} over {os.path.basename(by_key[key][0])}")
        by_key[key] = (e, cfg)

    runs, iters, experiments = [], [], []
    for key, (e, cfg) in by_key.items():
        df = pd.read_pickle(os.path.join(e, "experiment_dataframe.pkl"))
        experiments.append(dict(instance=meta["instance"], config_key=key,
                                folder=e, date=parse_exp_date(e), config=cfg))
        for rid, rdf in df.groupby("run_id"):
            rdf = rdf.sort_values("iteration").copy()
            setup, total_vars, names = _run_log_info(e, int(rid))
            (t, v), end = lns_trajectory(rdf, start_value, setup)
            if not (v[-1] == min(rdf.best.min(), start_value)):
                warnings.warn(f"{e} run {rid}: trajectory does not reach the logged best")
            runs.append(dict(config_key=key, family="lns", instance=meta["instance"],
                             run=int(rid), t=t, v=v, final=v[-1], end_time=end,
                             setup_time=setup, n_iters=len(rdf)))
            prev = np.minimum.accumulate(np.concatenate([[start_value], rdf.best.values[:-1]]))
            phase = _phase_objective_columns(rdf, start_value)
            it = rdf.drop(columns=[c for c in ITER_DROP if c in rdf.columns])
            for c, vals in phase.items():
                it[c] = vals
            it["prev_best"] = prev
            it["t_end"] = setup + rdf.total_iter_time.cumsum().values
            it["strategy_name"] = it.strategy.map(names)
            it["unfrozen_pct"] = 100 * it.num_optimized / total_vars
            it["config_key"] = key
            it["instance"] = meta["instance"]
            iters.append(it)
    return runs, iters, experiments


# --------------------------------------------------------------------------- #
# Everything
# --------------------------------------------------------------------------- #
class ExperimentData:
    """All runs of all methods under one experiment root."""

    def __init__(self, instances, runs, iters, experiments, size_order):
        self.instances = instances      # one row per instance
        self.runs = runs                # one row per run of any method
        self.iters = iters              # one row per LNS iteration
        self.experiments = experiments  # one row per (instance, LNS config)
        self.size_order = size_order


def load_root(root, pattern="GEN*", with_lns=True, with_cp=True, with_heuristics=True):
    inst_rows, runs, iters, exps = [], [], [], []
    for d in instance_dirs(root, pattern):
        meta = load_instance_meta(d)
        heur = load_heuristics(d, meta)
        start = [h for h in heur if h["is_start"]]
        meta["start_value"] = start[0]["final"] if start else np.nan
        inst_rows.append(meta)
        if with_heuristics:
            runs += heur
        if with_cp:
            runs += load_cp(d, meta, meta["start_value"])
        if with_lns:
            if np.isnan(meta["start_value"]):
                warnings.warn(f"{d}: no heuristic_summary.txt, LNS start unknown; skipped")
                continue
            r, i, e = load_lns(d, meta, meta["start_value"])
            runs += r
            iters += i
            exps += e
    if not inst_rows:
        raise FileNotFoundError(f"no instance folders matching {pattern} in {root}")
    instances = pd.DataFrame(inst_rows)
    sizes, order = size_labels(instances)
    instances["size"] = [sizes[(s, o)] for s, o in zip(instances.stations, instances.orders)]
    size_of = instances.set_index("instance")["size"]
    runs = pd.DataFrame(runs)
    runs["size"] = runs.instance.map(size_of)
    iters = pd.concat(iters, ignore_index=True) if iters else pd.DataFrame()
    if len(iters):
        iters["size"] = iters.instance.map(size_of)
        iters["improving"] = iters.best < iters.prev_best
        iters["delta"] = (iters.prev_best - iters.best).clip(lower=0)
        iters["delta_pct"] = 100 * iters.delta / iters.prev_best
        # improvement of the objective optimised in the iteration (phase objective)
        iters["phase_delta"] = (iters.phase_prev - iters.phase_found).clip(lower=0).fillna(0)
        iters["phase_improving"] = iters.phase_delta > 0
        iters["phase_delta_pct"] = 100 * iters.phase_delta / iters.phase_prev
    return ExperimentData(instances, runs, iters, pd.DataFrame(exps), order)
