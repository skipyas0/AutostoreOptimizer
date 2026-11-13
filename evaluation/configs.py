"""Name LNS configurations by the parameters in which they differ."""

import json

import pandas as pd

from .data import normalize_config


def _objective(cfg):
    sched, dur = cfg.get("cp_objective_phase_schedule"), cfg.get("cp_objective_phase_duration")
    if not sched or len(set(sched)) == 1 or not dur:
        return f"obj={sched[0] if sched else '?'}"
    return f"obj={sched}/{dur}"


# key -> function(normalized config) giving a short label part. Keys sharing a
# formatter (the objective schedule and its phase length) produce one part.
FORMATTERS = {
    "only_first_sol": lambda c: "first-sol" if c["only_first_sol"] else "all-sols",
    "combine_strat_with_tail": lambda c: f"tail={c['combine_strat_with_tail']}",
    "cp_objective_phase_schedule": _objective,
    "cp_objective_phase_duration": _objective,
    "lns_model": lambda c: c["lns_model"],
    "iter_time_limit": lambda c: f"iter={c['iter_time_limit']:g}s",
    "strat_choice": lambda c: f"choice={c['strat_choice']}",
    "strat_preset": lambda c: f"strats={c['strat_preset']}",
    "cp_objective": _objective,  # the first phase of the schedule
    "default_percent": lambda c: f"nbh={100 * c['default_percent']:g}%",
    "rrt_threshold": lambda c: f"rrt={c['rrt_threshold']:g}",
    "stagnation_th": lambda c: f"stag={c['stagnation_th']}" if c["stagnation_th"] < 1e5 else "",
    "workers": lambda c: f"{c['workers']} workers",
    "iters": lambda c: f"{c['iters']} iters" if c.get("iters") else "",
    "time_limit": lambda c: "" if c.get("time_limit") else "no time limit",
}


def config_table(experiments, prefix="LNS"):
    """One row per distinct config: id, label, differing params, #instances, first date.

    Ids are assigned in order of the first run of each config.
    """
    g = experiments.groupby("config_key").agg(first=("date", "min"),
                                              n_instances=("instance", "nunique"))
    g = g.sort_values("first").reset_index()
    cfgs = [normalize_config(json.loads(k)) for k in g.config_key]
    keys = sorted({k for c in cfgs for k in c})
    differing = [k for k in keys if len({json.dumps(c.get(k)) for c in cfgs}) > 1]
    priority = list(FORMATTERS)
    differing.sort(key=lambda k: (priority.index(k) if k in priority else len(priority), k))

    labels = []
    for c in cfgs:
        parts = []
        for k in differing:
            f = FORMATTERS.get(k, lambda c, k=k: f"{k}={c.get(k)}")
            p = f(c)
            if p and p not in parts:
                parts.append(p)
        labels.append(" · ".join(parts) if parts else "default")
    g["method"] = [f"{prefix}-{i + 1}" for i in range(len(g))]
    g["label"] = labels
    for k in differing:
        g[k] = [c.get(k) for c in cfgs]
    g.attrs["differing"] = differing
    g.attrs["common"] = {k: cfgs[0].get(k) for k in keys if k not in differing}
    return g


def assign_lns_methods(runs, iters, cfg_table):
    m = cfg_table.set_index("config_key").method
    runs = runs.copy()
    lns = runs.family == "lns"
    runs.loc[lns, "method"] = runs.loc[lns, "config_key"].map(m)
    if len(iters):
        iters = iters.copy()
        iters["method"] = iters.config_key.map(m)
    return runs, iters


def config_params_table(cfg_table):
    rows = []
    for r in cfg_table.itertuples():
        row = {"Method": r.method, "Label": r.label}
        for k in cfg_table.attrs["differing"]:
            row[k] = cfg_table.loc[r.Index, k]
        row["Instances"] = r.n_instances
        row["First run"] = r.first.strftime("%Y-%m-%d") if pd.notna(r.first) else ""
        rows.append(row)
    return pd.DataFrame(rows)
