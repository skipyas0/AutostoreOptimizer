#!/usr/bin/env python3
"""
Aggregate and plot v4 scaling benchmark results from many single-run JSON files.

It expects JSONs created by benchmark_v4_scaling_single.py in a directory,
groups them by (parameter, value), and for each parameter builds:

  - median Cmax(t) over seeds
  - min Cmax(t)
  - max Cmax(t)

For each parameter (stations, lanes, orders, movecap) it produces:
  - one PNG plot (matplotlib)
  - one HTML plot (plotly)
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from statistics import median

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import plotly.express as px
import plotly.graph_objects as go


def _safe_median(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return float(median(vals))


def _safe_min(values):
    vals = [v for v in values if v is not None]
    return float(min(vals)) if vals else None


def _safe_max(values):
    vals = [v for v in values if v is not None]
    return float(max(vals)) if vals else None


def load_runs(input_dir):
    """Load all single-run JSONs from input_dir."""
    pattern = os.path.join(input_dir, "*.json")
    files = sorted(glob.glob(pattern))
    runs = []

    if not files:
        raise SystemExit(f"No JSON files found in {input_dir}")

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"Warning: failed to parse {path}: {e}")
                continue
        runs.append((path, data))

    print(f"Loaded {len(runs)} runs from {input_dir}")
    return runs


def group_by_param_value(runs):
    """
    Group runs by (parameter, value).
    Returns:
      grouped[param][value] = list of run dicts (each with keys 'meta', 'config', 'result', 'progress').
    Also returns a reference_config per parameter (from first run).
    """
    grouped = defaultdict(lambda: defaultdict(list))
    ref_config = {}

    for path, data in runs:
        meta = data.get("meta", {})
        param = meta.get("parameter")
        val = meta.get("value")
        if param is None or val is None:
            print(f"Warning: {path} missing parameter/value in meta, skipping.")
            continue

        grouped[param][val].append(data)

        if param not in ref_config and "reference_config" in meta:
            ref_config[param] = meta["reference_config"]

    return grouped, ref_config


def build_seed_curves(grouped):
    """
    grouped[param][value] = list of run dicts

    Returns:
      seed_curves[param][value] = list of dicts:
        {
          "seed": seed_id,
          "time": [t0, t1, ...],
          "best": [cmax0, cmax1, ...],
        }
    """
    per_param = {}

    for param, value_to_runs in grouped.items():
        value_curves = defaultdict(list)
        for value, runs in value_to_runs.items():
            for data in runs:
                meta = data.get("meta", {})
                seed = meta.get("seed")
                prog = sorted(data.get("progress", []), key=lambda r: r["time"])
                if not prog:
                    continue
                t = [rec["time"] for rec in prog]
                y = [rec["best"] for rec in prog]
                value_curves[value].append(
                    {"seed": seed, "time": t, "best": y}
                )
        per_param[param] = value_curves

    return per_param


def last_best_before_time(progress, t):
    """Return best objective value in progress list at or before time t, or None if no solution yet."""
    last = None
    for rec in progress:
        if rec["time"] <= t:
            last = rec
        else:
            break
    return last["best"] if last is not None else None


def summarize_parameter(param, value_to_runs):
    """
    Build a summary table for one parameter.

    Returns a list of row dicts, each with keys:
      'param', 'value', 'n_runs', 'n_feas', 'n_opt',
      'median_Cmax', 'min_Cmax', 'max_Cmax',
      'median_t_first', 'median_t_last',
      'median_final_gap', 'median_solve_time'
    """
    rows = []

    for value in sorted(value_to_runs.keys()):
        runs = value_to_runs[value]
        n_runs = len(runs)

        final_bests = []
        first_times = []
        last_times = []
        final_gaps = []
        solve_times = []
        statuses = []

        for data in runs:
            res = data.get("result", {}) or {}
            status = res.get("status")
            statuses.append(status)
            st = res.get("solve_time")
            if st is not None:
                solve_times.append(float(st))

            prog = data.get("progress", []) or []
            if prog:
                # last improvement
                last = prog[-1]
                best = last.get("best")
                gap = last.get("gap")

                final_bests.append(float(best) if best is not None else None)
                first_times.append(float(prog[0].get("time", 0.0)))
                last_times.append(float(last.get("time", 0.0)))
                if gap is not None:
                    final_gaps.append(float(gap))

        n_feas = len(final_bests)
        n_opt = sum(
            1 for s in statuses
            if s is not None and "OPTIMAL" in str(s).upper()
        )

        row = {
            "param": param,
            "value": value,
            "n_runs": n_runs,
            "n_feas": n_feas,
            "n_opt": n_opt,
            "median_Cmax": _safe_median(final_bests),
            "min_Cmax": _safe_min(final_bests),
            "max_Cmax": _safe_max(final_bests),
            "median_t_first": _safe_median(first_times),
            "median_t_last": _safe_median(last_times),
            "median_final_gap": _safe_median(final_gaps),
            "median_solve_time": _safe_median(solve_times),
        }
        rows.append(row)

    return rows


def write_summary_csv(path, rows):
    fieldnames = [
        "param", "value",
        "n_runs", "n_feas", "n_opt",
        "median_Cmax", "min_Cmax", "max_Cmax",
        "median_t_first", "median_t_last",
        "median_final_gap", "median_solve_time",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Saved CSV summary: {path}")


def write_summary_latex(path, param, rows, experiment_label="", model_version=""):
    """
    Write a LaTeX tabular for one parameter.
    """
    caption = (
        f"Summary of scaling in {param} "
        f"({experiment_label})"
        if experiment_label
        else f"Summary of scaling in {param}"
    )
    label = f"tab:{model_version}-summary-{param}"

    with open(path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{lrrrrrrrrrr}\n")
        f.write("\\toprule\n")
        f.write(
            "Level & "
            "$n_\\text{runs}$ & $n_\\text{feas}$ & $n_\\text{opt}$ & "
            "$\\widetilde{C_{\\max}}$ & "
            "$C_{\\max}^{\\min}$ & $C_{\\max}^{\\max}$ & "
            "$\\widetilde{t_\\text{first}}$ & "
            "$\\widetilde{t_\\text{last}}$ & "
            "$\\widetilde{\\text{gap}}$ & "
            "$\\widetilde{t_\\text{solve}}$ \\\\\n"
        )
        f.write("\\midrule\n")

        for r in rows:
            def fmt(x, nd=1):
                if x is None:
                    return "--"
                return f"{x:.{nd}f}"

            f.write(
                f"{r['value']} & "
                f"{r['n_runs']} & {r['n_feas']} & {r['n_opt']} & "
                f"{fmt(r['median_Cmax'])} & "
                f"{fmt(r['min_Cmax'])} & "
                f"{fmt(r['max_Cmax'])} & "
                f"{fmt(r['median_t_first'])} & "
                f"{fmt(r['median_t_last'])} & "
                f"{fmt(r['median_final_gap'])} & "
                f"{fmt(r['median_solve_time'])} \\\\\n"
            )

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write("\\end{table}\n")
    print(f"Saved LaTeX summary: {path}")


def aggregate_for_parameter(param, value_to_runs, time_limit=None):
    """
    For a single parameter (e.g., 'orders'), aggregate all runs per value.

    value_to_runs: dict[value] = list of run dicts

    Returns:
      aggregated[value]      = list of dicts {time, median, min, max}
      sorted_values          = sorted list of parameter values
      improve_points[value]  = list of (time, median) for every solution event
      opt_points[value]      = list of (solve_time, objective_value) for Optimal runs
      sentinel               = y-top sentinel used for max band
    """
    # --- Build common time grid and find a global sentinel ---
    all_times = set()
    all_best_vals = []

    for value, runs in value_to_runs.items():
        for data in runs:
            # Inject synthetic progress for heuristic runs (or others missing progress logs)
            # that nevertheless have a valid result.
            if len(data.get("progress")) == 0 and data.get("result"):
                res = data["result"]
                obj = res.get("objective_value")
                st = res.get("solve_time")
                if obj is not None and st is not None:
                    data["progress"] = [{
                        "time": float(st),
                        "best": float(obj),
                        "gap": None
                    }]

            for rec in data.get("progress", []):
                all_times.add(rec["time"])
                if "best" in rec and rec["best"] is not None:
                    all_best_vals.append(rec["best"])

    time_grid = sorted(all_times)

    if all_best_vals:
        sentinel = max(all_best_vals) * 1.05
    else:
        sentinel = 1.0

    aggregated = {}
    sorted_values = sorted(value_to_runs.keys())
    opt_points = defaultdict(list)
    failed_values = []

    # --- Aggregate per value ---
    for value in sorted_values:
        runs = value_to_runs[value]
        if not any(len(d.get("progress", [])) > 0 for d in runs):
            failed_values.append(value)
            aggregated[value] = []
            continue
        curve = []
        if not runs:
            aggregated[value] = curve
            continue

        all_finished = all(
            d.get("result", {}).get("status") in ("Optimal", "Infeasible")
            for d in runs
        )

        for t in time_grid:
            solved_vals = []
            max_candidates = []

            for data in runs:
                prog = data.get("progress", [])
                v = last_best_before_time(prog, t)
                if v is not None:
                    solved_vals.append(v)
                    max_candidates.append(v)
                else:
                    max_candidates.append(sentinel)

            if not solved_vals:
                # No seed has any solution yet at this time -> skip
                continue

            med = float(median(solved_vals))
            minv = float(min(solved_vals))
            maxv = float(max(max_candidates))  # may be sentinel

            curve.append(
                {"time": t, "median": med, "min": minv, "max": maxv}
            )

        if curve and time_limit is not None:
            last_pt = curve[-1]
            if last_pt["time"] < time_limit:
                # Duplicate the last stats but at the max time
                final_pt = last_pt.copy()
                final_pt["time"] = time_limit
                final_pt["is_ext"] = True
                final_pt["all_finished"] = all_finished
                curve.append(final_pt)

        aggregated[value] = curve

        # Optimal points per run
        for data in runs:
            res = data.get("result", {})
            status = res.get("status")
            obj = res.get("objective_value")
            t_opt = res.get("solve_time")
            if status == "Optimal" and obj is not None and t_opt is not None:
                opt_points[value].append((float(t_opt), float(obj)))

    # --- "o" markers: every solution event for this value ---
    improve_points = {}
    for value in sorted_values:
        runs = value_to_runs[value]
        if not runs:
            improve_points[value] = []
            continue

        # 1) union of all progress times for this value (all seeds)
        sol_times = set()
        for data in runs:
            for rec in data.get("progress", []):
                sol_times.add(rec["time"])
        sol_times = sorted(sol_times)

        # 2) exclude exact optimal times (to avoid overlap with diamonds)
        opt_t_set = {t for (t, _) in opt_points.get(value, [])}
        sol_times = [t for t in sol_times if t not in opt_t_set]

        # 3) map time -> median on the aggregated curve, so markers lie on the line
        curve = aggregated[value]

        def get_median_at_t(lookup_t, curve_data):
            # Simple search since strictly increasing time
            last_med = None
            for c in curve_data:
                if c["time"] == lookup_t:
                    return c["median"]
                if c["time"] > lookup_t:
                    return last_med
                last_med = c["median"]
            return last_med

        pts = []
        for t in sol_times:
            m = get_median_at_t(t, curve)
            if m is not None:
                pts.append((t, m))

        improve_points[value] = pts

    return aggregated, sorted_values, improve_points, opt_points, sentinel, failed_values


def plot_matplotlib(param, aggregated, values, improve_points, opt_points, sentinel, ref_cfg, out_png, failed_values,
                    model_prefix, constant_config):
    plt.figure(figsize=(8, 5))

    for value in values:
        curve = aggregated[value]
        if not curve:
            continue

        real_c = [c for c in curve if not c.get("is_ext")]
        t = [c["time"] for c in real_c]
        med = [c["median"] for c in real_c]
        minv = [c["min"] for c in real_c]
        maxv = [c["max"] for c in real_c]

        label = f"{param}={value}"
        line, = plt.plot(t, med, label=label)
        color = line.get_color()
        plt.fill_between(t, minv, maxv, alpha=0.2, color=color)

        # Plot dashed extension if it exists
        if len(curve) > len(real_c):
            ext_t = [real_c[-1]["time"], curve[-1]["time"]]
            ext_med = [real_c[-1]["median"], curve[-1]["median"]]
            ext_min = [real_c[-1]["min"], curve[-1]["min"]]
            ext_max = [real_c[-1]["max"], curve[-1]["max"]]
            ls = ":" if curve[-1].get("all_finished", False) else "-"
            plt.plot(ext_t, ext_med, color=color, linestyle=ls)
            plt.fill_between(ext_t, ext_min, ext_max, alpha=0.1, color=color)

        # "o" markers at every solution event for this value
        imp = improve_points.get(value, [])
        if imp:
            imp_t, imp_y = zip(*imp)
            plt.scatter(
                imp_t,
                imp_y,
                marker="o",
                s=25,
                color=color,
                edgecolors="none",
                zorder=3,
            )

        # Diamonds at optimal runs
        opt = opt_points.get(value, [])
        if opt:
            opt_t, opt_y = zip(*opt)
            plt.scatter(
                opt_t,
                opt_y,
                marker="D",
                s=35,
                facecolors=color,
                edgecolors="black",
                linewidths=0.5,
                zorder=4,
            )
    for value in failed_values:
        plt.plot([], [], color='gray', marker='x', markersize=8,
                 label=f"{param}={value}")

    # Clamp y-axis top to sentinel
    all_mins = [c["min"] for v in aggregated.values() for c in v]
    ymin = min(all_mins) if all_mins else 0.0

    plt.xlim(left=0, right=ref_cfg.get('timelimit'))
    plt.ylim(top=sentinel)
    # plt.yscale("log"); plt.ylim(bottom=max(1, ymin), top=sentinel)
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=20))

    plt.xlabel("Solve time [s]")
    plt.ylabel("Cmax (makespan)")
    title = f"{model_prefix} model scaling for {param}"
    # title = f"Model scaling for {param}"
    # if ref_cfg:
    #     title += (
    #         f" (ref: stations={ref_cfg.get('stations')}, "
    #         f"lanes={ref_cfg.get('lanes')}, "
    #         f"orders={ref_cfg.get('orders')}"
    #         # f"movecap={ref_cfg.get('movecap')})"
    #     )
    if constant_config:
        parts = [f"{k}={v}" for k, v in constant_config.items()]
        title += f" ({', '.join(parts)})"
    plt.title(title, fontsize=10)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(title="5 runs Median", title_fontsize="x-small", fontsize="x-small", loc=1)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Saved PNG: {out_png}")


def plot_plotly(param, aggregated, values, improve_points, opt_points, sentinel, ref_cfg, seed_curves, out_html,
                failed_values, model_prefix, constant_config):
    fig = go.Figure()
    color_seq = px.colors.qualitative.Plotly

    for idx, value in enumerate(values):
        curve = aggregated[value]
        if not curve:
            continue

        # Rozdělení na reálné body a bod prodloužení
        real_c = [c for c in curve if not c.get("is_ext")]
        t = [c["time"] for c in real_c]
        med = [c["median"] for c in real_c]

        full_t = [c["time"] for c in curve]
        full_min = [c["min"] for c in curve]
        full_max = [c["max"] for c in curve]

        group_name = f"{param}={value}"
        color = color_seq[idx % len(color_seq)]

        # 1. Median line (pouze reálná část)
        fig.add_trace(
            go.Scatter(
                x=t,
                y=med,
                mode="lines",
                name=f"{group_name}",
                line=dict(color=color),
                meta=dict(group=group_name, role="median"),
                hovertemplate=(
                    "t = %{x:.1f} s<br>"
                    "median Cmax = %{y}<br>"
                    f"{param} = {value}<extra></extra>"
                ),
            )
        )

        # 2. Prodloužená čára (pokud existuje)
        if len(curve) > len(real_c):
            ext_c = curve[-1]
            ext_t = [real_c[-1]["time"], ext_c["time"]]
            ext_med = [real_c[-1]["median"], ext_c["median"]]

            # Styl podle toho, jestli všichni skončili
            dash_style = "dot" if ext_c.get("all_finished", False) else "solid"

            fig.add_trace(
                go.Scatter(
                    x=ext_t,
                    y=ext_med,
                    mode="lines",
                    name=f"{group_name} (ext)",
                    line=dict(color=color, dash=dash_style),
                    showlegend=False,
                    meta=dict(group=group_name, role="median"),  # Stejná role zajistí správný hover efekt
                    hoverinfo="skip"
                )
            )

        # 3. Band (min/max - ten můžeme vykreslit rovnou celý, protože nemá okraje)
        fig.add_trace(
            go.Scatter(
                x=full_t + full_t[::-1],
                y=full_max + full_min[::-1],
                fill="toself",
                opacity=0.15,
                line=dict(width=0),
                fillcolor=color,
                showlegend=False,
                meta=dict(group=group_name, role="band"),
                hoverinfo="skip",
            )
        )
        # "o" markers
        imp = improve_points.get(value, [])
        if imp:
            imp_t, imp_y = zip(*imp)
            fig.add_trace(
                go.Scatter(
                    x=imp_t,
                    y=imp_y,
                    mode="markers",
                    marker=dict(symbol="circle", size=6, color=color),
                    name=f"{group_name} solutions",
                    showlegend=False,
                    meta=dict(group=group_name, role="imp"),
                    hovertemplate="t = %{x:.1f} s<br>Cmax = %{y}<extra></extra>",
                )
            )

        # Diamonds for optimal runs
        opt = opt_points.get(value, [])
        if opt:
            opt_t, opt_y = zip(*opt)
            fig.add_trace(
                go.Scatter(
                    x=opt_t,
                    y=opt_y,
                    mode="markers",
                    marker=dict(
                        symbol="diamond",
                        size=8,
                        color=color,
                        line=dict(width=1, color="black"),
                    ),
                    name=f"{group_name} optimal",
                    showlegend=False,
                    meta=dict(group=group_name, role="opt"),
                    hovertemplate="t = %{x:.1f} s<br>Cmax = %{y}<extra></extra>",
                )
            )

        # ---- Per-seed curves: hidden by default ----
        for seed_curve in seed_curves.get(value, []):
            sc_t = seed_curve["time"]
            sc_y = seed_curve["best"]
            seed = seed_curve["seed"]

            fig.add_trace(
                go.Scatter(
                    x=sc_t,
                    y=sc_y,
                    mode="lines",
                    line=dict(color=color, width=1, dash="dot"),
                    name=f"{group_name}, seed={seed}",
                    showlegend=False,
                    visible=False,  # hidden until hovered
                    meta=dict(group=group_name, role="seed"),
                    hovertemplate=(
                        "t = %{x:.1f} s<br>"
                        "Cmax (seed) = %{y}<br>"
                        f"{param} = {value}, seed={seed}"
                        "<extra></extra>"
                    ),
                )
            )

    for value in failed_values:
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="lines+markers",
                marker=dict(symbol="x", size=10, color="gray"),
                line=dict(color="gray", dash="dot"),
                name=f"<s>{param}={value}</s>",
                showlegend=True,
                hoverinfo="skip"
            )
        )

    all_mins = [c["min"] for v in aggregated.values() for c in v]
    ymin = min(all_mins) if all_mins else 0.0

    fig.update_yaxes(range=[0, sentinel])
    # fig.update_yaxes(type="log", range=[math.log10(max(1, ymin)), math.log10(sentinel)])
    title = f"{model_prefix} model scaling for {param}"
    # if ref_cfg:
    #     title += (
    #         f" (ref: stations={ref_cfg.get('stations')}, "
    #         f"lanes={ref_cfg.get('lanes')}, "
    #         f"orders={ref_cfg.get('orders')}"
    #         # f"movecap={ref_cfg.get('movecap')})"
    #     )
    if constant_config:
        parts = [f"{k}={v}" for k, v in constant_config.items()]
        title += f" ({', '.join(parts)})"

    fig.update_layout(
        title=title,
        xaxis_title="Solve time [s]",
        yaxis_title="Cmax (makespan)",
        legend_title_text="5 runs Median"
    )

    div_id = f"model_scaling_{param}"
    html = fig.to_html(
        full_html=True,
        include_plotlyjs="cdn",
        div_id=div_id,
    )

    js = f"""
    <script>
    document.addEventListener("DOMContentLoaded", function() {{
      var plot = document.getElementById("{div_id}");
      if (!plot) return;

      var origOpacity = [];
      var origVisible = [];
      var initDone = false;
      var currentGroup = null;

      // Capture initial state
      if(plot.data) {{
        for (var i = 0; i < plot.data.length; i++) {{
          origOpacity[i] = (plot.data[i].opacity === undefined) ? 1 : plot.data[i].opacity;
          // Seeds are hidden by default, so 'true' visible is only for non-seeds
          origVisible[i] = (plot.data[i].visible === undefined) ? true : plot.data[i].visible;
        }}
        initDone = true;
      }}
      
      function getMeta(i) {{
        var m = (plot.data && plot.data[i]) ? (plot.data[i].meta || {{}}) : {{}};
        return {{
          group: m.group || null,
          role:  m.role  || null
        }};
      }}

      // Reset logic: restores everything to default visibility
      function resetView() {{
        if (currentGroup === null) return;
        currentGroup = null;
        
        var updateOp = [];
        var updateVis = [];
        var indices = [];
        
        for (var i = 0; i < plot.data.length; i++) {{
            var m = getMeta(i);
            var role = m.role;
            
            // Restore default opacity
            var op = origOpacity[i];
            
            // Restore default visibility. 
            // Crucially: 'seed' traces must be hidden again.
            var vis = origVisible[i];
            if (role === 'seed') {{
                vis = false; 
            }} else {{
                vis = true;
            }}
            
            updateOp.push(op);
            updateVis.push(vis);
            indices.push(i);
        }}
        Plotly.restyle(plot, {{
            opacity: updateOp,
            visible: updateVis
        }}, indices);
      }}

      // Hover logic: Dim others, highlight group, show seeds
      plot.on('plotly_hover', function(ev) {{
        if (!ev || !ev.points || ev.points.length === 0) return;
        var curveIdx = ev.points[0].curveNumber;
        var metaHovered = getMeta(curveIdx);
        var group = metaHovered.group;
        
        // If verify we are focusing a new group
        if (!group) return; 
        if (group === currentGroup) return; // Optimization: don't restyle if already highlighting this group
        
        currentGroup = group;

        var updateOp = [];
        var updateVis = [];
        var indices = [];

        for (var i = 0; i < plot.data.length; i++) {{
          var m = getMeta(i);
          var role = m.role;
          var g    = m.group;

          var newOp  = origOpacity[i];
          var newVis = origVisible[i];

          if (g === group) {{
            // Highlight THIS group
            if (role === 'band') {{
              newVis = false; // hide band to reduce clutter
            }} else if (role === 'seed') {{
              newVis = true;  // SHOW seeds
              newOp  = 0.5;   // slightly transparent seeds
            }} else {{
              newVis = true;  // main line / points
              newOp  = 1.0;
            }}
          }} else {{
            // Dim OTHER groups
            if (role === 'seed') {{
              newVis = false; // ensure their seeds are hidden
            }} else {{
              newOp  = 0.1;   // dim their main lines
              newVis = true;
            }}
          }}
          
          updateOp.push(newOp);
          updateVis.push(newVis);
          indices.push(i);
        }}
        
        Plotly.restyle(plot, {{
            opacity: updateOp,
            visible: updateVis
        }}, indices);
      }});

      // Use 'mouseleave' on the container to detect when interaction ends completely.
      // 'plotly_unhover' fires too aggressively (e.g. between points) causing flicker.
      plot.addEventListener('mouseleave', function() {{
         resetView();
      }});
      
      // Also reset on double-click (often used to reset zoom)
      plot.on('plotly_doubleclick', function() {{
         resetView();
      }});
    }});
    </script>
    """

    html = html.replace("</body>", js + "\\n</body>")

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved HTML: {out_html}")


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate and plot AutoStore vX scaling results from many JSONs."
    )
    ap.add_argument(
        "--input-dir",
        required=True,
        help="Directory with JSON files from benchmark_vX_scaling_single.py",
    )
    ap.add_argument(
        "--output-dir",
        required=True,
        help="Directory to store generated PNG and HTML plots.",
    )
    ap.add_argument(
        "--no-plots",
        action="store_true",
        help="If set, do not generate plots (only summaries).",
    )
    ap.add_argument(
        "--experiment-label",
        type=str,
        default="",
        help="Short description used in LaTeX captions, e.g. '30 orders, T=2400 s'.",
    )
    ap.add_argument(
        "--max-seeds",
        type=int,
        default=None,
        help="Limit the number of seeds used (e.g., 3 means seeds 0, 1, 2). Default: use all found.",
    )
    ap.add_argument(
        "--model-version",
        type=str,
        default="v4",
        help="Model version (e.g., v4 or v5).",
    )
    args = ap.parse_args()
    if args.model_version not in ("v4", "v5"):
        raise ValueError(f"Invalid model version: {args.model_version}")
    else:
        model_version_str = args.model_version
    runs = load_runs(args.input_dir)
    # Filter runs based on max_seeds if provided
    if args.max_seeds is not None and args.max_seeds > 0:
        print(f"Filtering runs to keep only {args.max_seeds} seeds...")
        filtered_runs = []
        for path, data in runs:
            meta = data.get("meta", {})
            seed_val = meta.get("seed")
            local_filtered_runs = 0
            if seed_val is None:
                # Skip runs without a seed if we are filtering
                print(f"  Skipping run {path}: no seed value")
                continue

            try:
                seed_idx = int(seed_val)
                if local_filtered_runs < args.max_seeds:
                    filtered_runs.append((path, data))
                    local_filtered_runs += 1
            except (ValueError, TypeError):
                # Skip runs with invalid seed values
                print(f"  Skipping run {path}: invalid seed value {seed_val}")
                continue

        print(f"Using {len(filtered_runs)} out of {len(runs)} loaded runs after seed filtering.")
        runs = filtered_runs
    grouped, ref_configs = group_by_param_value(runs)
    seed_curves_all = build_seed_curves(grouped)

    os.makedirs(args.output_dir, exist_ok=True)

    # For each parameter (stations, lanes, orders, movecap) produce plots
    for param, value_to_runs in grouped.items():
        print(f"\n=== Aggregating parameter: {param} ===")

        ref_cfg = ref_configs.get(param, {})
        tl = float(ref_cfg.get('timelimit')) if ref_cfg.get('timelimit') else None
        print(f"  Reference config: {ref_cfg}")

        all_runs_for_param = [run for runs in value_to_runs.values() for run in runs]
        constant_config = {}
        for key in ["stations", "lanes", "orders", "skus", "movecap"]:
            unique_vals = set(r.get("config", {}).get(key)
                              for r in all_runs_for_param if r.get("config", {}).get(key) is not None)
            if len(unique_vals) == 1:
                constant_config[key] = unique_vals.pop()

        print(f"  Constant config: {constant_config}")

        aggregated, sorted_values, improve_points, opt_points, sentinel, failed_values = \
            aggregate_for_parameter(param, value_to_runs, tl)

        if not sorted_values:
            print(f"  No values for parameter {param}, skipping.")
            continue

        base = f"{model_version_str}_scaling_{param}"
        out_png = os.path.join(args.output_dir, base + ".png")
        out_html = os.path.join(args.output_dir, base + ".html")

        # Detect run mode (cp, warmstart, heuristic) from metadata
        run_modes = {r.get("meta", {}).get("mode") for r in all_runs_for_param}
        # Filter None
        run_modes = {m for m in run_modes if m}

        mode_suffix = ""
        if "warmstart" in run_modes:
            mode_suffix = " + Warmstart"
            print("Warmstart used")
        elif "heuristic" in run_modes:
            mode_suffix = " Greedy Heuristic"
            if model_version_str is None:
                model_version_str = "v5"
            print("Heuristic only used")
        else:
            print("No heuristic used")

        if model_version_str == "v4":
            model_prefix = f"Single-SKU{mode_suffix}"
        elif model_version_str == "v5":
            model_prefix = f"Multi-SKU{mode_suffix}"
        else:
            raise ValueError(f"Unknown model version: {model_version_str}")
        plot_matplotlib(param, aggregated, sorted_values, improve_points, opt_points, sentinel, ref_cfg, out_png,
                        failed_values, model_prefix, constant_config)

        seed_curves = seed_curves_all.get(param, {})
        # aggregated, sorted_values, improve_points, opt_points, sentinel = aggregate_for_parameter(param, value_to_runs)

        plot_plotly(param, aggregated, sorted_values, improve_points, opt_points, sentinel, ref_cfg, seed_curves,
                    out_html, failed_values, model_prefix, constant_config)

        rows = summarize_parameter(param, value_to_runs)
        csv_path = os.path.join(args.output_dir, f"summary_{param}.csv")
        tex_path = os.path.join(args.output_dir, f"summary_{param}.tex")
        write_summary_csv(csv_path, rows)
        write_summary_latex(tex_path, param, rows, args.experiment_label, model_version_str)


if __name__ == "__main__":
    main()
