"""
Interactive schedule visualiser for order_station_assign.py

Features
- One subplot per *used* station (S0, S1, ...)
- Station-level rows:
    - Fetch/Return lanes: S{s} · F/R Lane #0..N (packed, Fetches and Returns for the same
                                                bin copy (s,k,e) are on the same row)
    - Bin row:            S{s} · Bin            (exactly one bin at station by model)
    - Pick row:           S{s} · Pick           (disjunctive by model; may keep for reference)
- Per-station·lane rows:
    - S{s} · Lane {ℓ}: order windows as bars; consumptions drawn as thin lines inside the bar
      (For v3 models w/o I_os_lane, orders are packed visually into lanes)
- Legend, hover, gridlines, makespan marker, station dropdown

Requires: plotly>=5, docplex.cp
"""
from __future__ import annotations
from typing import Dict, Tuple, Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from docplex.cp.solution import CpoSolveResult
except Exception:  # pragma: no cover
    CpoSolveResult = Any  # type: ignore

# Consumption visuals / hover tuning
CONS_LINE_WIDTH_PX = 6         # ≈ half the apparent height of an order bar (tune 8–12)


# --- styling ---
COLORS = {
    "order":  "#6C5CE7",  # violet
    "fetch":  "#FF8C00",  # darkorange
    "pick":   "#E63946",  # crimson
    "cons":   "#FF0015",  # teal
    # "cons":   "#2A9D8F",  # teal
    "return": "#1D4ED8",  # blue
    "bin":    "#6C757D",  # slate gray
}

FONT = dict(family="Inter, Segoe UI, Roboto, Arial, sans-serif", size=12)
GRID_COLOR = "#E9ECEF"
BG = "white"


# ---- CP helper accessors (CP Optimizer 22.1) ----
def _iv_present(sol, x) -> bool:
    vs = sol.get_var_solution(x)
    return (vs is not None) and vs.is_present()


def _iv_start(sol, x) -> int:
    return sol.get_var_solution(x).get_start()


def _iv_end(sol, x) -> int:
    return sol.get_var_solution(x).get_end()


# ---- plotting helpers ----
def _legend_stub(trace_name: str, color: str) -> go.Bar:
    # Invisible bar to get a proper legend swatch
    return go.Bar(
        x=[0], y=["legend"], orientation="h", name=trace_name,
        marker_color=color, showlegend=True, hoverinfo="skip", visible=True
    )


def _pack_lanes(intervals):
    """Greedy lane packing (visual only).
       intervals: list[(start, end, payload)] -> list[(start, end, payload, lane)]"""
    events = sorted(intervals, key=lambda t: (t[0], t[1]))
    lane_end = []
    out = []
    for st, en, pl in events:
        lane = None
        for i, e in enumerate(lane_end):
            if e <= st:
                lane = i
                lane_end[i] = en
                break
        if lane is None:
            lane = len(lane_end)
            lane_end.append(en)
        out.append((st, en, pl, lane))
    return out


def _add_bar(fig: go.Figure, *, y: str, start: int, end: int, color: str, hover: str,
             legendgroup: str, name: str, meta_station: int, row: int, text: str = None,
             type_label: str = ""):
    dur = max(0, end - start)
    if dur <= 0:
        return
    fig.add_trace(go.Bar(
        x=[dur], base=[start], y=[y], orientation="h",
        marker=dict(
            color=color,
            line=dict(color="rgba(255,255,255,0.9)", width=1),
        ),
        text=text, textposition="inside", insidetextanchor="middle", textangle=0,
        textfont=dict(size=14, color="white"),
        opacity=0.95,
        customdata=[[start, end, hover, dur, type_label]],
        hovertemplate="<b>%{customdata[4]}</b><br>%{customdata[0]} → %{customdata[1]}<br>Δ %{customdata[3]}<extra>%{customdata[2]}</extra>",
        showlegend=False, legendgroup=legendgroup, name=name,
        meta=dict(station=meta_station)
    ), row=row, col=1)


def _add_cons_line(fig: go.Figure, *, y: str, start: int, end: int, color: str,
                   hover: str, meta_station: int, row: int, width_px: int = CONS_LINE_WIDTH_PX,
                   type_label: str = ""):
    # Use a bar at the bottom of the row to "underline" the order
    # Assuming standard bar width is ~0.8, we place this at the bottom edge.
    # offset=-0.4 starts at the bottom edge.

    CONS_HEIGHT = 0.15
    CONS_OFFSET = -0.4

    dur = max(0, end - start)
    if dur <= 0:
        return

    # White outline (slightly larger)
    fig.add_trace(go.Bar(
        x=[dur], base=[start], y=[y], orientation="h",
        width=CONS_HEIGHT + 0.04,
        offset=CONS_OFFSET - 0.02,
        marker=dict(color="white", line_width=0),
        hoverinfo="skip",
        showlegend=False, legendgroup="cons_outline", name="Consumption (outline)",
        meta=dict(station=meta_station, role="cons_outline"),
    ), row=row, col=1)

    # Colored bar
    fig.add_trace(go.Bar(
        x=[dur], base=[start], y=[y], orientation="h",
        width=CONS_HEIGHT,
        offset=CONS_OFFSET,
        marker=dict(color=color, line_width=0),
        customdata=[[start, end, hover, dur, type_label]],
        hovertemplate="<b>%{customdata[4]}</b><br>%{customdata[0]} → %{customdata[1]}<br>Δ %{customdata[3]}<extra>%{customdata[2]}</extra>",
        showlegend=False, legendgroup="cons", name="Consumption",
        meta=dict(station=meta_station, role="cons"),
    ), row=row, col=1)


# ---- main API ----
def plot_schedule(solution: CpoSolveResult, handles: Dict[str, Any], *,
                  include_consumptions: bool = True, show: bool = True,
                  utilization_plot_top: bool = False) -> go.Figure:
    """Build and (optionally) show a schedule visualisation from the CP solution."""
    if solution is None:
        raise ValueError("No solution to visualise.")

    I_os = handles["I_os"]
    I_os_lane = handles.get("I_os_lane")  # v3 compatible: OK if this is None
    C = handles["C"]
    P = handles["P"]
    F = handles["F"]
    R = handles.get("R", {})  # v2
    B = handles.get("B", {})  # v2
    U = handles["U"]
    orders_req = handles["orders_req"]
    S, L, K, O = handles["S"], handles["L"], handles["K"], handles["O"]

    # --- figure out which stations are actually used in the solution ---
    def _stations_used():
        used = []
        for s in S:
            any_present = False
            # any order window at station s?
            for o in O:
                iv = I_os.get((o, s))
                if iv is not None and _iv_present(solution, iv):
                    any_present = True
                    break
            # any station-level activity?
            if not any_present:
                for k in K:
                    for e in range(U.get(k, 0)):
                        hits = [
                            P.get((s, k, e)), F.get((s, k, e)),
                            R.get((s, k, e)), B.get((s, k, e))
                        ]
                        if any(iv is not None and _iv_present(solution, iv) for iv in hits):
                            any_present = True
                            break
                    if any_present:
                        break
            if any_present:
                used.append(s)
        return used

    stations = _stations_used()
    if not stations:
        raise ValueError("No active stations to visualise.")

    # ---- compute makespan and lane assignment ----
    makespan = 0
    order_intervals_by_station = {s: [] for s in S}
    for o in O:
        for s in S:
            iv = I_os.get((o, s))
            if iv is not None and _iv_present(solution, iv):
                makespan = max(makespan, _iv_end(solution, iv))
                # For v3 packing:
                order_intervals_by_station[s].append(
                    (_iv_start(solution, iv), _iv_end(solution, iv), o)  # (start, end, payload=order_id)
                )
                break  # Order is only at one station

    assign: Dict[int, Tuple[int, int]] = {}  # o -> (s, ln)
    if I_os_lane:
        # v2 logic: assignment is in the model
        print("[Visualizer] Using v2 logic: I_os_lane is present.")
        for o in O:
            s_sel = next(s for s in S if _iv_present(solution, I_os[(o, s)]))
            try:
                ln_sel = next(ln for ln in L if _iv_present(solution, I_os_lane[(o, s_sel, ln)]))
            except StopIteration:
                raise RuntimeError(f"No lane selected for order {o} at station {s_sel}. Check model constraints.")
            assign[o] = (s_sel, ln_sel)
    else:
        # v3 logic: no lanes in model, pack them visually
        print("[Visualizer] Using v3 logic: I_os_lane not found. Packing orders into lanes.")
        for s in stations:
            # Pack orders for this station
            packed_orders = _pack_lanes(order_intervals_by_station[s])  # list of (st, en, o, lane)
            for st, en, o, ln in packed_orders:
                if ln >= len(L) and len(L) > 0:  # Check vs. model capacity
                    print(f"[Visualizer] Warning: Order {o} (at S{s}) packed into visual lane {ln}, "
                          f"which exceeds model lane capacity |L|={len(L)}.")
                assign[o] = (s, ln)

    # ---- calculate metrics ----
    if makespan == 0:
        makespan_calc = 1
    else:
        makespan_calc = makespan

    # 1. Peak & Avg Robot Concurrency
    robot_events = []
    for k in K:
        for e in range(U.get(k, 0)):
            for s in S:
                ivf = F.get((s, k, e))
                ivr = R.get((s, k, e))
                if ivf is not None and _iv_present(solution, ivf):
                    robot_events.extend([(_iv_start(solution, ivf), 1), (_iv_end(solution, ivf), -1)])
                if ivr is not None and _iv_present(solution, ivr):
                    robot_events.extend([(_iv_start(solution, ivr), 1), (_iv_end(solution, ivr), -1)])

    robot_events.sort(key=lambda x: (x[0], x[1]))
    current_robots = 0
    peak_robots = 0
    robot_active_time = 0
    last_time = 0

    for t, change in robot_events:
        if current_robots > 0:
            robot_active_time += (t - last_time) * current_robots
        current_robots += change
        if current_robots > peak_robots:
            peak_robots = current_robots
        last_time = t

    avg_robots = robot_active_time / max(makespan_calc, last_time) if max(makespan_calc, last_time) > 0 else 0.0

    # 2. Avg Picker Idle Time -> Picker Idle
    total_idle_time = 0
    total_span_time = 0
    for s in stations:
        picks = []
        for o in O:
            for k in K:
                for e in range(U.get(k, 0)):
                    ivp = P.get((o, s, k, e))
                    if ivp is not None and _iv_present(solution, ivp):
                        picks.append((_iv_start(solution, ivp), _iv_end(solution, ivp)))
        if not picks:
            continue
        picks.sort(key=lambda x: x[0])
        first_pick = picks[0][0]
        last_pick = max(p[1] for p in picks)
        span = last_pick - first_pick
        if span > 0:
            # Merge overlapping intervals to avoid double-counting concurrent picks
            merged_end = picks[0][0]
            active_pick_time = 0
            for ps, pe in picks:
                if ps >= merged_end:
                    active_pick_time += pe - ps
                    merged_end = pe
                elif pe > merged_end:
                    active_pick_time += pe - merged_end
                    merged_end = pe
            total_span_time += span
            total_idle_time += (span - active_pick_time)

    # Calculate picker idle over the active picking spans
    picker_idle_pct = total_idle_time / total_span_time if total_span_time > 0 else 0.0

    # 3. SKU Re-fetch Ratio
    sku_fetches = {}
    for k in K:
        fetches = 0
        for e in range(U.get(k, 0)):
            for s in S:
                ivf = F.get((s, k, e))
                if ivf is not None and _iv_present(solution, ivf):
                    fetches += 1
        if fetches > 0:
            sku_fetches[k] = fetches

    sku_refetch_ratio = sum(sku_fetches.values()) / len(sku_fetches) if sku_fetches else 0.0

    # 4. Lane Occupancy Ratio — denominator is active schedule window per station
    lane_active_time = 0
    schedule_first = None
    schedule_last = None
    for o in O:
        for s in S:
            iv = I_os.get((o, s))
            if iv is not None and _iv_present(solution, iv):
                st, en = _iv_start(solution, iv), _iv_end(solution, iv)
                lane_active_time += en - st
                schedule_first = st if schedule_first is None else min(schedule_first, st)
                schedule_last = en if schedule_last is None else max(schedule_last, en)

    active_window = (schedule_last - schedule_first) if (schedule_first is not None and schedule_last !=
                                                         schedule_first) else 1
    total_lane_capacity_time = len(stations) * len(L) * active_window if len(L) > 0 else 1
    lane_occupancy = lane_active_time / total_lane_capacity_time if len(L) > 0 else 0.0

    # 5. Avg Order Completion Time
    order_times = []
    for o in O:
        for s in S:
            iv = I_os.get((o, s))
            if iv is not None and _iv_present(solution, iv):
                order_times.append(_iv_end(solution, iv) - _iv_start(solution, iv))

    avg_order_time = sum(order_times) / len(order_times) if order_times else 0.0

    # ---- figure scaffold: one subplot per used station + 1 for global util ----
    # Break each station into 2 subplots: Gantt and Util, with an empty gap row between stations.
    total_subplots = (len(stations) * 3) - 1  # Gantt + Util + Gap for all except last station
    if utilization_plot_top or not utilization_plot_top:
        total_subplots += 2

    titles = []
    row_heights = []
    specs = []

    if utilization_plot_top:
        titles.append("Global Robot Fleet Utilization")
        row_heights.extend([0.3, 0.12])
        specs.extend([[{"type": "xy"}], [None]])
        global_row = 1
        station_start_row = 3
    else:
        global_row = total_subplots
        station_start_row = 1

    for i, s in enumerate(stations):
        titles.extend([f"Station S{s+1}", ""])
        row_heights.extend([0.6, 0.2])
        specs.extend([[{"type": "xy"}], [{"type": "xy"}]])
        if i < len(stations) - 1:
            row_heights.append(0.12)
            specs.append([None])

    if not utilization_plot_top:
        titles.append("Global Robot Fleet Utilization")
        row_heights.extend([0.12, 0.3])
        specs.extend([[None], [{"type": "xy"}]])

    fig = make_subplots(rows=total_subplots, cols=1, shared_xaxes=True,
                        vertical_spacing=0.01,  # actual spacing between adjacent (Gantt and Util) subplots
                        row_heights=row_heights,
                        subplot_titles=titles,
                        specs=specs)

    # Per-subplot y category order
    y_order_by_row = {i+1: [] for i in range(total_subplots)}

    def y_add_row(row_idx: int, cat: str):
        lst = y_order_by_row[row_idx]
        if cat not in lst:
            lst.append(cat)

    # ---- station-level rows: Fetch/Return lanes, Bin, Pick ----
    fetch_lane_map = {}  # (s, k, e) -> lane
    global_robot_start = 1
    for s_idx, s in enumerate(stations):
        row = station_start_row + (s_idx * 3)

        # Fetch/Return lanes (packed together to avoid overlap)
        fr_events = []
        for k in K:
            for e in range(U.get(k, 0)):
                # Fetch
                ivf = F.get((s, k, e))
                if ivf is not None and _iv_present(solution, ivf):
                    fr_events.append({
                        'type': 'fetch',
                        'start': _iv_start(solution, ivf),
                        'end': _iv_end(solution, ivf),
                        'sku': k,
                        'copy': e
                    })
                # Return
                ivr = R.get((s, k, e))
                if ivr is not None and _iv_present(solution, ivr):
                    fr_events.append({
                        'type': 'return',
                        'start': _iv_start(solution, ivr),
                        'end': _iv_end(solution, ivr),
                        'sku': k,
                        'copy': e
                    })

        # Pack all F/R events
        packed_fr = _pack_lanes([(e['start'], e['end'], e) for e in fr_events])

        max_lane_in_station = -1
        if packed_fr:
            max_lane_in_station = max(p[3] for p in packed_fr)

        for st, en, evt, lane in packed_fr:
            y_fr_lane = f"S{s+1} · Robot {global_robot_start + lane}"
            y_add_row(row, y_fr_lane)
            k, e = evt['sku'], evt['copy']

            if evt['type'] == 'fetch':
                _add_bar(fig, y=y_fr_lane, start=st, end=en, color=COLORS["fetch"],
                         hover=f"Station: S{s+1}<br>SKU: {k}<br>Visit: {e}",
                         type_label="Fetch",
                        #  type_label=f"Fetch F[{s},{k},{e}]",
                         legendgroup="fetch", name="Fetch", meta_station=s, row=row, text=f"ID: {k}")
            else:
                _add_bar(fig, y=y_fr_lane, start=st, end=en, color=COLORS["return"],
                         hover=f"Station: S{s+1}<br>SKU: {k}<br>Visit: {e}",
                         type_label="Return",
                        #  type_label=f"Return R[{s},{k},{e}]",
                         legendgroup="return", name="Return", meta_station=s, row=row, text=f"ID: {k}")

        if max_lane_in_station != -1:
            global_robot_start += (max_lane_in_station + 1)

        # Bin presence (single row; no overlap by model)
        y_bin = f"S{s+1} · Bin"
        y_add_row(row, y_bin)
        for k in K:
            for e in range(U.get(k, 0)):
                ivb = B.get((s, k, e))
                if ivb is not None and _iv_present(solution, ivb):
                    st, en = _iv_start(solution, ivb), _iv_end(solution, ivb)
                    _add_bar(fig, y=y_bin, start=st, end=en, color=COLORS["bin"],
                             hover=f"Station: S{s+1}<br>SKU: {k}<br>Visit: {e}",
                             type_label="Bin Presence",
                            #  type_label=f"Bin B[{s},{k},{e}]",
                             legendgroup="bin", name="Bin", meta_station=s, row=row, text=f"ID: {k}")

        # Pick row (still useful to see pick sequencing)
        y_pick = f"S{s+1} · Pick"
        y_add_row(row, y_pick)
        for k in K:
            for e in range(U.get(k, 0)):
               # --- MODIFICATION START ---
                # In v3, P is indexed by (o, s, k, e). We must iterate over orders 'o'
                # to find all pick intervals that use this (s, k, e) copy.
                # All will be plotted on the same y_pick row.
                for o in O:
                    ivp = P.get((o, s, k, e))  # <-- Correct v3 key
                    if ivp is not None and _iv_present(solution, ivp):
                        st, en = _iv_start(solution, ivp), _iv_end(solution, ivp)

                        _add_bar(fig, y=y_pick, start=st, end=en, color=COLORS["pick"],
                                 hover=f"Order: {o}<br>Station: S{s+1}<br>SKU: {k}<br>Visit: {e}",
                                 type_label="Pick",
                                #  type_label=f"Pick P[{o},{s},{k},{e}]",
                                 legendgroup="pick", name="Pick", meta_station=s, row=row, text=f"ID: {k}")

    # ---- lane rows: per-lane orders + thin consumptions inside order bar ----
    for s_idx, s in enumerate(stations):
        row = station_start_row + (s_idx * 3)
        for ln in L:
            y_lane = f"S{s+1} · Lane {ln+1}"
            # Only add the lane row if it's actually used by an order
            has_order_on_lane = any(assign.get(o) == (s, ln) for o in O)

            if not has_order_on_lane and len(L) > 0:  # If L=0, we must add at least one lane
                if not any(assign.get(o) == (s, 0) for o in O):  # ...unless we are packing to lane 0
                    continue

            y_add_row(row, y_lane)

            for o in O:
                if assign.get(o) == (s, ln):
                    iv = I_os[(o, s)]
                    st, en = _iv_start(solution, iv), _iv_end(solution, iv)

                    # Calculate SKU list in pick order
                    picks = []
                    for k_req in orders_req[o]:
                        for e_req in range(U.get(k_req, 0)):
                            ivp = P.get((o, s, k_req, e_req))
                            if ivp is not None and _iv_present(solution, ivp):
                                picks.append((_iv_start(solution, ivp), k_req))
                    picks.sort(key=lambda x: x[0])
                    sku_list_str = ", ".join(str(p[1]) for p in picks)

                    _add_bar(fig, y=y_lane, start=st, end=en, color=COLORS["order"],
                             hover=f"Station: S{s+1} · Lane {ln+1}<br>SKUs: {orders_req[o]}",
                             type_label=f"Order {o}",
                             legendgroup="order", name="Order", meta_station=s, row=row, text=f"[{sku_list_str}]")

                    if include_consumptions:
                        for k in orders_req[o]:
                            Civ = C[(o, k, s)]
                            if _iv_present(solution, Civ):
                                cs, ce = _iv_start(solution, Civ), _iv_end(solution, Civ)
                                _add_cons_line(fig, y=y_lane, start=cs, end=ce, color=COLORS["cons"],
                                               hover=f"Order: {o}<br>SKU: {k}<br>Station: S{s+1}",
                                               type_label="Consumption",
                                            #    type_label=f"C[{o},{k},S{s+1}]",
                                               meta_station=s, row=row, width_px=CONS_LINE_WIDTH_PX)

    # ---- station sparklines (stacked area) ----
    for s_idx, s in enumerate(stations):
        util_row = station_start_row + (s_idx * 3) + 1

        events = {"order": [], "fetch": [], "return": [], "pick": []}

        for o in O:
            iv = I_os.get((o, s))
            if iv is not None and _iv_present(solution, iv):
                events["order"].extend([(_iv_start(solution, iv), 1), (_iv_end(solution, iv), -1)])

            for k in K:
                for e in range(U.get(k, 0)):
                    ivp = P.get((o, s, k, e))
                    if ivp is not None and _iv_present(solution, ivp):
                        events["pick"].extend([(_iv_start(solution, ivp), 1), (_iv_end(solution, ivp), -1)])

        for k in K:
            for e in range(U.get(k, 0)):
                ivf = F.get((s, k, e))
                if ivf is not None and _iv_present(solution, ivf):
                    events["fetch"].extend([(_iv_start(solution, ivf), 1), (_iv_end(solution, ivf), -1)])
                ivr = R.get((s, k, e))
                if ivr is not None and _iv_present(solution, ivr):
                    events["return"].extend([(_iv_start(solution, ivr), 1), (_iv_end(solution, ivr), -1)])

        all_times = set([0, makespan_calc])
        for etype in events:
            events[etype].sort(key=lambda x: x[0])
            for t, _ in events[etype]:
                all_times.add(t)
        sorted_times = sorted(list(all_times))

        cur_vals = {"order": 0, "fetch": 0, "return": 0, "pick": 0}
        idx = {"order": 0, "fetch": 0, "return": 0, "pick": 0}

        stepped_t = []
        stepped_y = {"order": [], "fetch": [], "return": [], "pick": []}

        for t in sorted_times:
            if t > 0:
                stepped_t.append(t)
                for etype in cur_vals:
                    stepped_y[etype].append(cur_vals[etype])

            for etype in events:
                while idx[etype] < len(events[etype]) and events[etype][idx[etype]][0] == t:
                    cur_vals[etype] += events[etype][idx[etype]][1]
                    idx[etype] += 1

            stepped_t.append(t)
            for etype in cur_vals:
                stepped_y[etype].append(cur_vals[etype])

        labels = {
            "order": "Util: Lane Occ",
            "pick": "Util: Pick",
            "return": "Util: Return",
            "fetch": "Util: Fetch"
        }
        ranks = {
            "order": 1001,
            "pick": 1002,
            "return": 1003,
            "fetch": 1004
        }

        show_leg = (s_idx == 0)

        for etype in ["order", "pick", "return", "fetch"]:
            fig.add_trace(go.Scatter(
                x=stepped_t, y=stepped_y[etype],
                mode='lines',
                line=dict(width=0.5, color=COLORS[etype]),
                fill='tonexty',
                fillcolor=COLORS[etype],
                stackgroup=f'S{s}_util',
                hoverinfo="x+y+name",
                showlegend=show_leg,
                legendgroup=f"util_{etype}",
                legendrank=ranks[etype],
                name=labels[etype],

                opacity=0.6
            ), row=util_row, col=1)

        fig.update_yaxes(title_text="Station Util", row=util_row, col=1, showgrid=True, zeroline=True)

    # ---- global fleet utilization plot ----
    t_vals = [0]
    y_vals = [0]
    cur = 0
    for t, change in robot_events:
        t_vals.append(t)
        y_vals.append(cur)
        cur += change
        t_vals.append(t)
        y_vals.append(cur)
    t_vals.append(makespan_calc)
    y_vals.append(cur)

    fig.add_trace(go.Scatter(
        x=t_vals, y=y_vals,
        mode='lines',
        line=dict(width=2, color=COLORS["fetch"], shape='hv'),
        fill='tozeroy',
        fillcolor='rgba(255, 140, 0, 0.3)',
        hoverinfo='x+y',
        name="Active Robots",
        showlegend=False
    ), row=global_row, col=1)

    fig.update_yaxes(title_text="Robot Count", row=global_row, col=1)

    move_cap = handles.get("move_cap")
    if move_cap is not None:
        fig.add_hline(y=move_cap, line_dash="dash", line_color="black", row=global_row, col=1,
                      annotation_text="RobotCap", annotation_position="top left",
                      annotation_font=dict(color="black"))

    # ---- layout ----
    num_stations = len(stations)
    num_lanes = len(L)
    num_orders = len(O)
    num_skus = len(K)

    metrics_html = (
        f"<br><span style='font-size: 11px; color: #555; line-height: 1.2;'>"
        f"<b>Peak Robots:</b> {peak_robots} (Avg: {avg_robots:.1f}) | "
        f"<b>Picker Idle:</b> {picker_idle_pct:.1%} | "
        f"<b>SKU Re-fetch Ratio:</b> {sku_refetch_ratio:.2f} | "
        # f"<b>Lane Occupancy:</b> {lane_occupancy:.1%} | "
        f"<b>Avg Order Time:</b> {avg_order_time:.1f}s"
        f"</span>"
    )

    move_cap = handles.get("move_cap", handles.get("MoveCap"))
    robot_cap_str = f" | RobotCap: {move_cap}" if move_cap is not None else ""

    title_text = (f"Order Batching & Station Scheduling (makespan: {makespan})<br>"
                  f"<span style='font-size: 12px; line-height: 1.2;'>Stations: {num_stations} | Lanes: {num_lanes} | "
                  f"Orders: {num_orders} | SKUs: {num_skus}{robot_cap_str}</span>"
                  f"{metrics_html}")

    fig.update_layout(
        title=title_text,
        barmode="overlay",
        plot_bgcolor=BG, paper_bgcolor=BG, font=FONT,
        legend_title="Legend",
        margin=dict(l=90, r=20, t=140, b=40),
    )

    # Per-subplot y category ordering + grid
    for row, cats in y_order_by_row.items():
        # Filter categories
        robots = [c for c in cats if "Robot" in c]
        bins = [c for c in cats if "Bin" in c]
        picks = [c for c in cats if "Pick" in c]
        lanes = [c for c in cats if "Lane" in c]

        # Sort robots and lanes by number to ensure correct order
        # Assumes format "S... · Robot X" and "S... · Lane X"
        def get_num(s, key):
            try:
                return int(s.split(key)[1])
            except:
                return 0

        robots.sort(key=lambda x: get_num(x, "Robot "))
        lanes.sort(key=lambda x: get_num(x, "Lane "))

        # Desired Visual Top->Bottom: Lane 1..N, Pick, Bin, Robot 1..N, Utilization
        # categoryarray (Bottom->Top): Utilization, Robot N..1, Bin, Pick, Lane N..1

        utils = [c for c in cats if "Utilization" in c]
        new_cats = utils + list(reversed(robots)) + bins + picks + list(reversed(lanes))

        fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR,
                         categoryorder="array", categoryarray=new_cats,
                         row=row, col=1)

    # Add legend stubs using a valid Y category from the first row (if available)
    # This avoids adding a "legend" label to the Y axis.
    if y_order_by_row[1]:
        dummy_y = y_order_by_row[1][0]  # Use the first category (which will be at the top or bottom)

        def _add_stub(name, color):
            fig.add_trace(go.Bar(
                x=[0], y=[dummy_y], orientation="h", name=name,
                marker_color=color, showlegend=True, hoverinfo="skip", visible=True
            ), row=1, col=1)

        _add_stub("Order window", COLORS["order"])
        _add_stub("Fetch", COLORS["fetch"])
        _add_stub("Return", COLORS["return"])
        _add_stub("Bin (at station)", COLORS["bin"])
        _add_stub("Pick", COLORS["pick"])
        if include_consumptions:
            _add_stub("Consumption from Pick", COLORS["cons"])

    # makespan marker across all rows
    fig.add_vline(x=makespan, line_width=2, line_dash="dot", line_color="#555", row="all", col=1)

    # dynamic height
    total_rows = sum(len(cats) for cats in y_order_by_row.values())
    fig.update_layout(height=max(850, 60 * total_rows + 140))

    fig.update_layout(
        hovermode="closest",
        hoverdistance=12)
    # faint vertical spikeline helps reading exact times
    fig.update_xaxes(showspikes=True, spikemode="across", spikethickness=1,
                     spikecolor="#888", spikedash="dot")

    # Add X-axis labels to the bottom of each station's utilization graph
    for s_idx in range(len(stations)):
        util_row = station_start_row + (s_idx * 3) + 1
        fig.update_xaxes(showticklabels=True, title_text="Time", row=util_row, col=1)

    # Add Time label to the global fleet utilization graph if it's at the bottom
    if not utilization_plot_top:
        fig.update_xaxes(title_text="Time", row=global_row, col=1)

    # Station filter dropdown

    if show:
        fig.show()
    return fig


def write_html(fig: go.Figure, path: str = "schedule.html") -> str:
    fig.write_html(path, include_plotlyjs="cdn")
    return path
