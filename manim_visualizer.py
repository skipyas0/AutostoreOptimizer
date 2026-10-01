import re
from manim import *

# Override default background color
config.background_color = "#FFFFFF"

COLORS = {
    "order": "#6C5CE7",
    "fetch": "#FFD166",  # Yellow
    "pick": "#E63946",
    "cons": "#FF0015",
    "return": "#1D4ED8",
    "bin": "#6C757D",
    "grid": "#E9ECEF",
    "text": "#000000",
    "accent": "#00E5FF"  # Cyan
}

VTYPE_TO_COLOR_KEY = {
    "I_os_lane": "order",
    "C": "cons",
    "P": "pick",
    "F": "fetch",
    "R": "return",
    "B": "bin"
}

def parse_varname(varname):
    match = re.match(r"([a-zA-Z_0-9]+)\[(.*)\]", varname)
    if not match:
        return varname, []
    vtype = match.group(1)
    indices = [int(x.strip()) for x in match.group(2).split(',')]
    return vtype, indices

class IntervalVar(VGroup):
    def __init__(self, varname, start, end, row_idx, time_scale, y_positions, instance):
        super().__init__()
        self.varname = varname
        self.vtype, self.indices = parse_varname(varname)
        self.start = start
        self.end = end
        
        color_key = VTYPE_TO_COLOR_KEY.get(self.vtype, "")
        self.color = COLORS.get(color_key, BLACK)
            
        width = max(0.01, (end - start) * time_scale)
        z_idx = 1
        if self.vtype == "C":
            z_idx = 2

        if len(y_positions) > 1:
            y_step = abs(y_positions[0] - y_positions[1])
            self.base_height = y_step * 0.8
        else:
            self.base_height = 0.4

        self.rect = Rectangle(
            width=width,
            height=self.base_height,
            fill_color=self.color,
            fill_opacity=1.0,
            stroke_color=self.color,
            stroke_width=1,
            z_index=z_idx
        )
        self.add(self.rect)
        
        if self.vtype == "C":
            self.rect.stretch_to_fit_height(self.base_height * 0.2)
            self.rect.set_fill(COLORS["cons"])
            self.rect.set_stroke(width=0)
            
        self.move_to_pos(start, end, row_idx, time_scale, y_positions)
        
    def move_to_pos(self, start, end, row_idx, time_scale, y_positions):
        x = -6.0 + (start + end) / 2 * time_scale
        y = y_positions[row_idx]
        if self.vtype == "C":
            y -= self.base_height * 0.4
        self.move_to([x, y, 0])
        
    def update_state(self, start, end, row_idx, time_scale, y_positions, is_frozen, is_changed):
        self.start = start
        self.end = end
        
        new_x = -6.0 + (start + end) / 2 * time_scale
        new_y = y_positions[row_idx]
        if self.vtype == "C":
            new_y -= self.base_height * 0.4
            
        new_width = max(0.01, (end - start) * time_scale)
        
        target_opacity = 0.4 if is_frozen else 1.0
        
        if is_frozen:
            from manim.utils.color import interpolate_color, ManimColor
            target_fill_color = interpolate_color(ManimColor(self.color), ManimColor("#90CAF9"), 0.6) # blue tint
        else:
            target_fill_color = self.color if self.vtype != "C" else COLORS["cons"]
            
        target_stroke_color = COLORS["accent"] if is_changed else (GRAY if is_frozen else self.color)
        target_stroke_width = 3 if is_changed else 1
        if self.vtype == "C":
            target_stroke_width = 0
            
        anims = []
        anims.append(
            self.rect.animate.stretch_to_fit_width(new_width)
            .move_to([new_x, new_y, 0])
            .set_fill(color=target_fill_color, opacity=target_opacity)
            .set_stroke(color=target_stroke_color, width=target_stroke_width)
        )
        return AnimationGroup(*anims)
        
class LNSOptimizationScene(Scene):
    def __init__(self, instance, initial_solution, deltas, frozen_vars, objectives, improvements, makespans, fetches, flows, best_makespans, best_fetches, best_flows, cp_objs, statuses, strategies, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.instance = instance
        self.initial_solution = initial_solution
        self.deltas = deltas
        self.frozen_vars = frozen_vars
        self.objectives = objectives
        self.improvements = improvements
        self.makespans = makespans
        self.fetches = fetches
        self.flows = flows
        self.best_makespans = best_makespans
        self.best_fetches = best_fetches
        self.best_flows = best_flows
        self.cp_objs = cp_objs
        self.statuses = statuses
        self.strategies = strategies

    def construct(self):
        # 1. Determine timeline scaling
        max_time = max(self.makespans) * 1.1 if self.makespans else 100
        time_scale = 7.0 / max_time # X spans -6.0 to 1.0 (width 7)
        
        # 2. Compute dynamic robot lane packing logic
        def get_robot_assignments(state, previous_assignment):
            fr_events = []
            for k_var, v_state in state.items():
                vtype, indices = parse_varname(k_var)
                if not indices: continue
                if vtype in ["F", "R"]:
                    s = indices[0]
                    if s == 0:
                        fr_events.append({
                            "id": (indices[1], indices[2]), 
                            "start": v_state["start"], 
                            "end": v_state["end"]
                        })
            spans = {}
            for ev in fr_events:
                idx = ev["id"]
                if idx not in spans:
                    spans[idx] = [ev["start"], ev["end"]]
                else:
                    spans[idx][0] = min(spans[idx][0], ev["start"])
                    spans[idx][1] = max(spans[idx][1], ev["end"])
            
            assignment = {}
            lanes_content = {}
            
            def overlap(span1, span2):
                return not (span1[1] <= span2[0] or span1[0] >= span2[1])

            unplaced = []
            # Pass 1: Retain previous assignment if no overlap
            for idx, span in sorted(spans.items(), key=lambda x: x[0]):
                placed = False
                if idx in previous_assignment:
                    prev_lane = previous_assignment[idx]
                    lane_spans = lanes_content.get(prev_lane, [])
                    if not any(overlap(span, s) for s in lane_spans):
                        assignment[idx] = prev_lane
                        if prev_lane not in lanes_content:
                            lanes_content[prev_lane] = []
                        lanes_content[prev_lane].append(span)
                        placed = True
                if not placed:
                    unplaced.append((idx, span))
                    
            # Pass 2: Greedily place the rest by start time
            unplaced.sort(key=lambda x: x[1][0])
            for idx, span in unplaced:
                lane_idx = 0
                while True:
                    lane_spans = lanes_content.get(lane_idx, [])
                    if not any(overlap(span, s) for s in lane_spans):
                        assignment[idx] = lane_idx
                        if lane_idx not in lanes_content:
                            lanes_content[lane_idx] = []
                        lanes_content[lane_idx].append(span)
                        break
                    lane_idx += 1
                    
            max_l = max(assignment.values()) + 1 if assignment else 0
            return assignment, max_l

        current_state = {}
        for k, v in self.initial_solution.items():
            if isinstance(v, dict) and v.get("present"):
                current_state[k] = v
                
        max_lanes = 1
        robot_assignments = {}
        robot_assignments, l_count = get_robot_assignments(current_state, robot_assignments)
        max_lanes = max(max_lanes, l_count)
        
        for d in self.deltas:
            for k, v in d.items():
                if isinstance(v, dict):
                    if v.get("present"): current_state[k] = v
                    else: current_state.pop(k, None)
            robot_assignments, l_count = get_robot_assignments(current_state, robot_assignments)
            max_lanes = max(max_lanes, l_count)
            
        num_order_lanes = len(self.instance.L) if hasattr(self.instance, "L") else 1
        num_tracks = num_order_lanes + 2 + max_lanes
        
        # 3. Y-axis layout
        y_max = 2.0
        y_min = -3.5
        y_step = (y_max - y_min) / max(1, num_tracks - 1)
        y_positions = [y_max - i * y_step for i in range(num_tracks)]
        
        # Draw background and grid
        tick_step = max_time / 10
        for i in range(11):
            x = -6.0 + (i * tick_step) * time_scale
            self.add(Line(start=[x, y_min - 0.5, 0], end=[x, y_max + 0.5, 0], color=COLORS["grid"], stroke_width=1))
            self.add(Text(f"{int(i * tick_step)}", font_size=12, color=BLACK).move_to([x, y_min - 0.3, 0]))
            
        for y in y_positions:
            self.add(Line(start=[-6.0, y, 0], end=[1.0, y, 0], color=COLORS["grid"], stroke_width=1))
            
        # Add a subtle makespan marker
        makespan_marker = DashedLine(start=[-6.0, y_min - 0.5, 0], end=[-6.0, y_max + 0.5, 0], color=BLACK, dash_length=0.1)
        self.add(makespan_marker)
            
        # Draw Y-axis labels
        x_lbl = -6.7
        f_size = max(6, min(14, int(35 * y_step)))
        for i in range(num_order_lanes):
            self.add(Text(f"S0 · Lane {i}", font_size=f_size, color=BLACK).move_to([x_lbl, y_positions[i], 0]))
        self.add(Text("S0 · Pick", font_size=f_size, color=BLACK).move_to([x_lbl, y_positions[num_order_lanes], 0]))
        self.add(Text("S0 · Bin", font_size=f_size, color=BLACK).move_to([x_lbl, y_positions[num_order_lanes+1], 0]))
        for i in range(max_lanes):
            self.add(Text(f"S0 · Robot {i}", font_size=f_size, color=BLACK).move_to([x_lbl, y_positions[num_order_lanes+2+i], 0]))
            
        # Add KPI Graphs on the right
        iters = len(self.makespans)
        
        def get_axis_config(data):
            vmin, vmax = min(data), max(data)
            if vmin == vmax:
                return [vmin - 1, vmax + 1, 1]
            step = (vmax - vmin) / 3.0
            step = max(1, int(step))
            return [vmin, vmax + step * 0.1, step]

        graph_x, graph_w, graph_h = 4.0, 5.0, 1.4
        axis_cfg = {"color": BLACK}
        x_cfg = {"include_ticks": False, "include_numbers": False, "color": BLACK}
        y_cfg = {"include_ticks": True, "include_numbers": True, "font_size": 14, "color": BLACK, "decimal_number_config": {"color": BLACK}}

        def get_kpi_title(name, cur, best, color):
            val_txt = f"{cur:.1f}" if isinstance(cur, float) else f"{cur}"
            best_txt = f"{best:.1f}" if isinstance(best, float) else f"{best}"
            return Text(f"{name} (Curr: {val_txt}, Best: {best_txt})", font_size=14, color=color)

        ax_makespan = Axes(x_range=[0, iters, max(1, iters//5)], y_range=get_axis_config(self.makespans),
                           x_length=graph_w, y_length=graph_h, tips=False, axis_config=axis_cfg, x_axis_config=x_cfg, y_axis_config=y_cfg).move_to([graph_x, 2.0, 0])
        title_makespan = get_kpi_title("Makespan", self.makespans[0], self.makespans[0], COLORS["pick"]).next_to(ax_makespan, UP, buff=0.1).align_to(ax_makespan, LEFT)
        self.add(ax_makespan, title_makespan)

        ax_fetch = Axes(x_range=[0, iters, max(1, iters//5)], y_range=get_axis_config(self.fetches),
                        x_length=graph_w, y_length=graph_h, tips=False, axis_config=axis_cfg, x_axis_config=x_cfg, y_axis_config=y_cfg).move_to([graph_x, 0.0, 0])
        title_fetch = get_kpi_title("Bin Fetches", self.fetches[0], self.fetches[0], COLORS["fetch"]).next_to(ax_fetch, UP, buff=0.1).align_to(ax_fetch, LEFT)
        self.add(ax_fetch, title_fetch)

        ax_flow = Axes(x_range=[0, iters, max(1, iters//5)], y_range=get_axis_config(self.flows),
                       x_length=graph_w, y_length=graph_h, tips=False, axis_config=axis_cfg, x_axis_config=x_cfg, y_axis_config=y_cfg).move_to([graph_x, -2.0, 0])
        title_flow = get_kpi_title("Total Flow Time", self.flows[0], self.flows[0], COLORS["order"]).next_to(ax_flow, UP, buff=0.1).align_to(ax_flow, LEFT)
        self.add(ax_flow, title_flow)

        pt_makespan = ax_makespan.c2p(0, self.makespans[0])
        pt_fetch = ax_fetch.c2p(0, self.fetches[0])
        pt_flow = ax_flow.c2p(0, self.flows[0])
            
        # HUD Helper
        def get_hud(iter_idx, obj_txt, strat_txt, stat_txt):
            obj_txt = str(obj_txt) if obj_txt else "-"
            strat_txt = str(strat_txt) if strat_txt else "-"
            stat_txt = str(stat_txt) if stat_txt else "-"
            
            obj_clean = obj_txt.replace("_", " ").title()
            strat_clean = strat_txt.replace("_", " ").title()
            stat_clean = stat_txt.replace("_", " ").title()
            
            if "Makespan" in obj_clean: obj_color = COLORS["pick"]
            elif "Fetch" in obj_clean: obj_color = COLORS["fetch"]
            elif "Flow" in obj_clean: obj_color = COLORS["order"]
            else: obj_color = BLACK

            y_pos_1 = 3.8
            y_pos_2 = 3.4
            
            t_iter = Text(f"Iter: {iter_idx:03d}/{iters}", font="monospace", font_size=16, color=BLACK)
            t_iter.move_to([-6.9, y_pos_1, 0], aligned_edge=LEFT)
            
            t_obj_lbl = Text(" | Obj: ", font="monospace", font_size=16, color=BLACK)
            t_obj_lbl.move_to([-4.8, y_pos_1, 0], aligned_edge=LEFT)
            t_obj_val = Text(obj_clean, font="monospace", font_size=16, color=obj_color)
            t_obj_val.next_to(t_obj_lbl, RIGHT, buff=0.1, aligned_edge=DOWN)
            
            t_strat_lbl = Text("Strategy: ", font="monospace", font_size=16, color=BLACK)
            t_strat_lbl.move_to([-6.9, y_pos_2, 0], aligned_edge=LEFT)
            t_strat_val = Text(strat_clean, font="monospace", font_size=16, color=BLACK)
            t_strat_val.next_to(t_strat_lbl, RIGHT, buff=0.1, aligned_edge=DOWN)
            
            t_stat_lbl = Text(" | Status: ", font="monospace", font_size=16, color=BLACK)
            t_stat_lbl.move_to([0.2, y_pos_2, 0], aligned_edge=LEFT)
            t_stat_val = Text(stat_clean, font="monospace", font_size=16, color=BLACK)
            t_stat_val.next_to(t_stat_lbl, RIGHT, buff=0.1, aligned_edge=DOWN)
            
            return VGroup(t_iter, t_obj_lbl, t_obj_val, t_strat_lbl, t_strat_val, t_stat_lbl, t_stat_val)
            
        hud_vg = get_hud(0, "-", "-", "-")
        self.add(hud_vg)
        
        # Legend
        legend_labels = ["Order", "Pick", "Bin", "Fetch", "Return", "Cons"]
        legend_colors = [COLORS["order"], COLORS["pick"], COLORS["bin"], COLORS["fetch"], COLORS["return"], COLORS["cons"]]
        legend_group = VGroup()
        for lbl, col in zip(legend_labels, legend_colors):
            r = Rectangle(width=0.3, height=0.2, fill_color=col, fill_opacity=1, stroke_width=0)
            t = Text(lbl, font_size=14, color=BLACK)
            vg = VGroup(r, t).arrange(RIGHT, buff=0.1)
            legend_group.add(vg)
        legend_group.arrange(RIGHT, buff=0.4).to_corner(UL).shift(DOWN * 0.8)
        self.add(legend_group)

        def get_row_idx(varname, state_dict, robot_assigns):
            vtype, indices = parse_varname(varname)
            if not indices: return -1
            
            if vtype == "I_os_lane": s = indices[1]
            elif vtype == "C": s = indices[2]
            elif vtype == "P": s = indices[1]
            elif vtype in ["B", "F", "R"]: s = indices[0]
            else: return -1
            
            if s != 0: return -1 # Render station 0 only
            
            if vtype == "I_os_lane":
                return indices[2]
            elif vtype == "C":
                o = indices[0]
                for k_var in state_dict:
                    vt, ind = parse_varname(k_var)
                    if vt == "I_os_lane" and ind and ind[0] == o and ind[1] == 0:
                        return ind[2]
                return 0
            elif vtype == "P": return num_order_lanes
            elif vtype == "B": return num_order_lanes + 1
            elif vtype in ["F", "R"]:
                k_e = (indices[1], indices[2])
                return num_order_lanes + 2 + robot_assigns.get(k_e, 0)
            return -1
            
        current_state = {}
        for k, v in self.initial_solution.items():
            if isinstance(v, dict) and v.get("present"):
                current_state[k] = v
                
        robot_assignments = {}
        robot_assignments, _ = get_robot_assignments(current_state, robot_assignments)
        vgroup_map = {}
        
        for varname, state in current_state.items():
            row_idx = get_row_idx(varname, current_state, robot_assignments)
            if row_idx >= 0:
                iv = IntervalVar(varname, state["start"], state["end"], row_idx, time_scale, y_positions, self.instance)
                vgroup_map[varname] = iv
                self.add(iv)
                
        # Initial Makespan marker position
        current_makespan = self.makespans[0] if self.makespans else 0
        makespan_marker.move_to([-6.0 + current_makespan * time_scale, (y_min+y_max)/2, 0])
        self.wait(1)
        
        for i, (delta, frozen) in enumerate(zip(self.deltas, self.frozen_vars)):
            for varname, state in delta.items():
                if isinstance(state, dict):
                    if state.get("present"): current_state[varname] = state
                    else: current_state.pop(varname, None)
                    
            robot_assignments, _ = get_robot_assignments(current_state, robot_assignments)
            anims = []
            
            for varname, state in current_state.items():
                row_idx = get_row_idx(varname, current_state, robot_assignments)
                if row_idx < 0: continue
                
                is_frozen = varname in frozen
                is_changed = varname in delta
                
                if varname in vgroup_map:
                    anim = vgroup_map[varname].update_state(state["start"], state["end"], row_idx, time_scale, y_positions, is_frozen, is_changed)
                    anims.append(anim)
                else:
                    iv = IntervalVar(varname, state["start"], state["end"], row_idx, time_scale, y_positions, self.instance)
                    if is_frozen:
                        from manim.utils.color import interpolate_color, ManimColor
                        frozen_color = interpolate_color(ManimColor(iv.color), ManimColor("#90CAF9"), 0.6)
                        iv.rect.set_fill(color=frozen_color, opacity=0.4)
                        iv.rect.set_stroke(color=GRAY, width=1)
                    vgroup_map[varname] = iv
                    self.add(iv)
                    anims.append(FadeIn(iv))
                    
            for varname in list(vgroup_map.keys()):
                if varname not in current_state:
                    anims.append(FadeOut(vgroup_map[varname]))
                    del vgroup_map[varname]
                    
            # HUD Update
            obj_txt = self.cp_objs[i] if i < len(self.cp_objs) else "-"
            strat_txt = self.strategies[i] if i < len(self.strategies) else "-"
            stat_txt = self.statuses[i] if i < len(self.statuses) else "-"
            
            new_hud = get_hud(i + 1, obj_txt, strat_txt, stat_txt)
            anims.append(hud_vg.animate.become(new_hud))
            
            # KPI Titles Update
            def get_best(arr, idx):
                valid = [v for v in arr[:idx] if v >= 0]
                return min(valid) if valid else 0
                
            c_m = self.makespans[i+1] if i+1 < len(self.makespans) else self.makespans[-1]
            b_m = get_best(self.makespans, i+2)
            anims.append(title_makespan.animate.become(get_kpi_title("Makespan", c_m, b_m, COLORS["pick"]).next_to(ax_makespan, UP, buff=0.1).align_to(ax_makespan, LEFT)))
            
            c_f = self.fetches[i+1] if i+1 < len(self.fetches) else self.fetches[-1]
            b_f = get_best(self.fetches, i+2)
            anims.append(title_fetch.animate.become(get_kpi_title("Bin Fetches", c_f, b_f, COLORS["fetch"]).next_to(ax_fetch, UP, buff=0.1).align_to(ax_fetch, LEFT)))
            
            c_fl = self.flows[i+1] if i+1 < len(self.flows) else self.flows[-1]
            b_fl = get_best(self.flows, i+2)
            anims.append(title_flow.animate.become(get_kpi_title("Total Flow Time", c_fl, b_fl, COLORS["order"]).next_to(ax_flow, UP, buff=0.1).align_to(ax_flow, LEFT)))
            
            # Makespan marker
            cur_makespan = self.makespans[i+1] if i+1 < len(self.makespans) else self.makespans[-1]
            anims.append(makespan_marker.animate.move_to([-6.0 + cur_makespan * time_scale, (y_min+y_max)/2, 0]))
            
            # KPI Line Graphs
            if i < iters:
                pt_m_new = ax_makespan.c2p(i+1, self.makespans[i+1] if i+1 < len(self.makespans) else self.makespans[-1])
                anims.append(Create(Line(pt_makespan, pt_m_new, color=COLORS["pick"], stroke_width=3)))
                pt_makespan = pt_m_new

                pt_f_new = ax_fetch.c2p(i+1, self.fetches[i+1] if i+1 < len(self.fetches) else self.fetches[-1])
                anims.append(Create(Line(pt_fetch, pt_f_new, color=COLORS["fetch"], stroke_width=3)))
                pt_fetch = pt_f_new

                pt_fl_new = ax_flow.c2p(i+1, self.flows[i+1] if i+1 < len(self.flows) else self.flows[-1])
                anims.append(Create(Line(pt_flow, pt_fl_new, color=COLORS["order"], stroke_width=3)))
                pt_flow = pt_fl_new
            
            if anims:
                self.play(AnimationGroup(*anims), run_time=1.0)
                
            reset_anims = []
            for varname, iv in vgroup_map.items():
                if varname in delta:
                    is_frozen = varname in frozen
                    reset_color = GRAY if is_frozen else iv.color
                    reset_anims.append(iv.rect.animate.set_stroke(color=reset_color, width=1))
            
            if reset_anims:
                self.play(AnimationGroup(*reset_anims), run_time=0.5)
