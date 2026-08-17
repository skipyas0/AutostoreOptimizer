#!/usr/bin/env python3
import argparse
from collections import defaultdict

from ortools.sat.python import cp_model

from datagen import generate_data
from instance import Instance

try:
    from schedule_visualizer import plot_schedule, write_html
except ImportError:
    print("Warning: schedule_visualizer2 not found. Visualization will be disabled.")
    plot_schedule = None
    write_html = None


# --- Helper class for CP-SAT Optional Intervals ---
class OptInterval:
    def __init__(self, model, name, horizon, size=None):
        self.name = name
        self.start = model.NewIntVar(0, horizon, f"{name}_start")
        self.end = model.NewIntVar(0, horizon, f"{name}_end")
        self.pres = model.NewBoolVar(f"{name}_pres")

        if size is None:
            self.size = model.NewIntVar(0, horizon, f"{name}_size")
            self.interval = model.NewOptionalIntervalVar(
                self.start, self.size, self.end, self.pres, f"{name}_int"
            )
        else:
            self.size = size
            self.interval = model.NewOptionalIntervalVar(
                self.start, self.size, self.end, self.pres, f"{name}_int"
            )


# --------------------------
# Model building (OR-Tools)
# --------------------------
def build_model(instance, *args, **kwargs):
    if not isinstance(instance, Instance):
        S, L, K, orders_req, rt, p = args[0:6]
        rt_return = kwargs.get("rt_return", args[5] if len(args) > 5 else None)
        add_symmetry_breaking = kwargs.get(
            "add_symmetry_breaking", args[6] if len(args) > 6 else True
        )
        horizon = kwargs.get("horizon", args[7] if len(args) > 7 else 0)
        move_cap = kwargs.get("move_cap", args[8] if len(args) > 8 else None)
        N = kwargs.get("N", args[9] if len(args) > 9 else None)
        instance = Instance(S, L, K, orders_req, rt, p, N or {}, rt_ret=rt_return)
    else:
        rt_return = kwargs.get("rt_return", args[0] if len(args) > 0 else None)
        add_symmetry_breaking = kwargs.get(
            "add_symmetry_breaking", args[1] if len(args) > 1 else True
        )
        horizon = kwargs.get("horizon", args[2] if len(args) > 2 else 0)
        move_cap = kwargs.get("move_cap", args[3] if len(args) > 3 else None)

    S, L, K, orders_req, rt, p, N = instance
    if rt_return is None:
        rt_return = instance.rt_ret

    O = sorted(orders_req.keys())

    # Calculate requirements
    orders_demanding_sku = defaultdict(set)
    need_count = defaultdict(int)
    for o in O:
        for k in orders_req[o]:
            orders_demanding_sku[k].add(o)
            need_count[k] += 1
    active_K = [k for k in K if need_count[k] > 0]
    U = {k: need_count[k] for k in K}

    # In CP-SAT, variables require hard bounds on creation. Calculate horizon first.
    if horizon == 0:
        horizon = sum((rt[k] + p[k] + rt_return.get(k, rt[k])) * U[k] for k in active_K)
        print(f"New horizon {horizon}")

    model = cp_model.CpModel()
    all_intervals_flat = {}  # For the progress collector

    # --- intervals ---
    I_os, I_os_lane = {}, {}
    for o in O:
        for s in S:
            iv = OptInterval(model, f"I_os[{o},{s}]", horizon)
            I_os[(o, s)] = iv
            all_intervals_flat[iv.name] = iv
            for ln in L:
                iv_lane = OptInterval(model, f"I_os_lane[{o},{s},{ln}]", horizon)
                I_os_lane[(o, s, ln)] = iv_lane
                all_intervals_flat[iv_lane.name] = iv_lane

    C = {}
    for o in O:
        for k in orders_req[o]:
            for s in S:
                iv = OptInterval(model, f"C[{o},{k},{s}]", horizon, size=p[k])
                C[(o, k, s)] = iv
                all_intervals_flat[iv.name] = iv

    P, F, R, B, Block = {}, {}, {}, {}, {}
    for s in S:
        for k in active_K:
            for e in range(U[k]):
                all_picks_for_this_copy = []
                for o in O:
                    if k in orders_req[o]:
                        iv = OptInterval(
                            model, f"P[{o},{s},{k},{e}]", horizon, size=p[k]
                        )
                        P[(o, s, k, e)] = iv
                        all_intervals_flat[iv.name] = iv
                        all_picks_for_this_copy.append(iv)

                f_iv = OptInterval(model, f"F[{s},{k},{e}]", horizon, size=rt[k])
                r_iv = OptInterval(model, f"R[{s},{k},{e}]", horizon, size=rt_return[k])
                b_iv = OptInterval(model, f"B[{s},{k},{e}]", horizon)  # variable size
                block_iv = OptInterval(model, f"Block[{s},{k},{e}]", horizon)

                F[(s, k, e)], R[(s, k, e)], B[(s, k, e)], Block[(s, k, e)] = (
                    f_iv,
                    r_iv,
                    b_iv,
                    block_iv,
                )
                for iv in [f_iv, r_iv, b_iv, block_iv]:
                    all_intervals_flat[iv.name] = iv

                # Presence coupling
                model.Add(b_iv.pres == f_iv.pres)
                model.Add(r_iv.pres == b_iv.pres)
                model.Add(block_iv.pres == f_iv.pres)

                if all_picks_for_this_copy:
                    model.AddMaxEquality(
                        b_iv.pres, [iv.pres for iv in all_picks_for_this_copy]
                    )
                else:
                    model.Add(b_iv.pres == 0)

                # Temporal links
                for o in O:
                    if k in orders_req[o]:
                        p_iv = P[(o, s, k, e)]
                        # end_before_start
                        model.Add(f_iv.end <= p_iv.start).OnlyEnforceIf(
                            [f_iv.pres, p_iv.pres]
                        )
                        model.Add(p_iv.end <= r_iv.start).OnlyEnforceIf(
                            [p_iv.pres, r_iv.pres]
                        )

                # Bin presence window
                model.Add(b_iv.start == f_iv.end).OnlyEnforceIf(b_iv.pres)
                model.Add(b_iv.end == r_iv.start).OnlyEnforceIf(b_iv.pres)

                # Block span
                model.Add(block_iv.start == f_iv.start).OnlyEnforceIf(block_iv.pres)
                model.Add(block_iv.end == r_iv.end).OnlyEnforceIf(block_iv.pres)

    # --- assignment & lanes ---
    for o in O:
        model.AddExactlyOne([I_os[(o, s)].pres for s in S])

    for o in O:
        for s in S:
            parent = I_os[(o, s)]
            lane_candidates = [I_os_lane[(o, s, ln)] for ln in L]
            model.Add(parent.pres == sum(c.pres for c in lane_candidates))
            for c in lane_candidates:
                model.Add(parent.start == c.start).OnlyEnforceIf(c.pres)
                model.Add(parent.end == c.end).OnlyEnforceIf(c.pres)

    for s in S:
        for ln in L:
            lane_set = [I_os_lane[(o, s, ln)].interval for o in O]
            if len(lane_set) >= 2:
                model.AddNoOverlap(lane_set)

    # --- order completion ---
    for o in O:
        R_o = [k for k in orders_req[o]]
        for s in S:
            parent = I_os[(o, s)]
            if R_o:
                for k in R_o:
                    c = C[(o, k, s)]
                    model.Add(c.pres == parent.pres)
                    model.Add(parent.start <= c.start).OnlyEnforceIf(c.pres)
                    model.Add(parent.end >= c.end).OnlyEnforceIf(c.pres)

                # Tighten span via max/min auxiliary variables
                min_start = model.NewIntVar(0, horizon, f"min_start_{o}_{s}")
                max_end = model.NewIntVar(0, horizon, f"max_end_{o}_{s}")
                model.AddMinEquality(min_start, [C[(o, k, s)].start for k in R_o])
                model.AddMaxEquality(max_end, [C[(o, k, s)].end for k in R_o])
                model.Add(parent.start == min_start).OnlyEnforceIf(parent.pres)
                model.Add(parent.end == max_end).OnlyEnforceIf(parent.pres)
            else:
                model.Add(parent.start == parent.end)

    # --- bind consumption to pick ---
    for o in O:
        for k in orders_req[o]:
            for s in S:
                c = C[(o, k, s)]
                Uk = U[k]
                if Uk <= 0:
                    model.Add(c.pres == 0)
                else:
                    candidates = [P[(o, s, k, e)] for e in range(Uk)]
                    model.Add(c.pres == sum(p_cand.pres for p_cand in candidates))
                    for p_cand in candidates:
                        model.Add(c.start == p_cand.start).OnlyEnforceIf(p_cand.pres)
                        model.Add(c.end == p_cand.end).OnlyEnforceIf(p_cand.pres)

    # --- capacities ---
    for s in S:
        bins_here = [B[(s, k, e)].interval for k in active_K for e in range(U[k])]
        if len(bins_here) >= 2:
            model.AddNoOverlap(bins_here)

    for k in active_K:
        family = [Block[(s, k, e)] for s in S for e in range(U[k])]
        if len(family) <= 1:
            continue
        if N[k] <= 1:
            model.AddNoOverlap([b.interval for b in family])
        elif N[k] > len(family):
            continue
        else:
            model.AddCumulative([b.interval for b in family], [1] * len(family), N[k])

    if move_cap is not None:
        moves_intervals = []
        for s in S:
            for k in active_K:
                for e in range(U[k]):
                    moves_intervals.append(F[(s, k, e)].interval)
                    moves_intervals.append(R[(s, k, e)].interval)
        if moves_intervals:
            model.AddCumulative(moves_intervals, [1] * len(moves_intervals), move_cap)

    # --- symmetry breaking ---
    if add_symmetry_breaking:
        for s in S:
            for i in range(len(L) - 1):
                sum_i = sum(I_os_lane[(o, s, L[i])].pres for o in O)
                sum_next = sum(I_os_lane[(o, s, L[i + 1])].pres for o in O)
                model.Add(sum_i >= sum_next)

        for s in S:
            for k in active_K:
                Uk = U[k]
                for e in range(Uk - 1):
                    model.AddImplication(B[(s, k, e + 1)].pres, B[(s, k, e)].pres)
                    model.Add(B[(s, k, e)].end <= B[(s, k, e + 1)].start).OnlyEnforceIf(
                        [B[(s, k, e)].pres, B[(s, k, e + 1)].pres]
                    )

    # --- objective ---
    makespan = model.NewIntVar(0, horizon, "makespan")
    for o in O:
        for s in S:
            model.Add(makespan >= I_os[(o, s)].end).OnlyEnforceIf(I_os[(o, s)].pres)
    model.Minimize(makespan)

    handles = {
        "I_os_lane": I_os_lane,
        "I_os": I_os,
        "C": C,
        "P": P,
        "F": F,
        "R": R,
        "B": B,
        "Block": Block,
        "U": U,
        "orders_req": orders_req,
        "rt": rt,
        "rt_return": rt_return,
        "p": p,
        "S": S,
        "L": L,
        "K": K,
        "active_K": active_K,
        "orders_demanding_sku": orders_demanding_sku,
        "O": O,
        "N": N,
        "move_cap": move_cap,
        "all_intervals_flat": all_intervals_flat,
        "makespan_var": makespan,
    }
    return model, handles


# --------------------------
# Solution extraction
# --------------------------
def extract_and_print_solution(solver, handles):
    def iv_present(x):
        return solver.BooleanValue(x.pres)

    def iv_start(x):
        return solver.Value(x.start)

    def iv_end(x):
        return solver.Value(x.end)

    I_os_lane, I_os = handles["I_os_lane"], handles["I_os"]
    C, P, F, R, B = handles["C"], handles["P"], handles["F"], handles["R"], handles["B"]
    U, orders_req = handles["U"], handles["orders_req"]
    S, L, K, O = handles["S"], handles["L"], handles["K"], handles["O"]
    rt, rt_ret, p = handles["rt"], handles["rt_return"], handles["p"]

    print("\n=== Objective ===")
    makespan = 0
    for o in O:
        ends = [iv_end(I_os[(o, s)]) for s in S if iv_present(I_os[(o, s)])]
        if ends:
            makespan = max(makespan, max(ends))
    print(f"Makespan: {makespan}")

    assign = {}
    for o in O:
        s_sel = next((s for s in S if iv_present(I_os[(o, s)])), None)
        if s_sel is not None:
            ln_sel = next(ln for ln in L if iv_present(I_os_lane[(o, s_sel, ln)]))
            assign[o] = (s_sel, ln_sel)

    print("\n=== Order assignments (order -> station, lane, window) ===")
    for o in O:
        if o not in assign:
            continue
        s_sel, ln_sel = assign[o]
        st, en = iv_start(I_os[(o, s_sel)]), iv_end(I_os[(o, s_sel)])
        print(
            f"Order {o:>3} -> Station {s_sel}, Lane {ln_sel}, Window [{st}, {en}) | SKUs {orders_req[o]}"
        )

    print("\n=== Station timelines ===")
    for s in S:
        events = []
        for k in K:
            Uk = U[k]
            for e in range(Uk):
                if (s, k, e) in B and iv_present(B[(s, k, e)]):
                    for o in O:
                        if k in orders_req[o]:
                            if (o, s, k, e) in P and iv_present(P[(o, s, k, e)]):
                                events.append(
                                    (
                                        iv_start(B[(s, k, e)]),
                                        {
                                            "k": k,
                                            "e": e,
                                            "o": o,
                                            "F": (
                                                iv_start(F[(s, k, e)]),
                                                iv_end(F[(s, k, e)]),
                                            ),
                                            "B": (
                                                iv_start(B[(s, k, e)]),
                                                iv_end(B[(s, k, e)]),
                                            ),
                                            "P": (
                                                iv_start(P[(o, s, k, e)]),
                                                iv_end(P[(o, s, k, e)]),
                                            ),
                                            "R": (
                                                iv_start(R[(s, k, e)]),
                                                iv_end(R[(s, k, e)]),
                                            ),
                                        },
                                    )
                                )

        bins_data = defaultdict(list)
        for bs, ev in events:
            bins_data[(ev["k"], ev["e"])].append(ev)

        sorted_bins = sorted(bins_data.items(), key=lambda item: item[1][0]["B"][0])

        print(f"\nStation {s}:")
        if not sorted_bins:
            print("  (No bins present)")

        for (k, e), evs in sorted_bins:
            ev1 = evs[0]
            fs, fe = ev1["F"]
            bs, be = ev1["B"]
            rs, re = ev1["R"]

            pick_events = [f"P(o={ev['o']})[{ev['P'][0]},{ev['P'][1]})" for ev in evs]
            picks_str = " ".join(sorted(pick_events))
            print(
                f"  SKU {k} e={e}: F[{fs},{fe}) B[{bs},{be}) {picks_str} R[{rs},{re}) | rt={rt.get(k)}, p={p.get(k)}, rtr={rt_ret.get(k)}"
            )


class ProgressCollector(cp_model.CpSolverSolutionCallback):
    def __init__(self, obj_var, all_intervals_flat):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.obj_var = obj_var
        self.all_intervals_flat = all_intervals_flat
        self.records = []
        self.best_obj = None

    def on_solution_callback(self):
        obj = self.Value(self.obj_var)
        bound = self.BestObjectiveBound()
        wall_time = self.WallTime()
        gap = abs(obj - bound) / abs(obj) if obj != 0 else 0.0

        sol_dict = {}
        for name, iv in self.all_intervals_flat.items():
            pres = self.BooleanValue(iv.pres)
            sol_dict[name] = {
                "present": pres,
                "start": self.Value(iv.start) if pres else None,
                "end": self.Value(iv.end) if pres else None,
            }

        self.best_obj = obj
        self.records.append(
            {
                "time": wall_time,
                "best": obj,
                "bound": bound,
                "gap": gap,
                "sol": sol_dict,
            }
        )


def _compute_lane_remap(solution, S, L):
    lane_remap = {}
    for s in S:
        lane_counts = defaultdict(int)
        for o, (s_a, ln_a, _, _) in solution.order_assignments.items():
            if s_a == s:
                lane_counts[ln_a] += 1
        sorted_lanes = sorted(L, key=lambda ln: (-lane_counts.get(ln, 0), ln))
        for new_idx, orig_ln in enumerate(sorted_lanes):
            lane_remap[(s, orig_ln)] = L[new_idx]
    return lane_remap


def inject_warmstart(solution, pick_events: dict, model, handles):
    I_os, I_os_lane = handles["I_os"], handles["I_os_lane"]
    F, B, R, P, C, Block = (
        handles["F"],
        handles["B"],
        handles["R"],
        handles["P"],
        handles["C"],
        handles["Block"],
    )
    S, L, K, O, U = handles["S"], handles["L"], handles["K"], handles["O"], handles["U"]

    lane_remap = _compute_lane_remap(solution, S, L)

    for o in O:
        assignment = solution.order_assignments.get(o)
        s_target, ln_target, t_start, t_end = None, None, 0, 0
        is_assigned = False

        if assignment:
            s_target, ln_orig, t_start, t_end = assignment
            ln_target = lane_remap.get((s_target, ln_orig), ln_orig)
            is_assigned = True

        for s in S:
            iv = I_os.get((o, s))
            if iv:
                if is_assigned and s == s_target:
                    model.AddHint(iv.pres, 1)
                    model.AddHint(iv.start, t_start)
                    model.AddHint(iv.end, t_end)
                else:
                    model.AddHint(iv.pres, 0)

            for ln in L:
                iv_lane = I_os_lane.get((o, s, ln))
                if iv_lane:
                    if is_assigned and s == s_target and ln == ln_target:
                        model.AddHint(iv_lane.pres, 1)
                        model.AddHint(iv_lane.start, t_start)
                        model.AddHint(iv_lane.end, t_end)
                    else:
                        model.AddHint(iv_lane.pres, 0)

    be_to_e = {}
    set_vars = set()

    for s in S:
        evts = solution.bin_events.get(s, [])
        by_sku = defaultdict(list)
        for be in evts:
            by_sku[be.sku].append(be)

        for k_sku, k_evts in by_sku.items():
            k_evts.sort(key=lambda x: x.fetch_start)
            limit = U.get(k_sku, 0)
            for i, be in enumerate(k_evts):
                if i < limit:
                    be_to_e[id(be)] = i

    for s, evts in solution.bin_events.items():
        for be in evts:
            if id(be) not in be_to_e:
                continue
            e = be_to_e[id(be)]
            k = be.sku

            if (s, k, e) in F:
                model.AddHint(F[(s, k, e)].pres, 1)
                model.AddHint(F[(s, k, e)].start, be.fetch_start)
                model.AddHint(F[(s, k, e)].end, be.presence_start)
                set_vars.add(F[(s, k, e)].name)

            if (s, k, e) in B:
                model.AddHint(B[(s, k, e)].pres, 1)
                model.AddHint(B[(s, k, e)].start, be.presence_start)
                model.AddHint(B[(s, k, e)].end, be.presence_end)
                set_vars.add(B[(s, k, e)].name)

            if (s, k, e) in R:
                model.AddHint(R[(s, k, e)].pres, 1)
                model.AddHint(R[(s, k, e)].start, be.presence_end)
                model.AddHint(R[(s, k, e)].end, be.return_end)
                set_vars.add(R[(s, k, e)].name)

            if (s, k, e) in Block:
                model.AddHint(Block[(s, k, e)].pres, 1)
                model.AddHint(Block[(s, k, e)].start, be.fetch_start)
                model.AddHint(Block[(s, k, e)].end, be.return_end)
                set_vars.add(Block[(s, k, e)].name)

    for (o, s_sel, k), (ps, pe) in pick_events.items():
        found_be = None
        candidates = [be for be in solution.bin_events.get(s_sel, []) if be.sku == k]
        for be in candidates:
            if be.presence_start <= ps and pe <= be.presence_end:
                found_be = be
                break

        if found_be and id(found_be) in be_to_e:
            e = be_to_e[id(found_be)]
            if (o, s_sel, k, e) in P:
                model.AddHint(P[(o, s_sel, k, e)].pres, 1)
                model.AddHint(P[(o, s_sel, k, e)].start, ps)
                model.AddHint(P[(o, s_sel, k, e)].end, pe)
                set_vars.add(P[(o, s_sel, k, e)].name)

            if (o, k, s_sel) in C:
                model.AddHint(C[(o, k, s_sel)].pres, 1)
                model.AddHint(C[(o, k, s_sel)].start, ps)
                model.AddHint(C[(o, k, s_sel)].end, pe)
                set_vars.add(C[(o, k, s_sel)].name)

    for s in S:
        for k in U:
            for e in range(U[k]):
                if (s, k, e) in F and F[(s, k, e)].name not in set_vars:
                    model.AddHint(F[(s, k, e)].pres, 0)
                if (s, k, e) in B and B[(s, k, e)].name not in set_vars:
                    model.AddHint(B[(s, k, e)].pres, 0)
                if (s, k, e) in R and R[(s, k, e)].name not in set_vars:
                    model.AddHint(R[(s, k, e)].pres, 0)
                if (s, k, e) in Block and Block[(s, k, e)].name not in set_vars:
                    model.AddHint(Block[(s, k, e)].pres, 0)
                for o in O:
                    if (o, s, k, e) in P and P[(o, s, k, e)].name not in set_vars:
                        model.AddHint(P[(o, s, k, e)].pres, 0)

    for o in O:
        for k in K:
            for s in S:
                if (o, k, s) in C and C[(o, k, s)].name not in set_vars:
                    model.AddHint(C[(o, k, s)].pres, 0)


def main():
    ap = argparse.ArgumentParser(description="CP Autostoremodel CP-SAT.")
    ap.add_argument(
        "--stations", type=int, default=1, help="Number of picking stations |S|"
    )
    ap.add_argument("--lanes", type=int, default=4, help="Lanes per station")
    ap.add_argument("--orders", type=int, default=10, help="Number of orders |O|")
    ap.add_argument("--skus", type=int, default=10, help="Number of SKUs |K|")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--pick", type=int, default=4, help="Constant pick touch time")
    ap.add_argument(
        "--movecap", type=int, default=8, help="Max simultaneous moves (F or R)"
    )
    ap.add_argument(
        "--timelimit", type=int, default=5, help="Solver time limit (seconds)"
    )
    ap.add_argument(
        "--no_symmetry_breaking", action="store_true", help="Disable symmetry breaking"
    )
    ap.add_argument(
        "--horizon", type=int, default=10000, help="Maximal horizon (0 for unbounded)"
    )
    ap.add_argument("--newdatagen", action="store_true", help="Use new data generation")

    args = ap.parse_args()
    add_symmetry_breaking = not args.no_symmetry_breaking

    print("Generating data...")

    instance = generate_data(
        num_stations=args.stations,
        lanes_per_station=args.lanes,
        num_orders=args.orders,
        num_skus=args.skus,
        seed=args.seed,
        pick_touch_time=args.pick,
        movecap=args.movecap,
    )
    S, L, K, orders_req, rt, p, N = instance
    print(f"Stations S={S} | Lanes={len(L)} | Orders={len(orders_req)}")

    model, handles = build_model(
        instance,
        rt_return=instance.rt_ret,
        add_symmetry_breaking=add_symmetry_breaking,
        horizon=args.horizon,
        move_cap=args.movecap,
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = args.timelimit
    solver.parameters.log_search_progress = True

    # Progress callback mapping
    collector = ProgressCollector(
        handles["makespan_var"], handles["all_intervals_flat"]
    )

    print("\nSolving...")
    status = solver.Solve(model, collector)

    print(f"Solve status: {solver.StatusName(status)}")
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        extract_and_print_solution(solver, handles)
    else:
        print("No solution found.")


if __name__ == "__main__":
    main()
