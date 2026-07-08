#!/usr/bin/env python3
import argparse
from collections import defaultdict
import random
import math

from docplex.cp.model import CpoModel
from docplex.cp.solver.solver_listener import CpoSolverListener

from datagen import generate_data, generate_data_legacy

try:
    from schedule_visualizer import plot_schedule, write_html
except ImportError:
    print("Warning: schedule_visualizer2 not found. Visualization will be disabled.")
    plot_schedule = None
    write_html = None


def _old_generate_data(
        num_stations: int,
        lanes_per_station: int,
        num_orders: int,
        num_skus: int,
        seed: int,
        pick_touch_time: int = 4,
        order_size_dist: str = "poisson2_to_1_6",
        max_bins_per_sku: int = 5
):
    """
    Generates synthetic data.
    Returns:
        S, L, K, orders_requirements, rt, p
        https://gemini.google.com/share/a2eae4df20e5
        https://docs.google.com/document/d/1gFIFRr_p6QVPusrN5v0qeaQQ3KgbDgK8p_9tlmU6L3k/edit?usp=sharing
    """
    print("Generating synthetic data using older generator...")
    rng = random.Random(seed)

    S = list(range(num_stations))
    L = list(range(lanes_per_station))
    K = list(range(num_skus))

    # Retrieval times per SKU: triangular(low, high, mode)
    rt = {k: int(rng.triangular(5, 60, 25)) for k in K}
    rt = {k: max(1, t) for k, t in rt.items()}

    # Constant pick-touch time per SKU:
    p = {k: int(pick_touch_time) for k in K}

    # Generate physical bin counts per SKU
    N = {k: rng.randint(1, max_bins_per_sku) for k in K}

    # Order sizes (how many different SKUs per order)
    def sample_order_size():
        if order_size_dist == "poisson2_to_1_6":
            lam = 2.0
            Lp = math.exp(-lam)
            kk = 0
            prod = 1.0
            while prod > Lp:
                kk += 1
                prod *= rng.random()
            kk -= 1
            return min(6, max(1, kk))
        elif order_size_dist == "uniform_1_5":
            return rng.randint(1, 5)
        else:
            return rng.randint(1, 4)

    # Orders: choose distinct SKUs per order
    orders_requirements = {}
    for o in range(num_orders):
        size = sample_order_size()
        req = set(rng.sample(K, k=min(size, len(K)))) if K else set()
        if not req and len(K) > 0:
            req = {rng.choice(K)}
        orders_requirements[o] = sorted(req)

    return S, L, K, orders_requirements, rt, p, N

# --------------------------
# Model building (v3)
# --------------------------


def build_model(S, L, K, orders_req, rt, p, rt_return=None, add_symmetry_breaking=True, horizon=0, move_cap=None,
                N=None):
    """
    Intervals per (s,k,e):
      - F[s,k,e] : fetch (size = rt[k])
      - P[s,k,e] : pick  (size = p[k])
      - R[s,k,e] : return(size = rt_return[k])
      - B[s,k,e] : bin presence at station (free size) with start(B)=end(F), end(B)=start(R)

    Station capacity: no_overlap over B[s,*,*] (exactly one bin at station).
    Global single-bin per SKU: no_overlap over {F,B,R} across all stations.
    """
    mdl = CpoModel()
    O = sorted(orders_req.keys())
    if rt_return is None:
        rt_return = rt  # symmetric round-trip by default

    # --- demand and candidate copy counts ---
    need_count = defaultdict(int)
    for o in O:
        for k in orders_req[o]:
            need_count[k] += 1
    active_K = [k for k in K if need_count[k] > 0]
    Lcap = max(1, len(L))

    # Conservative U: one bin visit per order item requirement.
    # This prevents heuristic warmstart from failing if it doesn't cluster perfectly.
    # The CP model will prune unused F/B/R/P variables quickly anyway.
    U = {k: need_count[k] for k in K}

    # --- intervals ---
    # Station-level order window + lane window (keep I_os, I_os_lane)
    I_os, I_os_lane = {}, {}
    for o in O:
        for s in S:
            I_os[(o, s)] = mdl.interval_var(optional=True, name=f"I_os[{o},{s}]")
            for ln in L:
                I_os_lane[(o, s, ln)] = mdl.interval_var(optional=True, name=f"I_os_lane[{o},{s},{ln}]")

    # Consumptions: one per (order, required SKU, station)
    C = {}
    for o in O:
        for k in orders_req[o]:
            for s in S:
                C[(o, k, s)] = mdl.interval_var(size=p[k], optional=True, name=f"C[{o},{k},{s}]")

    # Fetch / Pick / Return / Bin-Presence, only for active SKUs
    P, F, R, B, Block = {}, {}, {}, {}, {}
    for s in S:
        for k in active_K:
            for e in range(U[k]):
                all_picks_for_this_copy = []
                for o in O:
                    if k in orders_req[o]:
                        P[(o, s, k, e)] = mdl.interval_var(size=p[k], optional=True, name=f"P[{o},{s},{k},{e}]")
                        all_picks_for_this_copy.append(P[(o, s, k, e)])
                F[(s, k, e)] = mdl.interval_var(size=rt[k], optional=True, name=f"F[{s},{k},{e}]")
                R[(s, k, e)] = mdl.interval_var(size=rt_return[k], optional=True, name=f"R[{s},{k},{e}]")
                B[(s, k, e)] = mdl.interval_var(optional=True, name=f"B[{s},{k},{e}]")  # free size
                # Presence coupling
                mdl.add(mdl.presence_of(B[(s, k, e)]) == mdl.presence_of(F[(s, k, e)]))
                mdl.add(mdl.presence_of(R[(s, k, e)]) == mdl.presence_of(B[(s, k, e)]))

                # The bin arrival (B) is present IFF at least one order's pick (P) uses it.
                # This links the presence of P[(o,s,k,e)] to B[(s,k,e)].
                if all_picks_for_this_copy:
                    # any_P_present is a 0/1 expression that is 1 if any P is present
                    any_P_present = mdl.max(mdl.presence_of(iv) for iv in all_picks_for_this_copy)
                    mdl.add(mdl.presence_of(B[(s, k, e)]) == any_P_present)
                else:
                    # No orders exist, so this B should never be present
                    mdl.add(mdl.presence_of(B[(s, k, e)]) == 0)

                # Temporal links: F -> P -> R
                for o in O:
                    if k in orders_req[o]:
                        mdl.add(mdl.end_before_start(F[(s, k, e)], P[(o, s, k, e)]))
                        mdl.add(mdl.end_before_start(P[(o, s, k, e)], R[(s, k, e)]))  # start(R) >= end(P)
                    # mdl.add(mdl.end_before_end(P[(o, s, k, e)], B[(s, k, e)])) does not help
                    # mdl.add(mdl.start_before_start(B[(s, k, e)], P[(o, s, k, e)]))

                # Bin presence window: [end(F), start(R)]
                mdl.add(mdl.start_at_end(B[(s, k, e)], F[(s, k, e)]))
                mdl.add(mdl.end_at_start(B[(s, k, e)], R[(s, k, e)]))

                # This Block spans from the start of Fetch to the end of Return
                Block[(s, k, e)] = mdl.interval_var(optional=True, name=f"Block[{s},{k},{e}]")
                mdl.add(mdl.span(Block[(s, k, e)], [F[(s, k, e)], R[(s, k, e)]]))

    if horizon == 0:
        horizon = sum((rt[k] + p[k] + rt_return.get(k, rt[k])) * U[k] for k in active_K)
        print(f"New horizon {horizon}")
    if move_cap is not None:
        moves = 0
        for s in S:
            for k in active_K:
                for e in range(U[k]):
                    moves += mdl.pulse(F[(s, k, e)], 1)
                    moves += mdl.pulse(R[(s, k, e)], 1)

        mdl.add(mdl.always_in(moves, (0, horizon), 0, move_cap))

    # --- assignment & lanes ---
    # (i) an order chooses exactly ONE station (via I_os presence)
    for o in O:
        mdl.add(mdl.sum(mdl.presence_of(I_os[(o, s)]) for s in S) == 1)

    # (ii) at chosen station, I_os equals exactly one lane window
    for o in O:
        for s in S:
            mdl.add(mdl.alternative(I_os[(o, s)], [I_os_lane[(o, s, ln)] for ln in L]))

    # (iii) lanes are unary (capacity L per station)
    for s in S:
        for ln in L:
            lane_set = [I_os_lane[(o, s, ln)] for o in O]
            if len(lane_set) >= 2:
                mdl.add(mdl.no_overlap(lane_set))

    # --- order completion = all consumptions at chosen station ---
    for o in O:
        R_o = [k for k in orders_req[o]]
        for s in S:
            if R_o:
                mdl.add(mdl.span(I_os[(o, s)], [C[(o, k, s)] for k in R_o]))
            else:
                mdl.add(mdl.length_of(I_os[(o, s)]) == 0)
            for k in R_o:
                mdl.add(mdl.presence_of(C[(o, k, s)]) == mdl.presence_of(I_os[(o, s)]))

    # --- bind each consumption to one pick at same station ---
    for o in O:
        for k in orders_req[o]:
            for s in S:
                Uk = U[k]
                if Uk <= 0:
                    # no demand => no picks exist; but we only create C for required k, so Uk>0 here
                    mdl.add(mdl.presence_of(C[(o, k, s)]) == 0)
                else:
                    candidates = [P[(o, s, k, e)] for e in range(Uk)]
                    mdl.add(mdl.alternative(C[(o, k, s)], candidates))

    # --- capacities (disjunctive only) ---
    # (1) Exactly one bin present at any station s at any time
    for s in S:
        bins_here = [B[(s, k, e)] for k in active_K for e in range(U[k])]
        if len(bins_here) >= 2:
            mdl.add(mdl.no_overlap(bins_here))

    # (2) Physical-bin capacity per SKU globally (<= Q[k] concurrent Blocks)
    #     - If Q[k] == 1: keep strong no_overlap propagation (v4 behavior)
    #     - Else: cumulative cap via step function pulses over Block intervals
    for k in active_K:
        family = [Block[(s, k, e)] for s in S for e in range(U[k])]
        if len(family) <= 1:
            continue

        if N[k] <= 1:
            if len(family) >= 2:
                mdl.add(mdl.no_overlap(family))
        if N[k] > len(family):
            # 1) Can't overlap more than the number of intervals you created
            continue
        if move_cap is not None and N[k] >= (len(S) + int(move_cap)):
            # 2) If move_cap exists, then at most move_cap bins can be moving (F/R) globally at any time,
            #    plus at most |S| bins can be at stations (B stage). So overlap for any single SKU
            #    can't exceed |S| + move_cap.
            continue

        bin_usage = 0
        for s in S:
            for e in range(U[k]):
                # pulse(interval, amount) adds 1 to the function during the Block
                bin_usage += mdl.pulse(Block[(s, k, e)], 1)

                # Constrain maximum concurrent usage to the available bins N[k]
        mdl.add(mdl.always_in(bin_usage, (0, horizon), 0, N[k]))

    # --- symmetry breaking ---
    if add_symmetry_breaking:
        # print("Adding symmetry breaking constraints...") # Quieter for benchmark
        # (A) Lane fill order: usage(L0) >= usage(L1) >= ... per station
        for s in S:
            for i in range(len(L) - 1):
                mdl.add(
                    mdl.sum(mdl.presence_of(I_os_lane[(o, s, i)]) for o in O) >=
                    mdl.sum(mdl.presence_of(I_os_lane[(o, s, i + 1)]) for o in O)
                )

        # (B) Ordered pick copies: for each (s,k), present copies form a prefix and are chained
        for s in S:
            for k in active_K:
                Uk = U[k]
                for e in range(Uk - 1):
                    # if P_{e+1} is present => P_e must be present  (prefix)
                    # mdl.add(mdl.if_then(mdl.presence_of(P[(s, k, e + 1)]) == 1,
                    #                     mdl.presence_of(P[(s, k, e)]) == 1))
                    mdl.add(mdl.if_then(mdl.presence_of(B[(s, k, e + 1)]) == 1,
                                        mdl.presence_of(B[(s, k, e)]) == 1))
                    # and order them in time
                    # mdl.add(mdl.end_before_start(P[(s, k, e)], P[(s, k, e + 1)]))
                    mdl.add(mdl.end_before_start(B[(s, k, e)], B[(s, k, e + 1)]))

        # (C) Orders assigned in order to stations
        # Station load: number of assigned orders at station s
        # load = {}
        # for s in S:
        #     load[s] = mdl.sum(mdl.presence_of(I_os[o, s]) for o in O)
        #
        # # Symmetry breaking: non-increasing loads by station index
        # for i in range(len(S) - 1):
        #     mdl.add(load[S[i]] >= load[S[i+1]])

    # --- maximal horizon ---
    if horizon > 0:
        # print(f"Adding maximal horizon constraint: end <= {horizon}") # Quieter
        for o in O:
            for s in S:
                # This constrains the end time *if* the interval is present
                mdl.add(mdl.end_of(I_os[(o, s)]) <= horizon)

    # --- objective (makespan over station windows) ---
    per_order_end = [mdl.max([mdl.end_of(I_os[(o, s)]) for s in S]) for o in O]
    mdl.minimize(mdl.max(per_order_end))

    handles = {
        "I_os_lane": I_os_lane,
        "I_os": I_os,
        "C": C, "P": P, "F": F, "R": R, "B": B, "Block": Block,
        "U": U, "orders_req": orders_req,
        "rt": rt, "rt_return": rt_return, "p": p,
        "S": S, "L": L, "K": K, "O": O,
        "N": N,
        "move_cap": move_cap,
    }
    return mdl, handles


# --------------------------
# Solution extraction
# --------------------------

def extract_and_print_solution(sol, handles):
    # CP 22.1 helpers
    def iv_present(x):
        vs = sol.get_var_solution(x)
        return (vs is not None) and vs.is_present()

    def iv_start(x):
        return sol.get_var_solution(x).get_start()

    def iv_end(x):
        return sol.get_var_solution(x).get_end()

    I_os_lane = handles["I_os_lane"]
    I_os = handles["I_os"]
    C = handles["C"]
    P = handles["P"]
    F = handles["F"]
    R = handles["R"]
    B = handles["B"]
    U = handles["U"]
    orders_req = handles["orders_req"]
    S, L, K, O = handles["S"], handles["L"], handles["K"], handles["O"]
    rt, rt_ret, p = handles["rt"], handles["rt_return"], handles["p"]

    if sol is None:
        print("No solution found.")
        return

    print("\n=== Objective ===")
    makespan = 0
    for o in O:
        ends = [iv_end(I_os[(o, s)]) for s in S if iv_present(I_os[(o, s)])]
        if ends:
            makespan = max(makespan, max(ends))
    print(f"Makespan: {makespan}")

    # Assignments
    assign = {}
    for o in O:
        s_sel = next(s for s in S if iv_present(I_os[(o, s)]))
        ln_sel = next(ln for ln in L if iv_present(I_os_lane[(o, s_sel, ln)]))
        assign[o] = (s_sel, ln_sel)

    print("\n=== Order assignments (order -> station, lane, window) ===")
    for o in O:
        s_sel, ln_sel = assign[o]
        st, en = iv_start(I_os[(o, s_sel)]), iv_end(I_os[(o, s_sel)])
        print(f"Order {o:>3} -> Station {s_sel}, Lane {ln_sel}, Window [{st}, {en}) | SKUs {orders_req[o]}")

    # Per-station sequences — show F, B, P, R chronologically (by start of B)
    print("\n=== Station timelines ===")
    for s in S:
        events = []
        for k in K:
            Uk = U[k]
            for e in range(Uk):
                if (s, k, e) in B and iv_present(B[(s, k, e)]):
                    for o in O:
                        if k in orders_req[o]:
                            # Find the matching P
                            if (o, s, k, e) in P and iv_present(P[(o, s, k, e)]):
                                ps, pe = iv_start(P[(o, s, k, e)]), iv_end(P[(o, s, k, e)])
                                bs, be = iv_start(B[(s, k, e)]), iv_end(B[(s, k, e)])
                                fs, fe = iv_start(F[(s, k, e)]), iv_end(F[(s, k, e)])
                                rs, re = iv_start(R[(s, k, e)]), iv_end(R[(s, k, e)])
                                events.append((bs, {
                                    "k": k, "e": e, "o": o,
                                    "F": (fs, fe), "B": (bs, be), "P": (ps, pe), "R": (rs, re)
                                }))

        # Group events by bin (k, e)
        bins_data = defaultdict(list)
        for bs, ev in events:
            bins_data[(ev['k'], ev['e'])].append(ev)

        # Sort bins by their start time
        sorted_bins = sorted(bins_data.items(), key=lambda item: item[1][0]['B'][0])

        print(f"\nStation {s}:")
        if not sorted_bins:
            print("  (No bins present)")

        for (k, e), evs in sorted_bins:
            # All events in 'evs' share the same F, B, R. Take from first.
            ev1 = evs[0]
            fs, fe = ev1["F"]
            bs, be = ev1["B"]
            rs, re = ev1["R"]

            pick_events = []
            for ev in evs:
                ps, pe = ev["P"]
                pick_events.append(f"P(o={ev['o']})[{ps},{pe})")

            picks_str = " ".join(sorted(pick_events))
            print(f"  SKU {k} e={e}: F[{fs},{fe}) B[{bs},{be}) {picks_str} R[{rs},{re})"
                  f" | rt={rt.get(k)}, p={p.get(k)}, rtr={rt_ret.get(k)}")

    # Coverage
    print("\n=== Order-SKU coverage via pick events ===")
    for o in O:
        (s_sel, ln_sel) = assign[o]
        for k in orders_req[o]:
            Civ = C[(o, k, s_sel)]
            if iv_present(Civ):
                st, en = iv_start(Civ), iv_end(Civ)
                chosen = None
                Uk = U[k]
                for e in range(Uk):
                    if (o, s_sel, k, e) in P:
                        Piv = P[(o, s_sel, k, e)]
                        if iv_present(Piv) and iv_start(Piv) == st and iv_end(Piv) == en:
                            chosen = e
                            break
                print(f"Order {o:>3} needs SKU {k:>3} -> uses P[{o},{s_sel},{k},{chosen}] at [{st},{en})")
            else:
                print(f"Order {o:>3} needs SKU {k:>3} -> MISSING (should not happen)")


def solve_instance(config):
    """
    Runs a single instance of the model based on the config dict.
    This function is designed to be imported by a benchmark script.

    Returns a dict with results:
        {
            'status': str,
            'solve_time': float,
            'objective_value': float | None,
            'num_vars': int,
            'progress': [
                {'time': float, 'best': float, 'bound': float, 'gap': float},
                ...
            ]
        }
    """
    # --- 1. Get parameters from config ---
    num_stations = config.get('stations', 2)
    lanes_per_station = config.get('lanes', 2)
    num_orders = config.get('orders', 10)
    num_skus = config.get('skus', 90)
    seed = config.get('seed', 42)
    pick_touch_time = config.get('pick', 4)
    timelimit = config.get('timelimit', 25)
    add_symmetry_breaking = config.get('symmetry_breaking', True)
    horizon = config.get('horizon', 10000)
    move_cap = config.get('movecap', 20)

    # "Quiet" -> no logs; anything else turns logs on
    verbose = config.get('verbose', "Quiet")
    collect_progress = config.get('collect_progress', True)
    is_verbose = verbose not in (None, False, "Quiet")

    if is_verbose:
        print("--- Running instance ---")
        print(f"Config: {config}")

    # --- 2. Generate Data ---
    try:
        S, L, K, orders_req, rt, p, N = _old_generate_data(
            num_stations=num_stations,
            lanes_per_station=lanes_per_station,
            num_orders=num_orders,
            num_skus=num_skus,
            seed=seed,
            pick_touch_time=pick_touch_time,
        )
        rt_return = dict(rt)
        if is_verbose:
            print("=== Generated data ===")
            print(f"Stations S={S}  |L|={len(L)} lanes each")
            # print(f"SKUs K={K}")
            # print(f"Bins of SKUs={N}")
            # print(f"Retrieval times rt_k: {rt}")
            # print(f"Return times rtr_k:   {rt_return}")
            # print(f"Pick touch time p_k:  {p[next(iter(p))]} (uniform)")
            print("Orders (o -> required SKUs R_o):")
            for o in sorted(orders_req.keys()):
                print(f"  o={o:>2}: {orders_req[o]}")
    except Exception as e:
        if is_verbose:
            print(f"Data generation failed: {e}")
        return {
            'status': 'DataGenError',
            'solve_time': 0.0,
            'objective_value': None,
            'num_vars': 0,
            'progress': [],
        }

    # --- 3. Build Model ---
    num_vars = 0
    try:
        if is_verbose and add_symmetry_breaking:
            print("Adding symmetry breaking constraints...")
        if is_verbose and horizon > 0:
            print(f"Adding maximal horizon constraint: end <= {horizon}")
        if is_verbose and move_cap is not None:
            print(f"Using move capacity: {move_cap}")

        mdl, handles = build_model(
            S, L, K, orders_req, rt, p,
            rt_return=rt_return,
            add_symmetry_breaking=add_symmetry_breaking,
            horizon=horizon,
            move_cap=move_cap,
            N=N
        )
        num_vars = len(mdl.get_all_variables())
        if is_verbose:
            print(f"Model built. Vars: {num_vars}")
    except Exception as e:
        print(f"Model build failed: {e}")
        return {
            'status': 'ModelBuildError',
            'solve_time': 0.0,
            'objective_value': None,
            'num_vars': num_vars,
            'progress': [],
        }

    # --- 4. Attach progress listener (optional) ---
    collector = None
    if collect_progress:
        collector = ProgressCollector()
        mdl.add_solver_listener(collector)

    # --- 5. Solve ---
    try:
        solve_kwargs = dict(
            Workers=1,
            TimeLimit=timelimit,
            LogVerbosity="Terse",  # keep benchmark runs quiet
        )
        if collect_progress:
            # Ask CP Optimizer to keep iterating with search_next()
            # so that the listener sees all incumbent improvements.
            solve_kwargs['solve_with_search_next'] = True

        sol = mdl.solve(**solve_kwargs)

        # --- 6. Extract Results ---
        status = sol.get_solve_status()
        solve_time = sol.get_solve_time()
        obj_val = None

        if status in ("Optimal", "Feasible"):
            obj_val = sol.get_objective_values()[0]
            if is_verbose:
                print(f"Solution found: {status}, Obj={obj_val}, Time={solve_time}")
        elif status == "Infeasible":
            if is_verbose:
                print(f"Problem Infeasible. Time={solve_time}")
        elif status == "Unknown":
            # CP Optimizer often returns "Unknown" on TimeLimit
            status = "TimeLimit"
            if is_verbose:
                print(f"Solver stopped with Unknown (likely timelimit). Time={solve_time}")
        else:
            if is_verbose:
                print(f"Solve failed. Status: {status}, Time={solve_time}")

        return {
            'status': status,
            'solve_time': solve_time,
            'objective_value': obj_val,
            'num_vars': num_vars,
            'progress': collector.records if collector is not None else [],
        }

    except Exception as e:
        if is_verbose:
            print(f"Solver crashed: {e}")

        # Try to recover solve_time if possible
        solve_time = 0.0
        try:
            if 'sol' in locals() and sol is not None:
                solve_time = sol.get_solve_time()
        except Exception:
            pass

        if timelimit > 0 and solve_time >= timelimit:
            status = "TimeLimit"
            solve_time = float(timelimit)
        else:
            status = "Crash"

        return {
            'status': status,
            'solve_time': solve_time,
            'objective_value': None,
            'num_vars': num_vars,
            'progress': collector.records if collector is not None else [],
        }


def _compute_lane_remap(solution, S, L):
    """Compute lane permutation per station to satisfy symmetry breaking constraint (A).

    The CP model requires count(orders at lane i) >= count(orders at lane i+1).
    The heuristic assigns orders to the earliest-free lane, which doesn't guarantee
    this ordering.  We fix it by relabelling: most-used lane -> index 0, etc.

    Returns dict: (s, original_lane) -> remapped_lane_index.
    """
    lane_remap = {}
    for s in S:
        lane_counts = defaultdict(int)
        for o, (s_a, ln_a, _, _) in solution.order_assignments.items():
            if s_a == s:
                lane_counts[ln_a] += 1
        # Sort by count descending, then lane index ascending for stability
        sorted_lanes = sorted(L, key=lambda ln: (-lane_counts.get(ln, 0), ln))
        for new_idx, orig_ln in enumerate(sorted_lanes):
            lane_remap[(s, orig_ln)] = L[new_idx]
    return lane_remap


def inject_warmstart(solution, pick_events: dict, mdl, handles):
    """Build a CpoStartingPoint from a heuristic Solution.

    Parameters
    ----------
    solution:
        ``autostore_heuristic.Solution`` returned by ``run_sgc``.
    pick_events:
        ``solution.pick_events`` — mapping ``(o, s, k) -> (pick_start, pick_end)``.
    mdl:
        The ``CpoModel`` returned by ``build_model`` for the *same* problem instance.
    handles:
        The handles dict returned alongside *mdl* by ``build_model``.

    Returns a ``CpoStartingPoint`` ready to pass as
    ``mdl.solve(starting_point=sp, ...)``.
    """

    sp = mdl.create_empty_solution()

    I_os = handles["I_os"]
    I_os_lane = handles["I_os_lane"]
    F, B, R, P, C = handles["F"], handles["B"], handles["R"], handles["P"], handles["C"]
    Block = handles["Block"]
    S, L, K, O = handles["S"], handles["L"], handles["K"], handles["O"]
    U = handles["U"]

    # --- 0. Remap lanes to satisfy symmetry breaking constraint (A) ---
    lane_remap = _compute_lane_remap(solution, S, L)

    # --- 1. Order windows and assignments ---
    for o in O:
        assignment = solution.order_assignments.get(o)

        # Determine actual assignment from heuristic
        # assignment = (station, lane, start, end)
        s_target, ln_target = None, None
        t_start, t_end = 0, 0
        is_assigned = False

        if assignment:
            s_target, ln_orig, t_start, t_end = assignment
            ln_target = lane_remap.get((s_target, ln_orig), ln_orig)
            is_assigned = True

        # I_os[(o, s)]
        for s in S:
            iv = I_os.get((o, s))
            if iv is None:
                continue

            if is_assigned and s == s_target:
                sp.add_interval_var_solution(iv, presence=True, start=t_start, end=t_end)
            else:
                sp.add_interval_var_solution(iv, presence=False)

        # I_os_lane[(o, s, ln)]
        for s in S:
            for ln in L:
                iv_lane = I_os_lane.get((o, s, ln))
                if iv_lane is None:
                    continue

                if is_assigned and s == s_target and ln == ln_target:
                    sp.add_interval_var_solution(iv_lane, presence=True, start=t_start, end=t_end)
                else:
                    sp.add_interval_var_solution(iv_lane, presence=False)

    # --- 2. Bin Events Mapping (Heuristic bins -> CP event indices) ---
    # remap[bin_event_obj] -> e_index
    be_to_e = {}

    # Track which CP intervals we have set to Present
    set_vars = set()

    for s in S:
        evts = solution.bin_events.get(s, [])
        # Group by SKU
        by_sku = defaultdict(list)
        for be in evts:
            by_sku[be.sku].append(be)

        for k_sku, k_evts in by_sku.items():
            # Sort by fetch start to align with CP symmetry preference (if any)
            k_evts.sort(key=lambda x: x.fetch_start)

            # Map to 0..U[k]-1
            limit = U.get(k_sku, 0)
            for i, be in enumerate(k_evts):
                if i < limit:
                    be_to_e[id(be)] = i
                else:
                    # Heuristic used more bin trips than U[k] allows?
                    # This implies U calculation mismatch or heuristic over-segmentation.
                    pass

    # --- 3. Bin Events (F, B, R, Block) ---
    for s, evts in solution.bin_events.items():
        for be in evts:
            if id(be) not in be_to_e:
                continue
            e = be_to_e[id(be)]
            k = be.sku

            # Set F, B, R, Block as present
            if (s, k, e) in F:
                sp.add_interval_var_solution(F[(s, k, e)], presence=True, start=be.fetch_start, end=be.presence_start)
                set_vars.add(F[(s, k, e)])

            if (s, k, e) in B:
                sp.add_interval_var_solution(B[(s, k, e)], presence=True, start=be.presence_start, end=be.presence_end)
                set_vars.add(B[(s, k, e)])

            if (s, k, e) in R:
                sp.add_interval_var_solution(R[(s, k, e)], presence=True, start=be.presence_end, end=be.return_end)
                set_vars.add(R[(s, k, e)])

            if (s, k, e) in Block:
                sp.add_interval_var_solution(Block[(s, k, e)], presence=True, start=be.fetch_start, end=be.return_end)
                set_vars.add(Block[(s, k, e)])

    # --- 4. Picks and Consumption ---
    # pick_events: (o, s, k) -> (start, end)
    for (o, s_sel, k), (ps, pe) in pick_events.items():
        # Find which bin event covers this pick
        found_be = None
        # Candidates: events at s_sel for k
        candidates = [be for be in solution.bin_events.get(s_sel, []) if be.sku == k]
        for be in candidates:
            # Pick must be within presence window
            if be.presence_start <= ps and pe <= be.presence_end:
                found_be = be
                break

        if found_be and id(found_be) in be_to_e:
            e = be_to_e[id(found_be)]

            # P[(o, s, k, e)]
            if (o, s_sel, k, e) in P:
                sp.add_interval_var_solution(P[(o, s_sel, k, e)], presence=True, start=ps, end=pe)
                set_vars.add(P[(o, s_sel, k, e)])

            # C[(o, k, s)]
            if (o, k, s_sel) in C:
                sp.add_interval_var_solution(C[(o, k, s_sel)], presence=True, start=ps, end=pe)
                set_vars.add(C[(o, k, s_sel)])

    # --- 5. SWEEP: Explicitly mark absent whatever is not in set_vars ---

    # F, B, R, Block are indexed by (s, k, e)
    for s in S:
        for k in U:
            for e in range(U[k]):
                # Check F
                if (s, k, e) in F and F[(s, k, e)] not in set_vars:
                    sp.add_interval_var_solution(F[(s, k, e)], presence=False)
                # Check B
                if (s, k, e) in B and B[(s, k, e)] not in set_vars:
                    sp.add_interval_var_solution(B[(s, k, e)], presence=False)
                # Check R
                if (s, k, e) in R and R[(s, k, e)] not in set_vars:
                    sp.add_interval_var_solution(R[(s, k, e)], presence=False)
                # Check Block
                if (s, k, e) in Block and Block[(s, k, e)] not in set_vars:
                    sp.add_interval_var_solution(Block[(s, k, e)], presence=False)

                # Check P[(o,s,k,e)]
                for o in O:
                    if (o, s, k, e) in P and P[(o, s, k, e)] not in set_vars:
                        sp.add_interval_var_solution(P[(o, s, k, e)], presence=False)

    # C[(o,k,s)]
    for o in O:
        for k in K:
            # Only relevant if k in orders_req[o]
            for s in S:
                if (o, k, s) in C and C[(o, k, s)] not in set_vars:
                    sp.add_interval_var_solution(C[(o, k, s)], presence=False)

    return sp


def validate_warmstart(solution, pick_events, handles):
    """
    Checks the heuristic solution for consistency with CP model constraints.
    Returns a list of violation messages.
    """
    violations = []

    # 1. Copy count check
    U = handles["U"]
    S = handles["S"]
    for s in S:
        evts = solution.bin_events.get(s, [])
        counts = defaultdict(int)
        for be in evts:
            counts[be.sku] += 1
        for k, count in counts.items():
            if count > U.get(k, 0):
                violations.append(
                    f"Station {s} SKU {k}: Heuristic used {count} trips, CP model limit U[{k}]={U.get(k,0)}")

    # 2. Pick within Presence
    for (o, s_sel, k), (ps, pe) in pick_events.items():
        found = False
        candidates = [be for be in solution.bin_events.get(s_sel, []) if be.sku == k]
        for be in candidates:
            if be.presence_start <= ps and pe <= be.presence_end:
                found = True
                break
        if not found:
            violations.append(
                f"Pick {o}@{s_sel} SKU {k} [{ps},{pe}] not covered by any BinEvent presence interval at station.")

    # 3. Order window consistent with Picks
    # The CP model uses `span(I_os, [C])`.
    # This implies start(I_os) == min(start(C)) and end(I_os) == max(end(C)).
    for o in solution.order_assignments:
        s_sel, ln_sel, t_start, t_end = solution.order_assignments[o]

        # Collect all picks for this order
        picks = []
        for (oo, ss, kk), (ps, pe) in pick_events.items():
            if oo == o:
                picks.append((ps, pe))

        if not picks:
            continue

        min_p = min(ps for ps, pe in picks)
        max_p = max(pe for ps, pe in picks)

        if t_start != min_p:
            violations.append(f"Order {o}: Heuristic Start {t_start} != Min Pick Start {min_p}")
        if t_end != max_p:
            violations.append(f"Order {o}: Heuristic End {t_end} != Max Pick End {max_p}")

    # 4. Lane fill order symmetry breaking (diagnostic on raw heuristic lanes)
    S = handles["S"]
    L = handles["L"]
    for s in S:
        lane_counts = defaultdict(int)
        for o, (s_a, ln_a, _, _) in solution.order_assignments.items():
            if s_a == s:
                lane_counts[ln_a] += 1
        for i in range(len(L) - 1):
            count_i = lane_counts.get(L[i], 0)
            count_next = lane_counts.get(L[i + 1], 0)
            if count_i < count_next:
                violations.append(
                    f"Symmetry (A) raw lanes at station {s}: "
                    f"lane {L[i]} has {count_i} orders < lane {L[i+1]} has {count_next} "
                    f"(inject_warmstart will remap)")

    # 5. Bin copy prefix ordering (temporal check on sorted bin events)
    for s in S:
        evts = solution.bin_events.get(s, [])
        by_sku = defaultdict(list)
        for be in evts:
            by_sku[be.sku].append(be)
        for k, k_evts in by_sku.items():
            k_sorted = sorted(k_evts, key=lambda x: x.fetch_start)
            for i in range(len(k_sorted) - 1):
                if k_sorted[i].presence_end > k_sorted[i + 1].presence_start:
                    violations.append(
                        f"Symmetry (B) at station {s} SKU {k}: "
                        f"B[e={i}].presence_end={k_sorted[i].presence_end} > "
                        f"B[e={i+1}].presence_start={k_sorted[i + 1].presence_start}")

    # 6. Order completeness — every order in the CP model must be assigned
    O_all = handles["O"]
    for o in O_all:
        if o not in solution.order_assignments:
            violations.append(f"Order {o} not assigned in solution")

    # 7. Lane no-overlap — mirrors validate_solution check 2
    lane_usage: dict = defaultdict(list)
    for o, (s_a, ln_a, t_s, t_e) in solution.order_assignments.items():
        lane_usage[(s_a, ln_a)].append((t_s, t_e, o))
    for (s, ln), intervals in lane_usage.items():
        intervals.sort()
        for i in range(len(intervals) - 1):
            _, end_i, o_i = intervals[i]
            start_j, _, o_j = intervals[i + 1]
            if end_i > start_j:
                violations.append(
                    f"Lane overlap at S{s} L{ln}: order {o_i} ends {end_i} > order {o_j} starts {start_j}")

    # 8. Pickface no-overlap — mirrors validate_solution check 3
    for s in S:
        presences = sorted(
            [(be.presence_start, be.presence_end, be.sku)
             for be in solution.bin_events.get(s, [])]
        )
        for i in range(len(presences) - 1):
            _, end_i, k_i = presences[i]
            start_j, _, k_j = presences[i + 1]
            if end_i > start_j:
                violations.append(
                    f"Pickface overlap at S{s}: SKU {k_i} ends {end_i} > SKU {k_j} starts {start_j}")

    # 9. Bin timing consistency — mirrors validate_solution check 6
    rt_dict = handles["rt"]
    rt_return_dict = handles["rt_return"]
    for s in S:
        for be in solution.bin_events.get(s, []):
            k = be.sku
            if k not in rt_dict:
                continue  # unknown SKU (e.g., injected in tests); skip
            if be.fetch_end - be.fetch_start != rt_dict[k]:
                violations.append(
                    f"S{s} SKU {k}: fetch duration {be.fetch_end - be.fetch_start} != rt[{k}]={rt_dict[k]}")
            if be.return_end - be.return_start != rt_return_dict[k]:
                violations.append(
                    f"S{s} SKU {k}: return duration {be.return_end - be.return_start} "
                    f"!= rt_return[{k}]={rt_return_dict[k]}")
            if be.presence_start != be.fetch_end:
                violations.append(
                    f"S{s} SKU {k}: presence_start {be.presence_start} != fetch_end {be.fetch_end}")
            if be.presence_end != be.return_start:
                violations.append(
                    f"S{s} SKU {k}: presence_end {be.presence_end} != return_start {be.return_start}")

    # 10. Block concurrency <= N[k] — mirrors validate_solution check 4
    N_map = handles.get("N")
    if N_map is not None:
        for k in handles["K"]:
            all_blocks = [
                (be.fetch_start, be.return_end)
                for s in S
                for be in solution.bin_events.get(s, [])
                if be.sku == k
            ]
            if len(all_blocks) <= N_map.get(k, 1):
                continue
            sweep = []
            for bs, be_end in all_blocks:
                sweep.append((bs, +1))
                sweep.append((be_end, -1))
            sweep.sort(key=lambda x: (x[0], x[1]))
            concurrent = 0
            for t, delta in sweep:
                concurrent += delta
                if concurrent > N_map[k]:
                    violations.append(
                        f"Block concurrency SKU {k}: {concurrent} > N[{k}]={N_map[k]} at t={t}")
                    break

    # 11. MoveCap — mirrors validate_solution check 5
    mc = handles.get("move_cap")
    if mc is not None:
        sweep = []
        for s in S:
            for be in solution.bin_events.get(s, []):
                sweep.append((be.fetch_start, +1))
                sweep.append((be.fetch_end, -1))
                sweep.append((be.return_start, +1))
                sweep.append((be.return_end, -1))
        sweep.sort(key=lambda x: (x[0], x[1]))
        concurrent = 0
        for t, delta in sweep:
            concurrent += delta
            if concurrent > mc:
                violations.append(f"MoveCap violation: {concurrent} > {mc} at t={t}")
                break

    return violations


class ProgressCollector(CpoSolverListener):
    """
    CP Optimizer listener that records every incumbent improvement.

    records = [
        {
            "time": <solve_time_in_seconds>,
            "best": <current best objective>,
            "bound": <current best bound>,
            "gap": <relative gap>,
        },
        ...
    ]
    """

    def __init__(self):
        super().__init__()
        self.records = []
        self.best_obj = None

    def result_found(self, solver, sres):
        """
        Called by CP Optimizer when a (new) solution is found.
        We only record strict improvements of the objective.
        """
        try:
            if not sres.is_solution():
                print(f"Warning: result_found called with non-solution result: {sres}")
                return

            obj = sres.get_objective_value()
            if obj is None:
                print(f"Warning: result_found called with solution without objective: {sres}")
                return

            # Only keep strict improvements
            if (self.best_obj is None) or (obj < self.best_obj):
                self.best_obj = obj
                rec = {
                    "time": sres.get_solve_time(),
                    "best": obj,
                    "bound": sres.get_objective_bound(),
                    "gap": sres.get_objective_gap(),
                }
                self.records.append(rec)
        except Exception as e:
            print("Exception in ProgressCollector.result_found:", e)


def main():
    ap = argparse.ArgumentParser(description="CP Autostoremodel.")
    ap.add_argument("--stations", type=int, default=1, help="Number of picking stations |S|")
    ap.add_argument("--lanes", type=int, default=4, help="Lanes per station (max concurrent open orders)")
    ap.add_argument("--orders", type=int, default=10, help="Number of orders |O|")
    ap.add_argument("--skus", type=int, default=10, help="Number of SKUs |K|")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--pick", type=int, default=4, help="Constant pick touch time for all SKUs")
    ap.add_argument("--movecap", type=int, default=8, help="Max number of simultaneous moves (F or R)")
    ap.add_argument("--timelimit", type=int, default=5, help="Solver time limit (seconds)")
    ap.add_argument("--no_symmetry_breaking", action="store_true", help="Disable symmetry breaking constraints")
    ap.add_argument("--no_vis", action="store_true", help="Do not plot the schedule")
    ap.add_argument("--horizon", type=int, default=10000, help="Maximal Cmax horizon (0 for unbounded)")
    ap.add_argument("--newdatagen", action="store_true", help="Use new data generation")

    args = ap.parse_args()

    # Determine if symmetry breaking should be added
    add_symmetry_breaking = not args.no_symmetry_breaking

    print("Generating data...")
    if args.newdatagen:
        gen_fn = generate_data
    else:
        gen_fn = _old_generate_data
    S, L, K, orders_req, rt, p, N = gen_fn(
        num_stations=args.stations,
        lanes_per_station=args.lanes,
        num_orders=args.orders,
        num_skus=args.skus,
        seed=args.seed,
        pick_touch_time=args.pick,
    )
    rt_return = dict(rt)  # symmetric by default; customize if needed

    print("=== Generated data ===")
    print(f"Stations S={S}  |L|={len(L)} lanes each")
    # print(f"SKUs K={K}")
    # print(f"Bins of SKUs={N}")
    # print(f"Retrieval times rt_k: {rt}")
    # print(f"Return times rtr_k:   {rt_return}")
    # print(f"Pick touch time p_k:  {p}")
    print("Orders (o -> required SKUs R_o):")
    for o in sorted(orders_req.keys()):
        print(f"  o={o:>2}: {orders_req[o]}")

    if add_symmetry_breaking:
        print("Adding symmetry breaking constraints...")
    if args.horizon > 0:
        print(f"Adding maximal horizon constraint: end <= {args.horizon}")

    mdl, handles = build_model(S, L, K, orders_req, rt, p, rt_return=rt_return,
                               add_symmetry_breaking=add_symmetry_breaking,
                               horizon=args.horizon, move_cap=args.movecap, N=N)
    print("\nSolving...")
    print(f"Number of variables: {len(mdl.get_all_variables())}")
    sol = mdl.solve(
        Workers=1,
        TimeLimit=args.timelimit,
        LogVerbosity="Terse",
        # SolutionLimit=1
    )
    print("Solve status:", sol.get_solve_status())
    if sol:
        # Note: solution printing/extraction happens *after* solve
        # We need the solution object 'sol' for this

        print("Objective value (makespan):", sol.get_objective_values()[0])
        data = []
        for var in mdl.get_all_variables():
            if var.get_type().name == "IntervalVar":  # filter interval variables
                sol_vars = sol.get_var_solution(var)
                if sol_vars.is_present():
                    data.append({
                        "Name": var.get_name(),
                        "Start": sol_vars.get_start(),
                        "End": sol_vars.get_end(),
                        "Length": sol_vars.get_length(),
                        "Presence": sol_vars.is_present()
                    })
        # df = pd.DataFrame(data)
        # pd.set_option("display.max_columns", None)  # Show all columns
        # pd.set_option("display.max_rows", None)  # Show all rows
        # df.sort_values(by=["Start", "End", "Name"], inplace=True)
        # print(df)
        # print("Other intervals are not present.")

        extract_and_print_solution(sol, handles)

        # Plot if the visualizer is available
        if plot_schedule is not None and not args.no_vis:
            try:
                fig = plot_schedule(sol, handles)
                html_file = "./order_station_assign_v5_solution.html"
                write_html(fig, html_file)
                print(f"\nWrote visualization to {html_file}")

            except Exception as e:
                print(f"\n[Visualizer] Error: {e}. Plotting failed.")
        elif args.no_vis:
            print("\nVisualization skipped (--no_vis).")
        else:
            print("\nVisualizer not available.")
    else:
        try:
            mdl.refine_conflict()
            print("--- Conflict Refiner ---")
            print(mdl.get_conflict())
            print("------------------------")
        except Exception as e:
            print(f"Could not refine conflict: {e}")
        print("No solution found.")


if __name__ == "__main__":
    main()
