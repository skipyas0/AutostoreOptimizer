#!/usr/bin/env python3
from collections import defaultdict

from docplex.cp.model import CpoModel
from docplex.cp.solver.solver_listener import CpoSolverListener

from instance import Instance


def build_model(instance: Instance, add_symmetry_breaking: bool, horizon: int):
    """
    Intervals per (s,k,e):
      - F[s,k,e] : fetch (size = rt[k])
      - P[s,k,e] : pick  (size = p[k])
      - R[s,k,e] : return(size = rt_return[k])
      - B[s,k,e] : bin presence at station (free size) with start(B)=end(F), end(B)=start(R)

    Station capacity: no_overlap over B[s,*,*] (exactly one bin at station).
    Global single-bin per SKU: no_overlap over {F,B,R} across all stations.
    """

    S, L, K, orders_req, rt, p, N = instance
    move_cap = instance.movecap
    rt_return = instance.rt_ret

    mdl = CpoModel()
    O = sorted(orders_req.keys())

    # --- demand and candidate copy counts ---
    orders_demanding_sku = defaultdict(set)
    need_count = defaultdict(int)
    for o in O:
        for k in orders_req[o]:
            orders_demanding_sku[k].add(o)
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
                I_os_lane[(o, s, ln)] = mdl.interval_var(
                    optional=True, name=f"I_os_lane[{o},{s},{ln}]"
                )

    # Consumptions: one per (order, required SKU, station)
    C = {}
    for o in O:
        for k in orders_req[o]:
            for s in S:
                C[(o, k, s)] = mdl.interval_var(
                    size=p[k], optional=True, name=f"C[{o},{k},{s}]"
                )

    # Fetch / Pick / Return / Bin-Presence, only for active SKUs
    P, F, R, B, Block = {}, {}, {}, {}, {}
    for s in S:
        for k in active_K:
            for e in range(U[k]):
                all_picks_for_this_copy = []
                for o in O:
                    if k in orders_req[o]:
                        P[(o, s, k, e)] = mdl.interval_var(
                            size=p[k], optional=True, name=f"P[{o},{s},{k},{e}]"
                        )
                        all_picks_for_this_copy.append(P[(o, s, k, e)])
                F[(s, k, e)] = mdl.interval_var(
                    size=rt[k], optional=True, name=f"F[{s},{k},{e}]"
                )
                R[(s, k, e)] = mdl.interval_var(
                    size=rt_return[k], optional=True, name=f"R[{s},{k},{e}]"
                )
                B[(s, k, e)] = mdl.interval_var(
                    optional=True, name=f"B[{s},{k},{e}]"
                )  # free size
                # Presence coupling
                mdl.add(mdl.presence_of(B[(s, k, e)]) == mdl.presence_of(F[(s, k, e)]))
                mdl.add(mdl.presence_of(R[(s, k, e)]) == mdl.presence_of(B[(s, k, e)]))

                # The bin arrival (B) is present IFF at least one order's pick (P) uses it.
                # This links the presence of P[(o,s,k,e)] to B[(s,k,e)].
                if all_picks_for_this_copy:
                    # any_P_present is a 0/1 expression that is 1 if any P is present
                    any_P_present = mdl.max(
                        mdl.presence_of(iv) for iv in all_picks_for_this_copy
                    )
                    mdl.add(mdl.presence_of(B[(s, k, e)]) == any_P_present)
                else:
                    # No orders exist, so this B should never be present
                    mdl.add(mdl.presence_of(B[(s, k, e)]) == 0)

                # Temporal links: F -> P -> R
                for o in O:
                    if k in orders_req[o]:
                        mdl.add(mdl.end_before_start(F[(s, k, e)], P[(o, s, k, e)]))
                        mdl.add(
                            mdl.end_before_start(P[(o, s, k, e)], R[(s, k, e)])
                        )  # start(R) >= end(P)
                    # mdl.add(mdl.end_before_end(P[(o, s, k, e)], B[(s, k, e)])) does not help
                    # mdl.add(mdl.start_before_start(B[(s, k, e)], P[(o, s, k, e)]))

                # Bin presence window: [end(F), start(R)]
                mdl.add(mdl.start_at_end(B[(s, k, e)], F[(s, k, e)]))
                mdl.add(mdl.end_at_start(B[(s, k, e)], R[(s, k, e)]))

                # This Block spans from the start of Fetch to the end of Return
                Block[(s, k, e)] = mdl.interval_var(
                    optional=True, name=f"Block[{s},{k},{e}]"
                )
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
                    mdl.sum(mdl.presence_of(I_os_lane[(o, s, i)]) for o in O)
                    >= mdl.sum(mdl.presence_of(I_os_lane[(o, s, i + 1)]) for o in O)
                )

        # (B) Ordered pick copies: for each (s,k), present copies form a prefix and are chained
        for s in S:
            for k in active_K:
                Uk = U[k]
                for e in range(Uk - 1):
                    # if P_{e+1} is present => P_e must be present  (prefix)
                    # mdl.add(mdl.if_then(mdl.presence_of(P[(s, k, e + 1)]) == 1,
                    #                     mdl.presence_of(P[(s, k, e)]) == 1))
                    mdl.add(
                        mdl.if_then(
                            mdl.presence_of(B[(s, k, e + 1)]) == 1,
                            mdl.presence_of(B[(s, k, e)]) == 1,
                        )
                    )
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
    makespan = mdl.max(per_order_end)
    mdl.minimize(makespan)

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
        "makespan": makespan,
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
        print(
            f"Order {o:>3} -> Station {s_sel}, Lane {ln_sel}, Window [{st}, {en}) | SKUs {orders_req[o]}"
        )

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
                                ps, pe = (
                                    iv_start(P[(o, s, k, e)]),
                                    iv_end(P[(o, s, k, e)]),
                                )
                                bs, be = iv_start(B[(s, k, e)]), iv_end(B[(s, k, e)])
                                fs, fe = iv_start(F[(s, k, e)]), iv_end(F[(s, k, e)])
                                rs, re = iv_start(R[(s, k, e)]), iv_end(R[(s, k, e)])
                                events.append(
                                    (
                                        bs,
                                        {
                                            "k": k,
                                            "e": e,
                                            "o": o,
                                            "F": (fs, fe),
                                            "B": (bs, be),
                                            "P": (ps, pe),
                                            "R": (rs, re),
                                        },
                                    )
                                )

        # Group events by bin (k, e)
        bins_data = defaultdict(list)
        for bs, ev in events:
            bins_data[(ev["k"], ev["e"])].append(ev)

        # Sort bins by their start time
        sorted_bins = sorted(bins_data.items(), key=lambda item: item[1][0]["B"][0])

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
            print(
                f"  SKU {k} e={e}: F[{fs},{fe}) B[{bs},{be}) {picks_str} R[{rs},{re})"
                f" | rt={rt.get(k)}, p={p.get(k)}, rtr={rt_ret.get(k)}"
            )

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
                        if (
                            iv_present(Piv)
                            and iv_start(Piv) == st
                            and iv_end(Piv) == en
                        ):
                            chosen = e
                            break
                print(
                    f"Order {o:>3} needs SKU {k:>3} -> uses P[{o},{s_sel},{k},{chosen}] at [{st},{en})"
                )
            else:
                print(f"Order {o:>3} needs SKU {k:>3} -> MISSING (should not happen)")


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
                sp.add_interval_var_solution(
                    iv, presence=True, start=t_start, end=t_end
                )
            else:
                sp.add_interval_var_solution(iv, presence=False)

        # I_os_lane[(o, s, ln)]
        for s in S:
            for ln in L:
                iv_lane = I_os_lane.get((o, s, ln))
                if iv_lane is None:
                    continue

                if is_assigned and s == s_target and ln == ln_target:
                    sp.add_interval_var_solution(
                        iv_lane, presence=True, start=t_start, end=t_end
                    )
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
                sp.add_interval_var_solution(
                    F[(s, k, e)],
                    presence=True,
                    start=be.fetch_start,
                    end=be.presence_start,
                )
                set_vars.add(F[(s, k, e)])

            if (s, k, e) in B:
                sp.add_interval_var_solution(
                    B[(s, k, e)],
                    presence=True,
                    start=be.presence_start,
                    end=be.presence_end,
                )
                set_vars.add(B[(s, k, e)])

            if (s, k, e) in R:
                sp.add_interval_var_solution(
                    R[(s, k, e)],
                    presence=True,
                    start=be.presence_end,
                    end=be.return_end,
                )
                set_vars.add(R[(s, k, e)])

            if (s, k, e) in Block:
                sp.add_interval_var_solution(
                    Block[(s, k, e)],
                    presence=True,
                    start=be.fetch_start,
                    end=be.return_end,
                )
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
                sp.add_interval_var_solution(
                    P[(o, s_sel, k, e)], presence=True, start=ps, end=pe
                )
                set_vars.add(P[(o, s_sel, k, e)])

            # C[(o, k, s)]
            if (o, k, s_sel) in C:
                sp.add_interval_var_solution(
                    C[(o, k, s_sel)], presence=True, start=ps, end=pe
                )
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
                print(
                    f"Warning: result_found called with solution without objective: {sres}"
                )
                return

            sol_dict = {}
            for var_sol in sres.get_all_var_solutions():
                val = var_sol.get_value()
                if hasattr(val, "is_present"):
                    sol_dict[var_sol.get_name()] = {
                        "present": val.is_present(),
                        "start": val.get_start() if val.is_present() else None,
                        "end": val.get_end() if val.is_present() else None,
                    }
                else:
                    sol_dict[var_sol.get_name()] = val

            # Only keep strict improvements
            if (self.best_obj is None) or (obj < self.best_obj):
                self.best_obj = obj
                rec = {
                    "time": sres.get_solve_time(),
                    "best": obj,
                    "bound": sres.get_objective_bound(),
                    "gap": sres.get_objective_gap(),
                    "sol": sol_dict,
                }
                self.records.append(rec)

        except Exception as e:
            print("Exception in ProgressCollector.result_found:", e)
