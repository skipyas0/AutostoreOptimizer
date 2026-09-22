from matheuristic import load_instance
from collections import defaultdict
import argparse
from cp_model import build_model, inject_warmstart
from instance import Instance
from jaccard_similarity import build_similarity_matrix
from neighborhood_selection import strategy_similar_orders
from autostore_heuristic import Solution, BinEvent
from heuristic_rdi_sgc import run_rdi_sgc
from neighborhood_selection import strategy_similar_orders
from autostore_heuristic import Solution, BinEvent

def extract_exo_state(sol, handles, overlap_threshold=0):
    """
    Extracts locked interval coordinates from a solved submodel to pass
    into subsequent decompositions.

    Parameters:
    - sol: The Docplex solution object returned by submodel.solve()
    - handles: The handles dictionary returned by build_model()
    - overlap_threshold: For rolling horizons, only extract intervals that
                         end strictly after this absolute timestamp.
    """
    exo_lanes = defaultdict(list)
    exo_blocks = defaultdict(list)
    exo_moves = []

    if sol is None or not sol.is_solution():
        return dict(exo_lanes), dict(exo_blocks), exo_moves

    def get_bounds(iv):
        vs = sol.get_var_solution(iv)
        if vs is not None and vs.is_present():
            return vs.get_start(), vs.get_end()
        return None

    # 1. Extract Lane Usage
    for (o, s, ln), iv in handles.get("I_os_lane", {}).items():
        bounds = get_bounds(iv)
        if bounds and bounds[1] > overlap_threshold:
            exo_lanes[(s, ln)].append(bounds)

    # 2. Extract SKU Bin Blocks
    for (s, k, e), iv in handles.get("Block", {}).items():
        bounds = get_bounds(iv)
        if bounds and bounds[1] > overlap_threshold:
            exo_blocks[k].append(bounds)

    # 3. Extract Robot Fetch/Return Moves
    for move_dict in (handles.get("F", {}), handles.get("R", {})):
        for key, iv in move_dict.items():
            bounds = get_bounds(iv)
            if bounds and bounds[1] > overlap_threshold:
                exo_moves.append(bounds)

    return dict(exo_lanes), dict(exo_blocks), exo_moves


def cp_solution_to_heuristic(sol, handles, instance):
    if sol is None or not sol.is_solution():
        return None
    
    order_assignments = {}
    bin_events = defaultdict(list)
    pick_events = {}
    
    def get_bounds(iv):
        vs = sol.get_var_solution(iv)
        if vs is not None and vs.is_present():
            return vs.get_start(), vs.get_end()
        return None

    O = handles.get("O", [])
    S = handles.get("S", [])
    L = handles.get("L", [])
    K = handles.get("K", [])
    U = handles.get("U", {})
    orders_req = handles.get("orders_req", {})

    # order_assignments
    for o in O:
        s_sel = next((s for s in S if get_bounds(handles["I_os"][(o, s)])), None)
        if s_sel is not None:
            ln_sel = next((ln for ln in L if get_bounds(handles["I_os_lane"][(o, s_sel, ln)])), None)
            
            if ln_sel is not None:
                st, en = get_bounds(handles["I_os"][(o, s_sel)])
                order_assignments[o] = (s_sel, ln_sel, st, en)
                
                # picks
                for k in orders_req.get(o, []):
                    for e in range(U.get(k, 0)):
                        p_bounds = get_bounds(handles.get("P", {}).get((o, s_sel, k, e)))
                        if p_bounds:
                            pick_events[(o, s_sel, k)] = p_bounds
                            break

    # bin_events
    for s in S:
        for k in K:
            for e in range(U.get(k, 0)):
                f_bounds = get_bounds(handles.get("F", {}).get((s, k, e)))
                b_bounds = get_bounds(handles.get("B", {}).get((s, k, e)))
                r_bounds = get_bounds(handles.get("R", {}).get((s, k, e)))
                if f_bounds and b_bounds and r_bounds:
                    ev = BinEvent(
                        sku=k,
                        copy_id=e,
                        fetch_start=f_bounds[0],
                        fetch_end=f_bounds[1],
                        presence_start=b_bounds[0],
                        presence_end=b_bounds[1],
                        return_start=r_bounds[0],
                        return_end=r_bounds[1],
                        orders_served=[]
                    )
                    bin_events[s].append(ev)

    makespan = max((bounds[3] for bounds in order_assignments.values()), default=0)
    total_moves = sum(len(evts) * 2 for evts in bin_events.values())
    
    return Solution(
        order_assignments=order_assignments,
        bin_events=dict(bin_events),
        makespan=makespan,
        total_moves=total_moves,
        feasible=True,
        pick_events=pick_events
    )

def stitch_heuristic_solutions(sols):
    valid_sols = [s for s in sols if s is not None]
    if not valid_sols:
        return None
    
    order_assignments = {}
    bin_events = defaultdict(list)
    pick_events = {}
    
    for sol in valid_sols:
        order_assignments.update(sol.order_assignments)
        pick_events.update(sol.pick_events)
        for s, evts in sol.bin_events.items():
            bin_events[s].extend(evts)
            
    makespan = max((bounds[3] for bounds in order_assignments.values()), default=0)
    total_moves = sum(len(evts) * 2 for evts in bin_events.values())
    
    return Solution(
        order_assignments=order_assignments,
        bin_events=dict(bin_events),
        makespan=makespan,
        total_moves=total_moves,
        feasible=all(sol.feasible for sol in valid_sols),
        pick_events=pick_events
    )


def create_similarity_partitions(instance, n_partitions=5):
    weighted_jaccard_matrix, _ = build_similarity_matrix(
        instance.O,
        instance.orders_req,
        instance.rt,
        instance.rt_ret,
        normalize=False,
    )

    order_set = set(instance.O)

    partitions = []
    for i in range(n_partitions - 1):
        fake_handles = {"O": list(order_set)}
        selection_result = strategy_similar_orders(
            fake_handles, 1 / n_partitions, weighted_jaccard_matrix
        )
        selected_orders = selection_result.seed_orders
        order_set -= selected_orders
        partitions.append(selected_orders)

    partitions.append(order_set)
    return partitions


def solve_station_decomposition(instance, exo_blocks_in=None, exo_lanes_in=None, exo_moves_in=None, batch_start_time=0):
    n_partitions = len(instance.S)
    partitions = create_similarity_partitions(instance, n_partitions)
    per_station_movecap = instance.movecap // n_partitions if instance.movecap else None
    
    subsolutions = []
    
    # We accumulate exo blocks so stations solved later respect bins used by earlier stations
    current_exo_blocks = defaultdict(list)
    if exo_blocks_in:
        for k, v in exo_blocks_in.items():
            current_exo_blocks[k].extend(v)
            
    for i, p in enumerate(partitions):
        subinstance = Instance(
            [instance.S[i]],
            instance.L,
            instance.K,
            {o: instance.orders_req[o] for o in p},
            instance.rt,
            instance.p,
            instance.N,
            per_station_movecap,
            instance.seed,
            instance.rt_ret,
            instance.pickcap,
        )
        
        # Build and solve CP model
        horizon = batch_start_time + 100000
        submodel, handles = build_model(
            subinstance, 
            add_symmetry_breaking=True, 
            horizon=horizon, 
            exo_lanes=exo_lanes_in, 
            exo_blocks=current_exo_blocks,
            exo_moves=exo_moves_in,
            batch_start_time=batch_start_time
        )
        
        heur_sol = run_rdi_sgc(
            subinstance, 
            horizon=horizon,
            exo_lanes=exo_lanes_in,
            exo_blocks=current_exo_blocks,
            exo_moves=exo_moves_in,
            batch_start_time=batch_start_time
        )
        if heur_sol and heur_sol.feasible:
            sp = inject_warmstart(heur_sol, heur_sol.pick_events, submodel, handles)
            submodel.set_starting_point(sp)
            
        subsolution_cp = submodel.solve(TimeLimit=30, LogVerbosity="Quiet") # configurable time limit
        
        # Extract constraints for horizontal sharing (spatial)
        _, new_blocks, _ = extract_exo_state(subsolution_cp, handles, overlap_threshold=-1)
        for k, blocks in new_blocks.items():
            current_exo_blocks[k].extend(blocks)
            
        heuristic_sol = cp_solution_to_heuristic(subsolution_cp, handles, subinstance)
        subsolutions.append(heuristic_sol)
        
    # Stitch solutions across stations
    return stitch_heuristic_solutions(subsolutions)


def solve_time_decomposition(instance, exo_blocks_in=None, exo_lanes_in=None, exo_moves_in=None, n_time_batches = 5, overlap_coeff = 0.1):
    """
    order_batches: list of lists of order IDs
    Solves them chronologically and stitches the result.
    """
    order_batches = create_similarity_partitions(instance, n_time_batches)
        
    subsolutions = []
    
    current_exo_lanes = defaultdict(list)
    if exo_lanes_in:
        for k, v in exo_lanes_in.items():
            current_exo_lanes[k].extend(v)
            
    current_exo_blocks = defaultdict(list)
    if exo_blocks_in:
        for k, v in exo_blocks_in.items():
            current_exo_blocks[k].extend(v)
            
    current_exo_moves = []
    if exo_moves_in:
        current_exo_moves.extend(exo_moves_in)
    
    current_batch_start = 0
    
    for i, batch in enumerate(order_batches):
        print(f"Solving time decomposition batch {i}")
        subinstance = Instance(
            instance.S,
            instance.L,
            instance.K,
            {o: instance.orders_req[o] for o in batch},
            instance.rt,
            instance.p,
            instance.N,
            instance.movecap,
            instance.seed,
            instance.rt_ret,
            instance.pickcap,
        )
        
        horizon = current_batch_start + 100000
        submodel, handles = build_model(
            subinstance,
            add_symmetry_breaking=True,
            horizon=horizon,
            exo_lanes=current_exo_lanes,
            exo_blocks=current_exo_blocks,
            exo_moves=current_exo_moves,
            batch_start_time=current_batch_start
        )
        
        heur_sol = run_rdi_sgc(
            subinstance, 
            horizon=horizon,
            exo_lanes=current_exo_lanes,
            exo_blocks=current_exo_blocks,
            exo_moves=current_exo_moves,
            batch_start_time=current_batch_start
        )
        if heur_sol and heur_sol.feasible:
            sp = inject_warmstart(heur_sol, heur_sol.pick_events, submodel, handles)
            submodel.set_starting_point(sp)
        
        subsolution_cp = submodel.solve(LogVerbosity="Quiet", TimeLimit=10)
        
        if subsolution_cp is not None and subsolution_cp.is_solution():
            heuristic_sol = cp_solution_to_heuristic(subsolution_cp, handles, subinstance)
            subsolutions.append(heuristic_sol)
            
            if heuristic_sol and heuristic_sol.order_assignments:
                max_end = max(b[3] for b in heuristic_sol.order_assignments.values())
                batch_span = max_end - current_batch_start
                next_batch_start = int(max_end - batch_span * overlap_coeff)
            else:
                next_batch_start = current_batch_start
            
            new_lanes, new_blocks, new_moves = extract_exo_state(
                subsolution_cp, handles, overlap_threshold=next_batch_start 
            )
            
            for key, intervals in new_lanes.items():
                current_exo_lanes[key].extend(intervals)
            for key, intervals in current_exo_lanes.items():
                current_exo_lanes[key] = [iv for iv in intervals if iv[1] > next_batch_start]
                
            for key, intervals in new_blocks.items():
                current_exo_blocks[key].extend(intervals)
            for key, intervals in current_exo_blocks.items():
                current_exo_blocks[key] = [iv for iv in intervals if iv[1] > next_batch_start]
                
            current_exo_moves.extend(new_moves)
            current_exo_moves = [iv for iv in current_exo_moves if iv[1] > next_batch_start]
            
            current_batch_start = next_batch_start            
        else:
            print("Failed to solve time batch")
            break

    return stitch_heuristic_solutions(subsolutions)


def solve_2d_decomposition(instance, n_time_batches=5):
    """
    Combines spatial and temporal decomposition.
    time_batches: list of lists of order IDs
    """
    subsolutions = []
    time_batches = create_similarity_partitions(instance, n_time_batches)
       
    current_exo_lanes = defaultdict(list)
    current_exo_blocks = defaultdict(list)
    current_exo_moves = []
    
    current_batch_start = 0
    
    for batch in time_batches:
        subinstance = Instance(
            instance.S,
            instance.L,
            instance.K,
            {o: instance.orders_req[o] for o in batch},
            instance.rt,
            instance.p,
            instance.N,
            instance.movecap,
            instance.seed,
            instance.rt_ret,
            instance.pickcap,
        )
        
        # We run the spatial decomposition on this time slice
        batch_sol = solve_station_decomposition(
            subinstance,
            exo_blocks_in=current_exo_blocks,
            exo_lanes_in=current_exo_lanes,
            exo_moves_in=current_exo_moves,
            batch_start_time=current_batch_start
        )
        
        if batch_sol:
            subsolutions.append(batch_sol)
            
            # We don't have a CP solution here to extract_exo_state easily, 
            # we need to extract from heuristic solution!
            current_exo_lanes.clear()
            current_exo_blocks.clear()
            current_exo_moves.clear()
            
            for o, (s_sel, ln_sel, st, en) in batch_sol.order_assignments.items():
                current_exo_lanes[(s_sel, ln_sel)].append((st, en))
                
            for s, evts in batch_sol.bin_events.items():
                for ev in evts:
                    current_exo_blocks[ev.sku].append((ev.fetch_start, ev.return_end))
                    current_exo_moves.append((ev.fetch_start, ev.fetch_end))
                    current_exo_moves.append((ev.return_start, ev.return_end))
                    
            # For a rolling horizon, we should ideally filter these to only intervals extending past current_batch_start.
            # But passing all is physically safe (just builds a bigger CP model over time).
            current_batch_start = batch_sol.makespan
        else:
            print("Failed to solve spatial decomposition for time batch")
            break
            
    return stitch_heuristic_solutions(subsolutions)


def solve_2d_decomposition_spatial_first(instance, n_time_batches=5):
    """
    Combines spatial and temporal decomposition by partitioning spatially first,
    then solving each station temporally.
    """
    n_partitions = len(instance.S)
    spatial_partitions = create_similarity_partitions(instance, n_partitions)
    per_station_movecap = instance.movecap // n_partitions if instance.movecap else None
    
    subsolutions = []
    current_exo_blocks = defaultdict(list)
    
    for i, p in enumerate(spatial_partitions):
        print(f"Solving station {i}")
        subinstance = Instance(
            [instance.S[i]],
            instance.L,
            instance.K,
            {o: instance.orders_req[o] for o in p},
            instance.rt,
            instance.p,
            instance.N,
            per_station_movecap,
            instance.seed,
            instance.rt_ret,
            instance.pickcap,
        )
        
        station_sol = solve_time_decomposition(
            subinstance, 
            exo_blocks_in=current_exo_blocks
        )
        
        if station_sol:
            subsolutions.append(station_sol)
            
            for s, evts in station_sol.bin_events.items():
                for ev in evts:
                    current_exo_blocks[ev.sku].append((ev.fetch_start, ev.return_end))
        else:
            print(f"Failed to solve time decomposition for station {instance.S[i]}")
            break
            
    return stitch_heuristic_solutions(subsolutions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Decomposition CP for autostore")
    parser.add_argument(
        "instance",
        type=str,
        help="Name of the precalculated instance folder under precalculated_instances/",
    )
    
    args = parser.parse_args()
    instance= args.instance
    if "precalculated_instances/" in instance:
            instance = instance.split("/")[1]
    (
            instance,
            heur_sol,
            # self.cp_records,
            instance_path,
            instance_config,
        ) = load_instance(None, instance_folder=f"precalculated_instances/{instance}")
    
    stitched_sol = solve_2d_decomposition_spatial_first(instance)
    
    from autostore_heuristic import build_viz_handles
    from schedule_visualizer import plot_schedule, write_html

    mock_sol, handles = build_viz_handles(stitched_sol, instance)
    fig = plot_schedule(mock_sol, handles, show=True)
    html_file = "./autostore_decomposition_solution.html"
    write_html(fig, html_file)
    print(f"\nWrote visualisation to {html_file}")