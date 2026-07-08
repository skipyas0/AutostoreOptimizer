from heuristic_gbs import solve_heuristic_instance
from heuristic_gbs import schedule_bin_event, check_completions
from heuristic_gbs import BinTask, score_bin_task, GBSState
from heuristic_gbs import seed_initial_lanes, admit_next_order, LaneSlot
from autostore_heuristic import DifferenceArray, BinCopyPool
from heuristic_gbs import (
    precompute_sku_demand, run_gbs
)
from autostore_heuristic import validate_solution
from datagen import generate_data as gen_v5


def _small_v5_instance(stations=1, lanes=2, orders=7, skus=5, seed=42):
    """Generate a small problem instance using v5's generate_data."""
    S, L, K, orders_req, rt, p, N = gen_v5(
        num_stations=stations, lanes_per_station=lanes, num_orders=orders,
        num_skus=skus, seed=seed, pick_touch_time=4,
    )
    rt_ret = dict(rt)
    O = sorted(orders_req.keys())
    return S, L, K, O, orders_req, rt, rt_ret, p, N


def test_precompute_sku_demand():

    O = [1, 2, 3]
    orders_req = {1: [100], 2: [100, 200], 3: [100]}
    rt = {100: 10, 200: 20}
    rt_ret = {100: 5, 200: 10}

    demand = precompute_sku_demand(O, orders_req, rt, rt_ret)

    assert 100 in demand
    assert demand[100].n_orders == 3
    assert demand[100].total_weight == 15  # 10 + 5


def test_seed_and_admit_orders():
    S = [0]
    L = [0, 1]
    O = [10, 11, 12]
    orders_req = {10: [1], 11: [1, 2], 12: [3]}
    rt = {1: 10, 2: 15, 3: 20}
    rt_ret = {1: 5, 2: 5, 3: 5}
    p = {1: 2, 2: 2, 3: 2}

    # sum_rt_asc:
    # 10: 10 (rt[1]) => easiest
    # 11: 25 (rt[1]+rt[2]) => medium
    # 12: 20 (rt[3]) => actually 12 is medium, 11 is hardest.
    # Sorted O: 10, 12, 11

    assignment, unscheduled = seed_initial_lanes(S, L, O, orders_req, rt, rt_ret, p)
    assert len(assignment) == 2
    assert 10 in assignment  # The easiest is the anchor
    assert len(unscheduled) == 1

    # Admit next
    freed_lane = LaneSlot(lane=0, station=0, current_order=None, free_at=100)
    active_orders = set(assignment.keys())
    pending_picks = {o: set(orders_req[o]) for o in active_orders}

    next_order = admit_next_order(freed_lane, unscheduled, active_orders, pending_picks, orders_req, rt, rt_ret)
    assert next_order is not None
    assert next_order in O


def test_score_bin_task():
    # Setup dummy task and state
    task = BinTask(sku=1, concurrent_orders=[10, 11], n_sharing=2,
                   sharing_savings=30.0, earliest_fetch_start=50, copy_id=0, score=0.0)
    gbs_state = GBSState(lane_slots=[], active_orders={10, 11}, pending_picks={10: {1}, 11: {1, 2}}, pickface_free=60, bin_events=[
    ], order_pick_times={}, order_windows={}, move_da=None, bin_pools={}, move_cap=None, order_admission_times={}, order_lanes={})

    rt = {1: 10, 2: 10}
    rt_ret = {1: 5, 2: 5}
    p = {1: 2, 2: 2}
    orders_req = {10: [1], 11: [1, 2]}

    # max_sharing
    score_max = score_bin_task(task, gbs_state, orders_req, rt, rt_ret, p, "max_sharing")
    assert score_max == 2 * (10 + 5)

    # critical_path
    score_cp = score_bin_task(task, gbs_state, orders_req, rt, rt_ret, p, "critical_path")
    assert score_cp == 34  # 11 needs 1 & 2 -> (10+2+5)*2

    # readiness_weighted
    score_rw = score_bin_task(task, gbs_state, orders_req, rt, rt_ret, p, "readiness_weighted")
    assert score_rw == 30.0  # Since fetch_arrival (50+10=60) == pickface_free (60), idle_gap is 0


def test_schedule_and_check():
    # Setup state
    gbs_state = GBSState(
        lane_slots=[LaneSlot(0, 0, 10, 0)], active_orders={10},
        pending_picks={10: {1}}, pickface_free=0, bin_events=[],
        order_pick_times={}, order_windows={10: (0, 0)},
        move_da=DifferenceArray(100),
        bin_pools={1: BinCopyPool(1, 1)}, move_cap=2,
        order_admission_times={10: 0}, order_lanes={10: 0}
    )
    task = BinTask(sku=1, concurrent_orders=[10], n_sharing=1,
                   sharing_savings=0.0, earliest_fetch_start=0, copy_id=0, score=0.0)
    rt = {1: 10, 2: 10}
    rt_ret = {1: 5, 2: 5}
    p = {1: 2, 2: 2}
    orders_req = {10: [1], 11: [2]}

    ev = schedule_bin_event(task, gbs_state, rt, rt_ret, p, 100)
    assert ev is not None
    assert ev.presence_end == 12  # 10 + 2
    assert gbs_state.pickface_free == 12
    assert len(gbs_state.pending_picks[10]) == 0
    assert gbs_state.order_pick_times[(10, 1)] == (10, 12)

    unscheduled = [11]
    newly_admitted = check_completions(gbs_state, unscheduled, orders_req, rt, rt_ret)
    assert len(newly_admitted) == 1
    assert 11 in newly_admitted


def test_solve_heuristic_instance():
    config = {
        "S": [0], "L": [0, 1], "K": [1, 2], "O": [10, 11],
        "orders_req": {10: [1], 11: [2]},
        "rt": {1: 10, 2: 10}, "rt_ret": {1: 5, 2: 5}, "p": {1: 2, 2: 2},
        "N": {1: 1, 2: 1}, "horizon": 1000, "move_cap": 2,
        "gbs_rule": "max_sharing"
    }

    sol = solve_heuristic_instance(config)
    assert sol["status"] == "Feasible"
    assert sol["objective_value"] > 0


def test_run_gbs_integration():
    """Test GBS on a synthetic datagen instance with different scoring rules."""
    S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance(
        stations=1, lanes=4, orders=12, skus=8, seed=42
    )
    horizon = 5000

    for rule in ["max_sharing", "critical_path", "readiness_weighted"]:
        sol = run_gbs(S, L, K, O, orders_req, rt, rt_ret, p, N, horizon, scoring_rule=rule)

        # Test solution validity
        assert sol.feasible, f"GBS failed to produce a feasible solution with rule {rule}"
        assert sol.makespan > 0

        # Validate through autostore_heuristic's strict validation
        errors = validate_solution(sol, S, L, K, O, orders_req, rt, rt_ret, p, N, horizon)
        assert len(errors) == 0, f"Validation failed for GBS with rule {rule}: {errors}"


def test_run_gbs_with_movecap():
    """Test GBS respects global movecap constraints."""
    S, L, K, O, orders_req, rt, rt_ret, p, N = _small_v5_instance(
        stations=1, lanes=2, orders=5, skus=5, seed=101
    )
    horizon = 5000
    move_cap = 1  # strict movecap

    sol = run_gbs(S, L, K, O, orders_req, rt, rt_ret, p, N, horizon, move_cap=move_cap)

    assert sol.feasible
    errors = validate_solution(sol, S, L, K, O, orders_req, rt, rt_ret, p, N, horizon, move_cap=move_cap)
    assert len(errors) == 0, f"MoveCap Validation failed for GBS: {errors}"
