import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from autostore_heuristic import init_state
from heuristic_cfss_sgc import snapshot_state
from heuristic_rdi_sgc import find_earliest_gap

def test_heuristic_state_intervals():
    state = init_state(S=[1], L=[1, 2], K=[101, 102], N={101: 10, 102: 10}, horizon=1000, move_cap=2)
    assert hasattr(state, "pickface_intervals"), "HeuristicState missing pickface_intervals"
    assert state.pickface_intervals[1] == [], "Intervals not initialized properly"
    
    # Test snapshot deep copy
    state.pickface_intervals[1].append((10, 20))
    new_state = snapshot_state(state)
    assert new_state.pickface_intervals[1] == [(10, 20)]
    new_state.pickface_intervals[1].append((30, 40))
    assert len(state.pickface_intervals[1]) == 1

def test_find_earliest_gap():
    intervals = [(0, 20), (25, 40), (142, 146)]
    assert find_earliest_gap(intervals, ready_time=0, duration=4) == 20
    assert find_earliest_gap(intervals, ready_time=20, duration=4) == 20
    assert find_earliest_gap(intervals, ready_time=0, duration=100) == 40
    assert find_earliest_gap(intervals, ready_time=150, duration=10) == 150
