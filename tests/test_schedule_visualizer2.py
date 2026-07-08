import plotly.graph_objects as go
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schedule_visualizer import plot_schedule


class MockVarSolution:
    def __init__(self, start, end, present=True):
        self._start = start
        self._end = end
        self._present = present

    def is_present(self): return self._present
    def get_start(self): return self._start
    def get_end(self): return self._end


class MockSolution:
    def __init__(self, intervals):
        self.intervals = intervals

    def get_var_solution(self, x):
        return self.intervals.get(x)


def test_plot_schedule_runs():
    I_os_var = "I_os_0_0"
    F_var = "F_0_1_0"
    P_var = "P_0_0_1_0"

    solution = MockSolution({
        I_os_var: MockVarSolution(0, 10),
        F_var: MockVarSolution(0, 5),
        P_var: MockVarSolution(5, 8)
    })

    handles = {
        "I_os": {(0, 0): I_os_var},
        "C": {(0, 1, 0): "C_0_1_0"},
        "P": {(0, 0, 1, 0): P_var},
        "F": {(0, 1, 0): F_var},
        "R": {},
        "B": {},
        "U": {1: 1},
        "orders_req": {0: [1]},
        "S": [0],
        "L": [0],
        "K": [1],
        "O": [0]
    }

    fig = plot_schedule(solution, handles, show=False)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_plot_schedule_computes_metrics():
    I_os_var = "I_os_0_0"
    F_var = "F_0_1_0"
    P_var = "P_0_0_1_0"
    solution = MockSolution({
        I_os_var: MockVarSolution(0, 10),
        F_var: MockVarSolution(0, 5),
        P_var: MockVarSolution(5, 8)
    })
    handles = {
        "I_os": {(0, 0): I_os_var},
        "C": {(0, 1, 0): "C_0_1_0"},
        "P": {(0, 0, 1, 0): P_var},
        "F": {(0, 1, 0): F_var},
        "R": {}, "B": {}, "U": {1: 1},
        "orders_req": {0: [1]}, "S": [0], "L": [0], "K": [1], "O": [0]
    }

    fig = plot_schedule(solution, handles, show=False)
    title = fig.layout.title.text
    assert "Peak Robots:" in title
    assert "Picker Idle:" in title
    assert "SKU Re-fetch Ratio:" in title


def test_plot_schedule_utilization_param():
    I_os_var = "I_os_0_0"
    solution = MockSolution({I_os_var: MockVarSolution(0, 10)})
    handles = {
        "I_os": {(0, 0): I_os_var}, "C": {}, "P": {}, "F": {}, "R": {}, "B": {},
        "U": {}, "orders_req": {0: []}, "S": [0], "L": [0], "K": [], "O": [0]
    }

    # default (bottom)
    fig = plot_schedule(solution, handles, show=False)
    assert len(fig.layout.annotations) == 2  # 1 station + 1 for global plot

    # top
    fig_top = plot_schedule(solution, handles, utilization_plot_top=True, show=False)
    assert len(fig_top.layout.annotations) == 2


def test_plot_schedule_row_heights():
    I_os_var = "I_os_0_0"
    solution = MockSolution({I_os_var: MockVarSolution(0, 10)})
    handles = {
        "I_os": {(0, 0): I_os_var}, "C": {}, "P": {}, "F": {}, "R": {}, "B": {},
        "U": {}, "orders_req": {0: []}, "S": [0], "L": [0], "K": [], "O": [0]
    }

    fig = plot_schedule(solution, handles, show=False)

    # We expect 2 subplots per station (Gantt + Util) + 1 for Global Util
    # For 1 station, total rows = 3
    # Subplot titles count = 2 (Station title + Global title). The Util row title is empty so it's stripped by plotly layout sometimes or it just doesn't have text.
    domain_y = fig.layout.yaxis.domain
    domain_y2 = fig.layout.yaxis2.domain
    # Gantt row (y) should be taller than Util row (y2)
    assert (domain_y[1] - domain_y[0]) > (domain_y2[1] - domain_y2[0])
