import functools
import types
from collections import defaultdict

import cp_lns_core

from cp_model_utils import CppSolveResult, MockIntervalVar


class MockCpoStartingPoint:
    def __init__(self):
        self.state_dict = {}

    def add_interval_var_solution(
        self, var, presence=False, start=None, end=None, size=None
    ):
        name = var.get_name() if hasattr(var, "get_name") else str(var)
        d = {"present": 1 if presence else 0}
        if start is not None:
            d["start"] = int(start)
        if end is not None:
            d["end"] = int(end)
        if size is not None:
            d["size"] = int(size)
        self.state_dict[name] = d

    def add_integer_var_solution(self, var, value):
        pass


class MockCpoModel:
    def create_empty_solution(self):
        return MockCpoStartingPoint()


class ProgressCollector:
    """
    CP Optimizer progress collector for C++ backend, storing solution deltas.
    Compatible with cp_model.ProgressCollector interface.
    """
    def __init__(self):
        self.records = []

    def get_records(self):
        return self.records

    @property
    def initial_solution(self):
        return self.records[0]["delta"] if self.records else {}

    @property
    def deltas(self):
        return [r["delta"] for r in self.records[1:]] if len(self.records) > 1 else []

    def get_solution(self, index=-1):
        if not self.records:
            return None
        if index < 0:
            index = len(self.records) + index
        sol = {}
        for i in range(index + 1):
            for k, v in self.records[i]["delta"].items():
                if isinstance(v, dict):
                    sol[k] = dict(v)
                else:
                    sol[k] = v
        return sol

    def reconstruct_solutions(self):
        current = {}
        solutions = []
        for rec in self.records:
            for k, v in rec["delta"].items():
                if isinstance(v, dict):
                    current[k] = dict(v)
                else:
                    current[k] = v
            solutions.append(dict(current))
        return solutions


def apply_warmstart(mdl, sp):
    """Applies a starting point (MockCpoStartingPoint or dict) to C++ model."""
    if sp is None:
        return
    warm_names, warm_p, warm_s, warm_e = [], [], [], []
    if hasattr(sp, "state_dict"):
        items = sp.state_dict.items()
    elif isinstance(sp, dict):
        items = sp.items()
    else:
        return
    for name, state in items:
        warm_names.append(name)
        warm_p.append(int(state.get("present", 0)))
        warm_s.append(int(state.get("start", 0)))
        warm_e.append(int(state.get("end", 0)))
    mdl.apply_warm_start(warm_names, warm_p, warm_s, warm_e)


def inject_warmstart(solution, pick_events, mdl, handles):
    """Injects warm start into C++ model using heuristic solution and handles."""
    import cp_model
    mock_mdl = MockCpoModel()
    mock_sp = cp_model.inject_warmstart(solution, pick_events, mock_mdl, handles)
    apply_warmstart(mdl, mock_sp)
    return mock_sp


def build_model(
    instance,
    add_symmetry_breaking: bool,
    horizon: int,
    exo_lanes: dict = None,  # (station, lane) -> list of (start, end)
    exo_blocks: dict = None,  # sku -> list of (start, end)
    exo_moves: list = None,  # list of (start, end) for F/R moves
    batch_start_time: int = 0,
    objective_func: str = "makespan",
):
    S, L, K, orders_req, rt, p, N = instance
    move_cap = instance.movecap
    pick_cap = getattr(instance, 'pickcap', None)
    rt_return = instance.rt_ret
    
    exo_lanes = exo_lanes or defaultdict(list)
    exo_blocks = exo_blocks or defaultdict(list)
    exo_moves = exo_moves or []

    O = sorted(orders_req.keys())

    orders_demanding_sku = defaultdict(set)
    need_count = defaultdict(int)
    for o in O:
        for k in orders_req[o]:
            orders_demanding_sku[k].add(o)
            need_count[k] += 1
    active_K = [k for k in K if need_count[k] > 0]

    U = {k: need_count[k] for k in K}

    # Initialize the Pybind11 C++ model
    cpp_model = cp_lns_core.CpLnsModel(
        S,
        L,
        K,
        orders_req,
        rt,
        p,
        N,
        move_cap,
        pick_cap,
        rt_return,
        active_K,
        U,
        horizon,
        add_symmetry_breaking,
        exo_lanes,
        exo_blocks,
        exo_moves,
        batch_start_time,
        objective_func,
    )

    # Reconstruct handles dictionary with MockIntervalVar
    I_os, I_os_lane = {}, {}
    for o in O:
        for s in S:
            I_os[(o, s)] = MockIntervalVar(f"I_os[{o},{s}]")
            for ln in L:
                I_os_lane[(o, s, ln)] = MockIntervalVar(f"I_os_lane[{o},{s},{ln}]")

    C = {}
    for o in O:
        for k in orders_req[o]:
            for s in S:
                C[(o, k, s)] = MockIntervalVar(f"C[{o},{k},{s}]")

    P, F, R, B, Block = {}, {}, {}, {}, {}
    for s in S:
        for k in active_K:
            for e in range(U[k]):
                for o in O:
                    if k in orders_req[o]:
                        P[(o, s, k, e)] = MockIntervalVar(f"P[{o},{s},{k},{e}]")

                F[(s, k, e)] = MockIntervalVar(f"F[{s},{k},{e}]")
                R[(s, k, e)] = MockIntervalVar(f"R[{s},{k},{e}]")
                B[(s, k, e)] = MockIntervalVar(f"B[{s},{k},{e}]")
                Block[(s, k, e)] = MockIntervalVar(f"Block[{s},{k},{e}]")

    all_intervals_flat = {}
    for dic in [I_os, I_os_lane, C, P, F, R, B, Block]:
        for val in dic.values():
            all_intervals_flat[val.get_name()] = val

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
        "pick_cap": pick_cap,
        "horizon": horizon,
        "all_intervals_flat": all_intervals_flat,
    }

    return CppModelWrapper(cpp_model), handles


class CppModelWrapper:
    def __init__(self, cpp_model):
        self._cpp_model = cpp_model
        self._listener = None

    def add_solver_listener(self, listener):
        self._listener = listener

    def set_starting_point(self, sp):
        apply_warmstart(self._cpp_model, sp)

    def solve(self, *args, **kwargs):
        param_aliases = {
            "TimeLimit": "time_limit",
            "Workers": "workers",
            "RandomSeed": "random_seed",
            "LogVerbosity": "log_verbosity",
            "LogPath": "log_filename",
        }
        for docplex_name, cpp_name in param_aliases.items():
            if docplex_name in kwargs and cpp_name not in kwargs:
                kwargs[cpp_name] = kwargs.pop(docplex_name)

        if "log_output" in kwargs:
            val = kwargs.pop("log_output")
            if isinstance(val, str) and "log_filename" not in kwargs:
                kwargs["log_filename"] = val

        kwargs.pop("solve_with_search_next", None)

        if self._listener is not None:
            kwargs.setdefault("collect_progress", True)

        res = self._cpp_model.solve(*args, **kwargs)

        if self._listener is not None:
            self._listener.records = res.get("progress", [])

        return CppSolveResult(res)

    def __getattr__(self, name):
        return getattr(self._cpp_model, name)


def export_model(model, filename: str):
    """Exports the C++ CP model to a .cpo file."""
    model.export_model(filename)
