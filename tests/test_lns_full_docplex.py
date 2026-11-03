"""Full-model LNS with the docplex backend: freezing and the improvement bound.

Regression tests for two bugs that made every iteration after the first one
infeasible or non-improving:
  * auxiliary variables of a solve result (station_span_*, shift_*) were frozen,
    which pinned every station's makespan;
  * the improvement-bound constraint was never removed (CpoModel.add returns None),
    so bounds of all objectives piled up in the model.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datagen import generate_data  # noqa: E402
from freezing_utils import (  # noqa: E402
    FreezeManager,
    generate_freeze_constraints,
    lns_decision_vars,
)
from heuristics import run_heuristic  # noqa: E402
from matheuristic import prepare_model_docplex  # noqa: E402

H = 10000


def _solved_model():
    inst = generate_data(num_stations=2, lanes_per_station=2, num_orders=10, num_skus=30,
                         seed=3, movecap=3)
    heur = run_heuristic("load_balanced_greedy", inst, horizon=H)
    mdl, handles, sp = prepare_model_docplex({"symmetry_breaking": True, "horizon": H},
                                             inst, heur, "makespan")
    mdl.set_starting_point(sp)
    res = mdl.solve(TimeLimit=2, Workers=1, LogVerbosity="Quiet")
    assert res.is_solution()
    return mdl, handles, res


def test_auxiliary_variables_are_never_frozen():
    mdl, handles, res = _solved_model()
    decision = lns_decision_vars(handles)
    aux = [v.get_var() for v in res.get_all_var_solutions() if v.get_var() not in decision]
    assert any(v.get_name().startswith("station_span") for v in aux)

    # Nothing selected -> every decision variable frozen, no auxiliary one
    fm = FreezeManager(mdl, "docplex", H)
    fm.apply_delta_freezing(res, set(), handles)
    assert set(fm.active_constraints) == decision

    # Everything selected -> nothing frozen
    fm = FreezeManager(mdl, "docplex", H)
    fm.apply_delta_freezing(res, decision, handles)
    assert not fm.active_constraints

    # Non-delta path: same rule
    assert generate_freeze_constraints(mdl, res, handles, decision) == []
    frozen = generate_freeze_constraints(mdl, res, handles, set())
    assert len(frozen) >= len(decision)


def test_improvement_bound_is_removed():
    """The pattern used in Solver.run_solver: keep the expression, then remove it."""
    mdl, handles, res = _solved_model()
    n0 = len(mdl.get_all_expressions())
    bound = handles["makespan"] <= int(res.get_kpis()["makespan"]) - 1
    mdl.add(bound)
    assert len(mdl.get_all_expressions()) == n0 + 1
    mdl.remove(bound)
    assert len(mdl.get_all_expressions()) == n0
