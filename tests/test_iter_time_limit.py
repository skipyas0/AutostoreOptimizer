"""Search-only iteration time limit (--iter-time-mode search).

Checked:
  * cp_time_limits maps the config to (engine TimeLimit, search-only limit)
  * the search is aborted ~limit seconds after extraction ends, on both backends,
    and the best solution found so far is kept
  * a solve right after an aborted one is not cut short (no stale abort / timer)
  * full-model and subproblem LNS runs work in search mode
"""

import json
import os
import pickle
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import subproblem  # noqa: E402
from cp_model_utils import cp_time_limits  # noqa: E402
from datagen import generate_data  # noqa: E402
from heuristics import HeuristicSolution, run_heuristic  # noqa: E402
from matheuristic import prepare_model_subproblem  # noqa: E402
from test_subproblem import BACKENDS, H  # noqa: E402

LIMIT = 1.0


def test_cp_time_limits():
    assert cp_time_limits({"iter_time_limit": 2.0}) == (2.0, None)  # legacy configs
    total = {"iter_time_limit": 2.0, "iter_time_mode": "total"}
    assert cp_time_limits(total, remaining=50.0) == (2.0, None)
    search = {"iter_time_limit": 2.0, "iter_time_mode": "search"}
    assert cp_time_limits(search) == (None, 2.0)
    assert cp_time_limits(search, remaining=50.0) == (50.0, 2.0)


@pytest.fixture(scope="module")
def setup():
    """Instance large enough that the search does not finish within LIMIT."""
    inst = generate_data(num_stations=3, lanes_per_station=2, num_orders=40, num_skus=60,
                         seed=1, movecap=4)
    heur = run_heuristic("load_balanced_greedy", inst, horizon=H)
    _, handles, sol = prepare_model_subproblem({"horizon": H}, inst, heur)
    return inst, handles, sol


def _search_time(raw, backend):
    total, extraction, search = subproblem._timings(raw, backend)
    return search


@pytest.mark.parametrize("backend", BACKENDS)
def test_search_is_aborted_after_limit(setup, backend):
    inst, handles, sol = setup
    ctx = subproblem.extract_context(sol, handles, inst, handles["O"])
    cfg = {"iter_time_limit": LIMIT, "iter_time_mode": "search", "workers": 1,
           "presolve": "On", "search_type": "Auto"}
    mdl, sub_handles = subproblem.build_submodel(ctx, backend, H, "makespan")
    for _ in range(2):  # the second solve must not inherit the first abort
        raw, _ = subproblem.solve_submodel(mdl, sub_handles, backend, "makespan", -1, cfg)
        status = raw.get_solve_status()
        assert status == "Feasible", "instance solved to optimality; make it harder"
        assert raw.is_solution()  # best solution survives the abort
        assert LIMIT * 0.9 <= _search_time(raw, backend) <= LIMIT + 0.5


def _write_instance(folder, inst, heur):
    (folder / "experiments").mkdir(parents=True)
    with open(folder / "instance.pkl", "wb") as f:
        pickle.dump(inst, f)
    with open(folder / "heuristic_solution.pkl", "wb") as f:
        pickle.dump(heur, f)
    with open(folder / "config.json", "w") as f:
        json.dump({"stations": 2, "lanes": 2, "orders": 16, "skus": 40, "movecap": 3,
                   "symmetry_breaking": True, "horizon": H, "time_limit": 60}, f)


@pytest.mark.parametrize("lns_model", ["full", "subproblem"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_lns_search_mode(tmp_path, backend, lns_model):
    from matheuristic import Solver

    inst = generate_data(num_stations=2, lanes_per_station=2, num_orders=16, num_skus=40,
                         seed=5, movecap=3)
    heur = run_heuristic("load_balanced_greedy", inst, horizon=H)
    folder = tmp_path / "GEN5-2-2-40-16-3"
    _write_instance(folder, inst, heur)
    cfg = {
        "runs": 1, "iters": 6, "stagnation_th": 9999, "severity_step": 1,
        "severity_backup": "none", "max_severity": 5, "iter_time_limit": 0.5,
        "iter_time_mode": "search", "delta_freezing": True,
        "starting_point_for_frozen": True, "rrt_threshold": 0.0, "time_limit": 120,
        "backend": backend, "improvement_constr": True, "presolve": "On", "workers": 1,
        "search_type": "Auto", "strat_choice": "uniform", "strat_preset": "all",
        "default_timeslice_width": 100, "default_n_lanes": 1, "default_percent": 0.2,
        "cp_objective": "makespan", "cp_objective_phase_duration": 0,
        "cp_objective_phase_schedule": "m", "render_video": False,
        "only_first_sol": False, "lns_model": lns_model, "validate_merges": True,
    }
    solver = Solver(cfg, instance_path=str(folder))
    assert solver.best_result <= heur.makespan
    final = HeuristicSolution.from_cp(solver.best_solution, solver.handles)
    assert final.validate(inst, horizon=H) == []
