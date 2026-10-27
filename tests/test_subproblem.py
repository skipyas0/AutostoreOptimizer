"""Correctness tests for the order-based LNS subproblem (subproblem.py).

Checked:
  * a subproblem over all orders is the full model (identical canonical .cpo)
  * the incumbent restricted to the active orders is a feasible warm start of the
    subproblem and reproduces the incumbent's KPIs exactly (exo data and objective
    offsets are consistent with the full model)
  * merged solutions after improving solves are valid full schedules with correct KPIs
  * the improvement-bound shortcut only fires when the frozen part alone breaks the bound
  * an end-to-end LNS run in subproblem mode with merge validation switched on
"""

import json
import os
import pickle
import random
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cp_model  # noqa: E402
import subproblem  # noqa: E402
from cp_model_utils import compare_cpo_files  # noqa: E402
from datagen import generate_data  # noqa: E402
from heuristics import HeuristicSolution, run_heuristic  # noqa: E402
from matheuristic import prepare_model_subproblem  # noqa: E402

H = 10000
BACKENDS = ["docplex"]
CFG = {"iter_time_limit": 5, "workers": 1, "presolve": "On", "search_type": "Auto"}


def _cpp_available():
    try:
        import cp_lns_core

        cp_lns_core.CpLnsModel  # noqa: B018
        import cpp_cp_model

        cpp_cp_model.build_model(
            generate_data(num_stations=1, lanes_per_station=1, num_orders=2,
                          num_skus=20, seed=1, movecap=2),
            add_symmetry_breaking=False, horizon=H,
        )
        return True
    except Exception:
        return False


if _cpp_available():
    BACKENDS.append("cpp")


@pytest.fixture(scope="module", params=[(1, 1), (7, 2)], ids=["pickcap1", "pickcap2"])
def setup(request):
    """Small instance with few SKUs (so bins get shared) and a tight MoveCap."""
    seed, pickcap = request.param
    inst = generate_data(num_stations=3, lanes_per_station=2, num_orders=24, num_skus=60,
                         seed=seed, movecap=4, pickcap=pickcap)
    heur = run_heuristic("load_balanced_greedy", inst, horizon=H)
    assert heur is not None and heur.feasible
    _, handles, sol = prepare_model_subproblem({"horizon": H}, inst, heur)
    return inst, handles, sol


def _kpis(sol, handles):
    hs = HeuristicSolution.from_cp(sol, handles)
    return {
        "makespan": hs.makespan,
        "bin_fetches": sum(len(v) for v in hs.bin_events.values()),
        "total_flow_time": sum(a[3] for a in hs.order_assignments.values()),
    }


def _random_neighbourhoods(handles, n, seed):
    rng = random.Random(seed)
    O = handles["O"]
    for _ in range(n):
        yield rng.sample(O, rng.randint(1, max(1, len(O) // 3)))


def _close_under_sharing(sol, handles, active):
    """Add every order that shares a bin visit with an active order (fixed point)."""
    hs = HeuristicSolution.from_cp(sol, handles)
    visits = [set(be.orders_served) for evts in hs.bin_events.values() for be in evts]
    active = set(active)
    changed = True
    while changed:
        changed = False
        for served in visits:
            if served & active and not served <= active:
                active |= served
                changed = True
    return sorted(active)


def test_all_orders_subproblem_is_full_model(setup, tmp_path):
    inst, handles, sol = setup
    ctx = subproblem.extract_context(sol, handles, inst, handles["O"])
    assert not ctx.exo_lanes and not ctx.exo_moves and not ctx.exo_station_bins
    sub_mdl, _ = subproblem.build_submodel(ctx, "docplex", H, "makespan")
    full_mdl, _ = cp_model.build_model(inst, add_symmetry_breaking=False, horizon=H)
    sub_mdl.export_model(str(tmp_path / "sub.cpo"))
    full_mdl.export_model(str(tmp_path / "full.cpo"))
    same, diff = compare_cpo_files(str(tmp_path / "full.cpo"), str(tmp_path / "sub.cpo"))
    assert same, "".join(diff[:40])


@pytest.mark.parametrize("backend", BACKENDS)
def test_warm_start_reproduces_incumbent(setup, backend):
    inst, handles, sol = setup
    current = _kpis(sol, handles)
    checked = 0
    for active in _random_neighbourhoods(handles, 8, seed=0):
        # Shared frozen visits make the incumbent unrepresentable (see KNOWN_ISSUES)
        active = _close_under_sharing(sol, handles, active)
        ctx = subproblem.extract_context(sol, handles, inst, active)
        assert not ctx.shared_picks
        mdl, sub_handles = subproblem.build_submodel(ctx, backend, H, "makespan")
        raw, _ = subproblem.solve_submodel(mdl, sub_handles, backend, "makespan", -1,
                                           {**CFG, "only_first_sol": True})
        assert raw.get_solve_status() in ("Feasible", "Optimal"), active
        merged = subproblem.to_full_result(sol, ctx, raw, sub_handles, handles, backend,
                                           "makespan")
        assert subproblem.check_merged(merged, handles, inst, H) == []
        reported = merged.get_kpis()
        for name, value in current.items():
            assert reported[name] == value, (name, active)
        checked += 1
    assert checked == 8


@pytest.mark.parametrize("backend", BACKENDS)
def test_improving_iterations_stay_valid(setup, backend):
    inst, handles, sol = setup
    improved = 0
    for active in _random_neighbourhoods(handles, 15, seed=1):
        bound = int(sol.get_kpis()["makespan"]) - 1
        result, _, ctx = subproblem.solve_iteration(
            sol, handles, inst, active, backend=backend, horizon=H,
            objective="makespan", impr_bound=bound, cfg=CFG,
        )
        if not result.is_solution():
            continue
        assert subproblem.check_merged(result, handles, inst, H) == []
        assert result.get_kpis()["makespan"] <= bound
        sol = result
        improved += 1
    assert improved >= 1


def test_bound_shortcut(setup):
    inst, handles, sol = setup
    hs = HeuristicSolution.from_cp(sol, handles)
    critical = [o for o, a in hs.order_assignments.items() if a[3] == hs.makespan]
    others = [o for o in handles["O"] if o not in critical][:3]

    ctx = subproblem.extract_context(sol, handles, inst, others)
    assert ctx.offsets.makespan_lb == hs.makespan
    assert subproblem.violates_bound(ctx, "makespan", hs.makespan - 1)
    assert not subproblem.violates_bound(ctx, "makespan", -1)

    ctx = subproblem.extract_context(sol, handles, inst, critical)
    assert ctx.offsets.makespan_lb < hs.makespan
    assert not subproblem.violates_bound(ctx, "makespan", hs.makespan - 1)


@pytest.mark.parametrize("backend", BACKENDS)
def test_lns_end_to_end(tmp_path, backend):
    from matheuristic import Solver

    inst = generate_data(num_stations=2, lanes_per_station=2, num_orders=16, num_skus=40,
                         seed=5, movecap=3)
    heur = run_heuristic("load_balanced_greedy", inst, horizon=H)
    folder = tmp_path / "GEN5-2-2-40-16-3"
    (folder / "experiments").mkdir(parents=True)
    with open(folder / "instance.pkl", "wb") as f:
        pickle.dump(inst, f)
    with open(folder / "heuristic_solution.pkl", "wb") as f:
        pickle.dump(heur, f)
    with open(folder / "config.json", "w") as f:
        json.dump({"stations": 2, "lanes": 2, "orders": 16, "skus": 40, "movecap": 3,
                   "symmetry_breaking": True, "horizon": H, "time_limit": 60}, f)

    cfg = {
        "runs": 1, "iters": 12, "stagnation_th": 9999, "severity_step": 1,
        "severity_backup": "none", "max_severity": 5, "iter_time_limit": 2.0,
        "delta_freezing": True, "starting_point_for_frozen": True, "rrt_threshold": 0.0,
        "time_limit": None, "backend": backend, "improvement_constr": True,
        "presolve": "On", "workers": 1, "search_type": "Auto", "strat_choice": "uniform",
        "strat_preset": "all", "default_timeslice_width": 100, "default_n_lanes": 1,
        "default_percent": 0.2, "cp_objective": "makespan",
        "cp_objective_phase_duration": 0, "cp_objective_phase_schedule": "m",
        "render_video": False, "only_first_sol": False,
        "lns_model": "subproblem", "validate_merges": True,
    }
    solver = Solver(cfg, instance_path=str(folder))
    assert solver.best_result <= heur.makespan
    final = HeuristicSolution.from_cp(solver.best_solution, solver.handles)
    assert final.validate(inst, horizon=H) == []


@pytest.mark.skipif("cpp" not in BACKENDS, reason="C++ extension not built")
def test_cpp_submodel_matches_docplex(setup, tmp_path):
    """The C++ port builds the same subproblem as docplex (frozen data included)."""
    import cpp_cp_model

    inst, handles, sol = setup
    for i, active in enumerate(_random_neighbourhoods(handles, 3, seed=2)):
        ctx = subproblem.extract_context(sol, handles, inst, active)
        assert ctx.exo_lanes and ctx.exo_moves and ctx.exo_station_bins
        py_mdl, _ = subproblem.build_submodel(ctx, "docplex", H, "makespan")
        cpp_mdl, _ = subproblem.build_submodel(ctx, "cpp", H, "makespan")
        py_path, cpp_path = tmp_path / f"py{i}.cpo", tmp_path / f"cpp{i}.cpo"
        py_mdl.export_model(str(py_path))
        cpp_cp_model.export_model(cpp_mdl, str(cpp_path))
        _, diff = compare_cpo_files(str(py_path), str(cpp_path))
        # The shift-length / wages section exists only in the docplex model (also for
        # the full model), so it is excluded from the comparison.
        diff = [ln for ln in diff if ln[:1] in "+-" and ln[:3] not in ("+++", "---")
                and "shift_len" not in ln and "shift_cost" not in ln]
        assert not diff, "".join(diff[:60])
