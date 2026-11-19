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


def test_close_under_sharing(setup):
    """Production closure = reference fixed point; afterwards no shared frozen visit."""
    inst, handles, sol = setup
    for active in _random_neighbourhoods(handles, 10, seed=3):
        closed = subproblem.close_under_sharing(sol, handles, active)
        assert closed == _close_under_sharing(sol, handles, active)
        assert set(active) <= set(closed)
        assert not subproblem.extract_context(sol, handles, inst, closed).shared_picks


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
        active = subproblem.close_under_sharing(sol, handles, active)
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
        # idle_time / max_picks / sum_makespans of the start solution come from
        # matheuristic.heuristic_kpis
        for name in ("idle_time", "max_picks", "sum_makespans"):
            assert reported[name] == sol.get_kpis()[name], (name, active)
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


@pytest.mark.parametrize("backend", BACKENDS)
def test_sum_makespans_objective(setup, backend):
    """sum_makespans subproblems improve the sum of station makespans with correct KPIs."""
    inst, handles, sol = setup
    improved = 0
    for active in _random_neighbourhoods(handles, 10, seed=4):
        active = subproblem.close_under_sharing(sol, handles, active)
        bound = int(sol.get_kpis()["sum_makespans"]) - 1
        result, _, ctx = subproblem.solve_iteration(
            sol, handles, inst, active, backend=backend, horizon=H,
            objective="sum_makespans", impr_bound=bound, cfg=CFG,
        )
        assert ctx.offsets.lower_bound("sum_makespans") <= sol.get_kpis()["sum_makespans"]
        if not result.is_solution():
            continue
        # check_merged recomputes sum_makespans from the merged schedule
        assert subproblem.check_merged(result, handles, inst, H) == []
        assert result.get_kpis()["sum_makespans"] <= bound
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


@pytest.mark.parametrize("neighbourhood,close_sharing",
                         [("graph", True), ("subproblem", True), ("subproblem", False)],
                         ids=["graph", "closure", "no_closure"])
@pytest.mark.parametrize("cycling", [False, True], ids=["makespan", "cycling"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_lns_end_to_end(tmp_path, backend, cycling, neighbourhood, close_sharing):
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
        "lns_model": "subproblem", "validate_merges": True, "close_sharing": close_sharing,
        "joinable_exo": True, "neighbourhood": neighbourhood,
        "improvement_ref": "current",
    }
    if cycling:
        cfg.update(runs=2, iters=8, cp_objective_phase_duration=1,
                   cp_objective_phase_schedule="mfbis")
    solver = Solver(cfg, instance_path=str(folder))
    assert solver.best_result <= heur.makespan
    final = HeuristicSolution.from_cp(solver.best_solution, solver.handles)
    assert final.validate(inst, horizon=H) == []
    if cycling:
        # every run starts at the beginning of the objective cycle
        seqs = [list(g.cp_obj) for _, g in solver.vlg.df.groupby("run_id")]
        assert len(seqs) == 2 and seqs[0] == seqs[1], seqs
        assert set(seqs[0]) == {"makespan", "total_flow_time", "bin_fetches", "idle_time",
                                "sum_makespans"}


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


# --------------------------------------------------------------------------- #
# Joinable frozen visits (exo_visits): validation against the full CP model
# --------------------------------------------------------------------------- #
KPIS = ("makespan", "bin_fetches", "total_flow_time", "idle_time", "max_picks",
        "sum_makespans")
DECISION = ("I_os", "I_os_lane", "C", "P", "F", "R", "B", "Block")


@pytest.fixture(scope="module", params=[(1, 1), (7, 2)], ids=["pickcap1", "pickcap2"])
def shared_setup(request):
    """Small instance with an RDI incumbent, in which about half the visits are shared."""
    seed, pickcap = request.param
    inst = generate_data(num_stations=2, lanes_per_station=2, num_orders=16, num_skus=12,
                         seed=seed, movecap=4, pickcap=pickcap)
    heur = run_heuristic("rdi", inst, horizon=H)
    assert heur is not None and heur.feasible
    _, handles, sol = prepare_model_subproblem({"horizon": H}, inst, heur)
    return inst, handles, sol


def _fix(mdl, var, value):
    """Fix an interval variable of a docplex model to a var_sols entry."""
    if value is not None and value.get("present"):
        mdl.add(mdl.presence_of(var) == 1)
        mdl.add(mdl.start_of(var) == value["start"])
        mdl.add(mdl.end_of(var) == value["end"])
    else:
        mdl.add(mdl.presence_of(var) == 0)


def _full_model_fixed(inst, sol, objective="makespan", active=None):
    """Full docplex model with the incumbent ``sol`` fixed.

    ``active=None``: every decision variable is fixed (a feasibility / KPI check).
    Otherwise the reference for a subproblem: the frozen orders' windows, lanes,
    consumptions and picks are fixed, and so are F, B, R, Block of every visit that
    serves a frozen order. Everything else is free, including the active orders'
    picks on those visits.
    """
    mdl, h = cp_model.build_model(inst, add_symmetry_breaking=False, horizon=H,
                                  objective_func=objective)
    vs = sol.var_sols

    def present(var):
        value = vs.get(var.get_name())
        return value is not None and value.get("present")

    if active is None:
        for group in DECISION:
            for var in h[group].values():
                _fix(mdl, var, vs.get(var.get_name()))
        return mdl, h

    active = set(active)
    for group in ("I_os", "I_os_lane", "C", "P"):
        for key, var in h[group].items():
            if key[0] not in active:
                _fix(mdl, var, vs.get(var.get_name()))
    spare = {}  # (s, k) -> free copy indices still allowed
    for (s, k, e), b in h["B"].items():
        demand = h["orders_demanding_sku"][k]
        frozen = [o for o in demand if o not in active and present(h["P"][(o, s, k, e)])]
        if present(b) and frozen:
            for group in ("F", "B", "R", "Block"):
                var = h[group][(s, k, e)]
                _fix(mdl, var, vs.get(var.get_name()))
            continue
        # Copy indices are interchangeable (no symmetry breaking), and the active orders
        # need at most one visit per order line: the surplus free copies are switched
        # off, otherwise CP Optimizer rarely proves optimality of the full model.
        left = spare.setdefault((s, k), len(demand & active))
        if left > 0:
            spare[(s, k)] = left - 1
        else:
            for group in ("F", "B", "R", "Block"):
                mdl.add(mdl.presence_of(h[group][(s, k, e)]) == 0)
    return mdl, h


def _assert_feasible_in_full_model(inst, result):
    """The merged solution satisfies the full CP model and its KPIs are the reported ones."""
    mdl, _ = _full_model_fixed(inst, result)
    res = mdl.solve(TimeLimit=30, Workers=1, LogVerbosity="Quiet")
    assert res.is_solution(), "merged solution is infeasible in the full CP model"
    full, reported = res.get_kpis(), result.get_kpis()
    for name in KPIS:
        assert full[name] == reported[name], (name, full[name], reported[name])


def _small_neighbourhoods(handles, n, seed, max_size=4):
    rng = random.Random(seed)
    for _ in range(n):
        yield rng.sample(handles["O"], rng.randint(1, max_size))


def _joins(ctx, result, handles):
    """Active orders' picks on frozen visits in ``result`` that are new (not in the incumbent)."""
    vs = result.var_sols
    found = []
    for (s, k), slots in ctx.frozen_slots.items():
        for e in slots:
            for o in ctx.active_orders:
                key = (o, s, k, e)
                if key in handles["P"] and key not in ctx.warm_px \
                        and vs[handles["P"][key].get_name()].get("present"):
                    found.append(key)
    return found


def test_pick_fits():
    fits = subproblem._pick_fits
    assert fits((0, 10), [], 10, 1) and not fits((0, 10), [], 11, 1)
    assert fits((0, 30), [(0, 10), (20, 30)], 10, 1)
    assert not fits((0, 30), [(0, 10), (15, 30)], 10, 1)
    # PickCap 2: one frozen pick leaves room everywhere, two overlapping ones do not
    assert fits((0, 10), [(0, 10)], 10, 2)
    assert not fits((0, 10), [(0, 10), (0, 10)], 10, 2)
    assert not fits((0, 20), [(0, 12), (8, 20)], 10, 2)  # free parts 8 + 8, not contiguous


def test_joinable_exo_off_is_the_old_subproblem(setup):
    inst, handles, sol = setup
    for active in _random_neighbourhoods(handles, 5, seed=6):
        ctx = subproblem.extract_context(sol, handles, inst, active, joinable_exo=False)
        assert not ctx.exo_visits and not ctx.warm_px
        _, sub_handles = subproblem.build_submodel(ctx, "docplex", H, "makespan")
        assert not sub_handles["PX"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_exo_visits_keep_the_incumbent(shared_setup, backend):
    """Without the closure, the warm start is a feasible solution of the subproblem with
    the incumbent's KPIs. The variables are fixed to the warm start, so a starting point
    that CP Optimizer would silently repair fails here."""
    inst, handles, sol = shared_setup
    with_shared = 0
    for active in _random_neighbourhoods(handles, 10, seed=0):
        ctx = subproblem.extract_context(sol, handles, inst, active)
        assert len(ctx.warm_px) == len(ctx.shared_picks)
        with_shared += bool(ctx.shared_picks)
        mdl, sub_handles = subproblem.build_submodel(ctx, backend, H, "makespan")
        variables = [v for group in DECISION + ("PX",) for v in sub_handles[group].values()]
        if backend == "docplex":
            sp = subproblem.warm_start(ctx, mdl, sub_handles)
            for var in variables:
                vsol = sp.get_var_solution(var)
                assert vsol is not None, var.get_name()  # complete starting point
                value = {"present": vsol.is_present()}
                if vsol.is_present():
                    value.update(start=vsol.get_start(), end=vsol.get_end())
                _fix(mdl, var, value)
        else:
            import cpp_cp_model

            state = subproblem.warm_start(ctx, cpp_cp_model.MockCpoModel(),
                                          sub_handles).state_dict
            names = [v.get_name() for v in variables]
            assert set(names) <= set(state)  # complete starting point
            mdl.apply_delta_freezing(
                names, [state[n]["present"] for n in names],
                [state[n].get("start", 0) for n in names],
                [state[n].get("end", 0) for n in names], [])
        res, _ = subproblem.solve_submodel(mdl, sub_handles, backend, "makespan", -1, CFG)
        assert res.is_solution(), active
        for name in KPIS:
            assert res.get_kpis()[name] == sol.get_kpis()[name], (name, active)
    assert with_shared >= 3


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("objective", KPIS)
def test_subproblem_optimum_equals_frozen_full_model(shared_setup, objective, backend):
    """A subproblem with joinable frozen visits has the same optimum as the full model
    in which the frozen orders and the visits serving them are fixed."""
    inst, handles, sol = shared_setup
    compared = with_exo = 0
    cfg = {**CFG, "iter_time_limit": 10}
    for active in _small_neighbourhoods(handles, 8, seed=21):
        ctx = subproblem.extract_context(sol, handles, inst, active)
        mdl, sub_handles = subproblem.build_submodel(ctx, backend, H, objective)
        raw, _ = subproblem.solve_submodel(mdl, sub_handles, backend, objective, -1, cfg)
        full_mdl, _ = _full_model_fixed(inst, sol, objective, active)
        ref = full_mdl.solve(TimeLimit=10, Workers=1, LogVerbosity="Quiet")
        assert raw.is_solution() and ref.is_solution(), active
        sub_val, ref_val = raw.get_kpis()[objective], ref.get_kpis()[objective]
        sub_opt = raw.get_solve_status() == "Optimal"
        ref_opt = ref.get_solve_status() == "Optimal"
        # A proven optimum of one model can never be beaten by the other
        if sub_opt:
            assert ref_val >= sub_val, (objective, active)
        if ref_opt:
            assert sub_val >= ref_val, (objective, active)
        if not (sub_opt and ref_opt):
            continue
        merged = subproblem.to_full_result(sol, ctx, raw, sub_handles, handles, backend,
                                           objective)
        assert subproblem.check_merged(merged, handles, inst, H) == []
        _assert_feasible_in_full_model(inst, merged)
        compared += 1
        with_exo += bool(ctx.exo_visits)
    assert compared >= 3 and with_exo >= 2, (compared, with_exo)


def _full_model_with_subproblem_freeze_set(inst, handles, sol, active, objective, backend):
    """Full model (no symmetry breaking) frozen the way the full-model LNS freezes it,
    with --full-neighbourhood subproblem."""
    from freezing_utils import subproblem_optimized_set

    free = {v.get_name() for v in subproblem_optimized_set(handles, sol, active)}
    if backend == "docplex":
        mdl, h = cp_model.build_model(inst, add_symmetry_breaking=False, horizon=H,
                                      objective_func=objective)
        for group in DECISION:
            for var in h[group].values():
                if var.get_name() not in free:
                    _fix(mdl, var, sol.var_sols.get(var.get_name()))
        return mdl, h
    import cpp_cp_model

    mdl, h = cpp_cp_model.build_model(inst, add_symmetry_breaking=False, horizon=H,
                                      objective_func=objective)
    names = [n for n in sol.var_sols if n not in free]  # as Solver.run with backend cpp
    vs = [sol.var_sols[n] for n in names]
    mdl.apply_delta_freezing(names, [1 if v.get("present") else 0 for v in vs],
                             [v.get("start", 0) if v.get("present") else 0 for v in vs],
                             [v.get("end", 0) if v.get("present") else 0 for v in vs], [])
    return mdl, h


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("objective", ["makespan", "bin_fetches", "total_flow_time"])
def test_full_neighbourhood_subproblem_matches_subproblem(shared_setup, objective, backend):
    """--full-neighbourhood subproblem frees exactly the subproblem's variables: the frozen
    full model and the subproblem (joinable exo visits) have the same optimum."""
    inst, handles, sol = shared_setup
    cfg = {**CFG, "iter_time_limit": 10}
    compared = with_exo = 0
    for active in _small_neighbourhoods(handles, 8, seed=21):
        ctx = subproblem.extract_context(sol, handles, inst, active)
        mdl, sub_handles = subproblem.build_submodel(ctx, backend, H, objective)
        raw, _ = subproblem.solve_submodel(mdl, sub_handles, backend, objective, -1, cfg)
        full_mdl, full_h = _full_model_with_subproblem_freeze_set(inst, handles, sol, active,
                                                                  objective, backend)
        ref, _ = subproblem.solve_submodel(full_mdl, full_h, backend, objective, -1, cfg)
        assert raw.is_solution() and ref.is_solution(), active  # incumbent is feasible in both
        sub_val, ref_val = raw.get_kpis()[objective], ref.get_kpis()[objective]
        sub_opt = raw.get_solve_status() == "Optimal"
        ref_opt = ref.get_solve_status() == "Optimal"
        if sub_opt:
            assert ref_val >= sub_val, (objective, active, ref_val, sub_val)
        if ref_opt:
            assert sub_val >= ref_val, (objective, active, sub_val, ref_val)
        if sub_opt and ref_opt:
            compared += 1
            with_exo += bool(ctx.exo_visits)
    assert compared >= 3 and with_exo >= 2, (compared, with_exo)


def _graph_neighbourhoods(handles, n, seed):
    """Graph-traversal neighbourhoods (get_associated_indices) from small random seeds."""
    from types import SimpleNamespace

    rng = random.Random(seed)
    O, K = sorted(handles["O"]), sorted(handles["orders_demanding_sku"])
    for _ in range(n):
        yield SimpleNamespace(seed_orders=rng.sample(O, rng.randint(1, 2)), seed_stations=[],
                              seed_skus=rng.sample(K, rng.randint(0, 1)))


def _full_model_graph_frozen(inst, handles, sol, seeds, objective, backend):
    """Full model without symmetry breaking, frozen like the full-model LNS (graph nbh.)."""
    from freezing_utils import get_optimized_set

    free = {v.get_name() for v in get_optimized_set(handles, sol, seeds)}
    if backend == "docplex":
        mdl, h = cp_model.build_model(inst, add_symmetry_breaking=False, horizon=H,
                                      objective_func=objective)
        for group in DECISION:
            for var in h[group].values():
                if var.get_name() not in free:
                    _fix(mdl, var, sol.var_sols.get(var.get_name()))
        return mdl, h, free
    import cpp_cp_model

    mdl, h = cpp_cp_model.build_model(inst, add_symmetry_breaking=False, horizon=H,
                                      objective_func=objective)
    names = [n for n in sol.var_sols if n not in free]  # as Solver.run with backend cpp
    vs = [sol.var_sols[n] for n in names]
    mdl.apply_delta_freezing(names, [1 if v.get("present") else 0 for v in vs],
                             [v.get("start", 0) if v.get("present") else 0 for v in vs],
                             [v.get("end", 0) if v.get("present") else 0 for v in vs], [])
    return mdl, h, free


def _graph_ctx(sol, handles, inst, seeds):
    from freezing_utils import get_associated_indices

    act_o, _, _, act_e = get_associated_indices(handles, sol, seeds)
    return subproblem.extract_context_graph(sol, handles, inst, act_o, act_e)


def _same(a, b):
    a, b = a or ABSENT_, b or ABSENT_
    if not a.get("present") or not b.get("present"):
        return bool(a.get("present")) == bool(b.get("present"))
    return (a["start"], a["end"]) == (b["start"], b["end"])


ABSENT_ = {"present": False}


def _assert_frozen_unchanged(sol, merged, handles, free):
    """Every decision variable the full-model LNS would freeze keeps its incumbent value."""
    for group in DECISION:
        for var in handles[group].values():
            name = var.get_name()
            if name not in free:
                assert _same(sol.var_sols.get(name), merged.var_sols.get(name)), name


@pytest.mark.parametrize("backend", BACKENDS)
def test_graph_subproblem_keeps_the_incumbent(shared_setup, backend):
    """Graph neighbourhood: the warm start, with every subproblem variable fixed to it
    (fixings included), is feasible and has the incumbent's KPIs."""
    inst, handles, sol = shared_setup
    pinned = 0
    for seeds in _graph_neighbourhoods(handles, 10, seed=3):
        ctx = _graph_ctx(sol, handles, inst, seeds)
        pinned += bool(ctx.pinned)
        mdl, sub_handles = subproblem.build_submodel(ctx, backend, H, "makespan")
        variables = [v for group in DECISION + ("PX",) for v in sub_handles[group].values()]
        if backend == "docplex":
            sp = subproblem.warm_start(ctx, mdl, sub_handles)
            for var in variables:
                vsol = sp.get_var_solution(var)
                assert vsol is not None, var.get_name()
                value = {"present": vsol.is_present()}
                if vsol.is_present():
                    value.update(start=vsol.get_start(), end=vsol.get_end())
                _fix(mdl, var, value)
        else:
            import cpp_cp_model

            state = subproblem.warm_start(ctx, cpp_cp_model.MockCpoModel(),
                                          sub_handles).state_dict
            names = [v.get_name() for v in variables]
            assert set(names) <= set(state)
            mdl.apply_delta_freezing(
                names, [state[n]["present"] for n in names],
                [state[n].get("start", 0) for n in names],
                [state[n].get("end", 0) for n in names], [])
        res, _ = subproblem.solve_submodel(mdl, sub_handles, backend, "makespan", -1, CFG)
        assert res.is_solution(), seeds
        for name in KPIS:
            assert res.get_kpis()[name] == sol.get_kpis()[name], (name, seeds)
    assert pinned >= 3, pinned


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("objective", KPIS)
def test_graph_subproblem_matches_full_model(shared_setup, objective, backend):
    """Graph neighbourhood: same optimum as the full model frozen by the full-model LNS;
    the merged solution is feasible, reports its KPIs, and leaves every variable the
    full-model LNS freezes at its incumbent value."""
    inst, handles, sol = shared_setup
    cfg = {**CFG, "iter_time_limit": 10}
    compared = pinned = 0
    for seeds in _graph_neighbourhoods(handles, 6, seed=5):
        ctx = _graph_ctx(sol, handles, inst, seeds)
        mdl, sub_handles = subproblem.build_submodel(ctx, backend, H, objective)
        raw, _ = subproblem.solve_submodel(mdl, sub_handles, backend, objective, -1, cfg)
        full_mdl, full_h, free = _full_model_graph_frozen(inst, handles, sol, seeds,
                                                          objective, backend)
        ref, _ = subproblem.solve_submodel(full_mdl, full_h, backend, objective, -1, cfg)
        assert raw.is_solution() and ref.is_solution(), seeds
        sub_val, ref_val = raw.get_kpis()[objective], ref.get_kpis()[objective]
        sub_opt = raw.get_solve_status() == "Optimal"
        ref_opt = ref.get_solve_status() == "Optimal"
        if sub_opt:
            assert ref_val >= sub_val, (objective, seeds, ref_val, sub_val)
        if ref_opt:
            assert sub_val >= ref_val, (objective, seeds, sub_val, ref_val)
        merged = subproblem.to_full_result(sol, ctx, raw, sub_handles, handles, backend,
                                           objective)
        assert subproblem.check_merged(merged, handles, inst, H) == []
        _assert_feasible_in_full_model(inst, merged)
        _assert_frozen_unchanged(sol, merged, handles, free)
        if sub_opt and ref_opt:
            compared += 1
            pinned += bool(ctx.pinned)
    assert compared >= 3 and pinned >= 2, (compared, pinned)


@pytest.mark.parametrize("backend", BACKENDS)
def test_active_orders_join_frozen_visits(shared_setup, backend):
    """Minimising fetches, active orders move onto frozen visits; every merged solution
    is feasible in the full CP model."""
    inst, handles, sol = shared_setup
    joined = solved = 0
    for active in _random_neighbourhoods(handles, 12, seed=8):
        bound = int(sol.get_kpis()["bin_fetches"]) - 1
        result, _, ctx = subproblem.solve_iteration(
            sol, handles, inst, active, backend=backend, horizon=H,
            objective="bin_fetches", impr_bound=bound, cfg=CFG,
        )
        if not result.is_solution():
            continue
        assert subproblem.check_merged(result, handles, inst, H) == []
        _assert_feasible_in_full_model(inst, result)
        assert result.get_kpis()["bin_fetches"] <= bound
        joined += bool(_joins(ctx, result, handles))
        solved += 1
        sol = result
    assert solved >= 1 and joined >= 1, (solved, joined)


@pytest.mark.skipif("cpp" not in BACKENDS, reason="C++ extension not built")
@pytest.mark.parametrize("objective", ["makespan", "idle_time", "max_picks"])
def test_cpp_submodel_matches_docplex_with_exo_visits(shared_setup, tmp_path, objective):
    """Same as test_cpp_submodel_matches_docplex, on subproblems with joinable frozen visits."""
    import cpp_cp_model

    inst, handles, sol = shared_setup
    with_exo = 0
    for i, active in enumerate(_random_neighbourhoods(handles, 6, seed=2)):
        ctx = subproblem.extract_context(sol, handles, inst, active)
        with_exo += bool(ctx.exo_visits)
        py_mdl, py_handles = subproblem.build_submodel(ctx, "docplex", H, objective)
        cpp_mdl, cpp_handles = subproblem.build_submodel(ctx, "cpp", H, objective)
        assert set(py_handles["PX"]) == set(cpp_handles["PX"])
        assert bool(py_handles["PX"]) == bool(ctx.exo_visits)
        py_path, cpp_path = tmp_path / f"py{i}.cpo", tmp_path / f"cpp{i}.cpo"
        py_mdl.export_model(str(py_path))
        cpp_cp_model.export_model(cpp_mdl, str(cpp_path))
        _, diff = compare_cpo_files(str(py_path), str(cpp_path))
        diff = [ln for ln in diff if ln[:1] in "+-" and ln[:3] not in ("+++", "---")
                and "shift_len" not in ln and "shift_cost" not in ln]
        if objective == "idle_time":
            # The two APIs print the idle-time sum in different forms (also before
            # exo_visits); its value is checked by the optimum-equality tests.
            diff = [ln for ln in diff if not ln[1:].startswith("minimize(")]
        assert not diff, "".join(diff[:60])
    assert with_exo >= 3, with_exo
