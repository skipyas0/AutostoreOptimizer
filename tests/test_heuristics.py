"""Tests for the heuristics package (API contract, cross-compatibility, invariants).

Exact behavioural equivalence with the pre-refactor code is checked separately by
tests/regression/equivalence.py.
"""

import os
import pickle
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datagen import generate_data  # noqa: E402
from heuristics import (  # noqa: E402
    ALL_RULES,
    HEURISTICS,
    ExoState,
    HeuristicSolution,
    Rule,
    resolve,
    run_ama,
    run_heuristic,
    run_partitioned_rdi,
    run_rdi,
    run_sgc,
)

H = 10000


@pytest.fixture(scope="module")
def inst():
    return generate_data(
        num_stations=2, lanes_per_station=2, num_orders=12, num_skus=2400, seed=3, pick_touch_time=4, movecap=5
    )


# Heuristics known to violate PickCap today (docs/KNOWN_ISSUES.md) — validated loosely.
PICKCAP_BROKEN = {"cfss", "cfss_normalized", "gbs_max_sharing", "gbs_critical_path",
                  "gbs_readiness_weighted", "gbs_adaptive"}
# Stitched per-station solutions may break cross-station bin-copy limits.
STITCHED = {"partitioned_rdi_random", "partitioned_rdi_similarity", "partitioned_rdi_load_balanced"}


@pytest.mark.parametrize("name", sorted(HEURISTICS))
def test_registry_heuristic_contract(inst, name):
    if name == "ama":
        pytest.skip("192 SGC runs; covered by test_ama_* with two_phase")
    sol = run_heuristic(name, inst, horizon=H)
    if sol is None:  # gbs_adaptive returns None if no rule is feasible
        assert name == "gbs_adaptive"
        return
    assert isinstance(sol, HeuristicSolution)
    assert sol.meta["name"] == name and sol.meta["runtime"] >= 0
    assert set(sol.order_assignments) == set(inst.O)
    assert sol.makespan == max(e for (_, _, _, e) in sol.order_assignments.values())
    assert sol.total_moves == 2 * sum(len(v) for v in sol.bin_events.values())
    violations = sol.validate(inst, H)
    if name in PICKCAP_BROKEN:
        assert all(v.startswith("PickCap") for v in violations), violations
    elif name in STITCHED:
        assert all("concurrency" in v for v in violations), violations
    else:
        assert violations == []


def test_aliases_resolve():
    assert resolve("load_balanced_greedy") == "lb_sgc"
    assert resolve("RDI-SGC") == "rdi"
    with pytest.raises(ValueError):
        resolve("nope")


def test_default_rules_equal_explicit(inst):
    a = run_sgc(inst, horizon=H)
    b = run_sgc(inst, horizon=H, order_rule=("sharing_degree", False), sku_rule=Rule("demand", True))
    assert a.order_assignments == b.order_assignments
    assert a.pick_events == b.pick_events


def test_all_rules_order():
    assert len(ALL_RULES) == 192
    assert ALL_RULES[0] == (Rule("sum_rt", True), Rule("rt", True))
    assert ALL_RULES[-1] == (Rule("min_copies", False), Rule("demand", False))


def test_ama_is_best_of_sgc_runs(inst):
    sol = run_ama(inst, horizon=H, mode="two_phase")
    runs = sol.meta["runs"]
    assert len(runs) == 16 + 12
    best = min(r["objective"] for r in runs if r["feasible"])
    assert sol.objective(1.0, 0.0, inst.S) == best
    order_rule, sku_rule = sol.meta["best_rule"]
    again = run_sgc(inst, horizon=H, order_rule=order_rule, sku_rule=sku_rule)
    assert again.order_assignments == sol.order_assignments


def test_rdi_exo_state_respected(inst):
    sol = run_rdi(inst, horizon=H, exo=ExoState(batch_start_time=100, moves=[(0, 500)]))
    assert sol.feasible
    assert min(st for (_, _, st, _) in sol.order_assignments.values()) >= 100
    assert min(ev.fetch_start for evs in sol.bin_events.values() for ev in evs) >= 100


def test_partitioned_rdi_meta(inst):
    assignment = {0: inst.O[::2], 1: inst.O[1::2]}
    sol = run_partitioned_rdi(inst, partition=assignment)
    assert sol.meta["assignment"] == {inst.S[0]: inst.O[::2], inst.S[1]: inst.O[1::2]}
    assert set(sol.meta["station_feasible"]) == set(inst.S)
    for o, (s, *_rest) in sol.order_assignments.items():
        assert o in sol.meta["assignment"][s]


def test_stitch_and_pickle_roundtrip(inst):
    a = run_rdi(inst.subinstance(inst.O[:6]), horizon=H)
    b = run_rdi(inst.subinstance(inst.O[6:]), horizon=H)
    st = HeuristicSolution.stitch([a, None, b])
    assert set(st.order_assignments) == set(inst.O)
    assert st.makespan == max(a.makespan, b.makespan)
    assert HeuristicSolution.stitch([None]) is None

    back = pickle.loads(pickle.dumps(st))
    assert back.order_assignments == st.order_assignments
    # objects pickled before `meta` existed
    legacy = HeuristicSolution.__new__(HeuristicSolution)
    state = dict(st.__dict__)
    state.pop("meta")
    legacy.__setstate__(state)
    assert legacy.meta == {}


def test_warmstart_cross_compat(inst):
    cp_model = pytest.importorskip("cp_model")
    sol = run_rdi(inst, horizon=H)
    mdl, handles = cp_model.build_model(inst, add_symmetry_breaking=True, horizon=H)
    sp = sol.to_warmstart(mdl, handles)
    assert sp is not None
    hard = [v for v in sol.validate_warmstart(handles) if not v.startswith("Symmetry")]
    assert hard == []


def test_viz_handles(inst):
    msol, handles = run_sgc(inst, horizon=H).to_viz(inst)
    assert set(handles) >= {"I_os", "F", "B", "R", "P", "C", "U", "S", "L", "K", "O"}
    o = inst.O[0]
    assert any(msol.get_var_solution(handles["I_os"][(o, s)]).is_present() for s in inst.S)


def test_instance_cached_attributes_invalidate(inst):
    sub = inst.subinstance(inst.O[:4])
    dc = sub.demand_count
    assert sum(dc.values()) == sum(len(r) for r in sub.orders_req.values())
    sub.orders_req = {o: sub.orders_req[o] for o in sub.O[:2]}
    assert sum(sub.demand_count.values()) == sum(len(r) for r in sub.orders_req.values())
    assert set(sub.sharing_degree) == set(sub.O)


# ---------------------------------------------------------------------------
# Single-item piggybacking (run_sgc(piggyback_singles=True))
# ---------------------------------------------------------------------------


@pytest.fixture
def toy():
    from instance import Instance

    # order 0 = [0, 1]; singles 1, 2 want SKU 1; single 3 wants SKU 2; order 4 = [0, 2]
    return Instance(
        S=[0], L=[0, 1], K=[0, 1, 2],
        orders_requirements={0: [0, 1], 1: [1], 2: [1], 3: [2], 4: [0, 2]},
        rt={0: 10, 1: 10, 2: 10}, p={0: 4, 1: 4, 2: 4}, N={0: 1, 1: 1, 2: 1}, movecap=None, seed=0,
    )


@pytest.mark.parametrize("load_balanced", [False, True])
def test_piggyback_singles_share_last_bin_on_vacated_lane(toy, load_balanced):
    sol = run_sgc(toy, horizon=1000, order_rule=("order_size", True), sku_rule=("rt", True),
                  load_balanced=load_balanced, piggyback_singles=True)
    assert sol.validate(toy, 1000) == []
    assert sol.meta["n_piggybacked"] == 3
    oa = sol.order_assignments
    # singles 1, 2 follow order 0 on its lane, back to back, from the same SKU-1 visit
    assert oa[1][:2] == oa[0][:2] and oa[2][:2] == oa[0][:2]
    assert oa[1][2] == oa[0][3] and oa[2][2] == oa[1][3]
    ev1 = next(ev for ev in sol.bin_events[0] if ev.sku == 1)
    assert ev1.orders_served == [0, 1, 2]
    # single 3 follows order 4 (whose last SKU is 2) from its SKU-2 visit
    ev2 = next(ev for ev in sol.bin_events[0] if ev.sku == 2)
    assert ev2.orders_served == [4, 3] and oa[3][:2] == oa[4][:2]
    plain = run_sgc(toy, horizon=1000, order_rule=("order_size", True), sku_rule=("rt", True),
                    load_balanced=load_balanced)
    assert sol.total_moves < plain.total_moves


def test_piggyback_is_valid_and_off_by_default(inst):
    assert "n_piggybacked" not in run_sgc(inst, horizon=H).meta
    for lb in (False, True):
        sol = run_sgc(inst, horizon=H, load_balanced=lb, piggyback_singles=True)
        assert sol.feasible and sol.validate(inst, H) == []


def test_ama_passes_piggyback_through(toy):
    sol = run_ama(toy, horizon=1000, mode="two_phase", piggyback_singles=True)
    assert sol.meta["piggyback_singles"] is True
    order_rule, sku_rule = sol.meta["best_rule"]
    again = run_sgc(toy, horizon=1000, order_rule=order_rule, sku_rule=sku_rule, piggyback_singles=True)
    assert again.order_assignments == sol.order_assignments
