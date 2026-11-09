"""Fast checks of the sequencing-vs-scheduling experiment code.

python -m pytest seq_vs_sched/tests -q
"""

import pytest

import cp_model
from datagen import generate_data
from heuristics.solution import HeuristicSolution
from seq_vs_sched.fix_schedule import constrain_schedule_to_sequence, greedy_schedule_from_sequence
from seq_vs_sched.metrics import compare
from seq_vs_sched.seq_model import solve_sequence_model
from seq_vs_sched.sequence import SequenceSolution, Visit, extract_sequence, sequence_from_schedule


@pytest.fixture(scope="module")
def inst():
    return generate_data(num_stations=2, lanes_per_station=2, num_orders=6, num_skus=12, seed=3, movecap=3)


@pytest.fixture(scope="module")
def seq(inst):
    s = solve_sequence_model(inst, secondary="balance", time_limit=10, workers=2)
    assert s is not None
    return s


def test_sequence_model_feasible(inst, seq):
    assert seq.validate(inst) == []
    distinct = len({k for v in inst.orders_req.values() for k in v})
    assert seq.n_fetches >= distinct


def test_roundtrip_and_self_similarity(seq):
    again = SequenceSolution.from_dict(seq.to_dict())
    assert again.visits == seq.visits and again.assignment == seq.assignment
    c = compare(seq, again)
    assert all(v == 1.0 for k, v in c.items() if k != "order_prec_pairs")


def test_station_relabelling_is_invisible(seq):
    flip = {s: 1 - s for s in seq.visits}
    other = SequenceSolution({o: flip[s] for o, s in seq.assignment.items()},
                             {flip[s]: vs for s, vs in seq.visits.items()})
    assert compare(seq, other)["assign_match"] == 1.0
    assert compare(seq, other)["visit_seq_sim"] == 1.0


@pytest.mark.parametrize("level", ["sequence", "visits", "assignment"])
def test_fixed_schedule_follows_sequence(inst, seq, level):
    mdl, h = cp_model.build_model(inst, add_symmetry_breaking=True, horizon=0)
    constrain_schedule_to_sequence(mdl, h, seq, level)
    res = mdl.solve(TimeLimit=10, Workers=2, LogVerbosity="Quiet", log_output=None)
    assert res and res.is_solution()
    hs = HeuristicSolution.from_cp(res, h)
    assert hs.validate(inst, h["horizon"]) == []
    got = extract_sequence(res, h)
    assert got.assignment == seq.assignment
    if level in ("sequence", "visits"):
        assert got.n_fetches == seq.n_fetches
    if level == "sequence":
        assert got.visits == seq.visits


def test_lane_semantics_detects_overfull_station(inst):
    o = sorted(inst.orders_req)[:3]
    k = 10**6  # dummy sku, only lanes are checked here
    seq = SequenceSolution({x: 0 for x in o}, {0: [Visit(k, tuple(o))]})
    assert seq.max_open_orders() == {0: 3}


def test_greedy_timing_is_feasible_and_follows_sequence(inst, seq):
    hs = greedy_schedule_from_sequence(inst, seq)
    mdl, h = cp_model.build_model(inst, add_symmetry_breaking=True, horizon=0)
    assert hs.validate(inst, h["horizon"]) == []
    assert sequence_from_schedule(hs).visits == seq.visits
    constrain_schedule_to_sequence(mdl, h, seq, "sequence")
    mdl.set_starting_point(hs.to_warmstart(mdl, h))
    res = mdl.solve(TimeLimit=5, Workers=1, LogVerbosity="Quiet", log_output=None)
    assert res.is_solution() and res.get_kpis()["makespan"] <= hs.makespan
