"""Tests for metrics module."""
from analyze_heuristics.metrics import compute_cp_gap
from analyze_heuristics.metrics import _add_metrics, compute_config_metrics
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))


# 2 instances × 3 configs.
# Instance 0: A=100, B=120, C=110
# Instance 1: A=90, B=90, C=95  (A and B tie)
DF_SMALL = pd.DataFrame({
    "module":      ["heuristic_ama_sgc"] * 6,
    "instance_id": ["inst_0"] * 3 + ["inst_1"] * 3,
    "config_id":   ["A", "B", "C"] * 2,
    "order_attr":  ["max_rt", "sum_rt", "max_rt"] * 2,
    "order_desc":  [True, True, False] * 2,
    "bin_attr":    ["rt", "rt", "rt"] * 2,
    "bin_desc":    [True, True, True] * 2,
    "feasible":    [True] * 6,
    "objective":   [100.0, 120.0, 110.0, 90.0, 90.0, 95.0],
    "total_moves": [10, 12, 11, 9, 11, 10],
    "phase":       ["full_grid"] * 6,
    "parameter":   ["stations"] * 6,
    "value":       [1] * 6,
    "seed":        [0, 0, 0, 1, 1, 1],
})


def test_add_metrics_gap_instance0():
    df = _add_metrics(DF_SMALL.copy(), "objective")
    inst0 = df[df["instance_id"] == "inst_0"].set_index("config_id")
    assert inst0.loc["A", "gap_objective"] == pytest.approx(0.0)
    assert inst0.loc["B", "gap_objective"] == pytest.approx(20.0)
    assert inst0.loc["C", "gap_objective"] == pytest.approx(10.0)


def test_add_metrics_gap_instance1_with_tie():
    df = _add_metrics(DF_SMALL.copy(), "objective")
    inst1 = df[df["instance_id"] == "inst_1"].set_index("config_id")
    assert inst1.loc["A", "gap_objective"] == pytest.approx(0.0)
    assert inst1.loc["B", "gap_objective"] == pytest.approx(0.0)
    assert inst1.loc["C", "gap_objective"] == pytest.approx(100.0 * 5 / 90, rel=1e-3)


def test_add_metrics_rank_no_tie():
    df = _add_metrics(DF_SMALL.copy(), "objective")
    inst0 = df[df["instance_id"] == "inst_0"].set_index("config_id")
    assert inst0.loc["A", "rank_objective"] == 1
    assert inst0.loc["C", "rank_objective"] == 2
    assert inst0.loc["B", "rank_objective"] == 3


def test_add_metrics_rank_dense_tie():
    df = _add_metrics(DF_SMALL.copy(), "objective")
    inst1 = df[df["instance_id"] == "inst_1"].set_index("config_id")
    # A=90, B=90 -> both rank 1; C=95 -> rank 2 (dense)
    assert inst1.loc["A", "rank_objective"] == 1
    assert inst1.loc["B", "rank_objective"] == 1
    assert inst1.loc["C", "rank_objective"] == 2


def test_compute_config_metrics_structure():
    result = compute_config_metrics(DF_SMALL, "heuristic_ama_sgc")
    assert "objective" in result
    assert "total_moves" in result
    for key in ("raw", "gap_agg", "rank_agg", "winrate"):
        assert key in result["objective"], f"Missing key: {key}"


def test_compute_config_metrics_gap_agg():
    result = compute_config_metrics(DF_SMALL, "heuristic_ama_sgc")
    gap_agg = result["objective"]["gap_agg"].set_index("config_id")
    assert gap_agg.loc["A", "mean_gap"] == pytest.approx(0.0)
    assert gap_agg.loc["B", "mean_gap"] == pytest.approx(10.0)  # (20+0)/2


def test_compute_config_metrics_rank_agg():
    result = compute_config_metrics(DF_SMALL, "heuristic_ama_sgc")
    rank_agg = result["objective"]["rank_agg"].set_index("config_id")
    assert rank_agg.loc["A", "mean_rank"] == pytest.approx(1.0)


def test_compute_config_metrics_winrate():
    result = compute_config_metrics(DF_SMALL, "heuristic_ama_sgc")
    wr = result["objective"]["winrate"].set_index("config_id")
    assert wr.loc["A", "win_rate"] == pytest.approx(1.0)
    assert wr.loc["B", "win_rate"] == pytest.approx(0.5)
    assert wr.loc["C", "win_rate"] == pytest.approx(0.0)


# ── compute_cp_gap tests ─────────────────────────────────────────────────────


DF_HEURISTICS = pd.DataFrame({
    "module":          ["autostore_heuristic", "heuristic_ama_sgc"] * 2,
    "parameter":       ["orders"] * 4,
    "value":           [10, 10, 20, 20],
    "seed":            [0, 0, 0, 0],
    "instance_id":     ["orders_10_0", "orders_10_0", "orders_20_0", "orders_20_0"],
    "objective_value": [65.0, 58.0, 130.0, 115.0],
    "solve_time":      [0.5, 12.0, 0.6, 14.0],
    "feasible":        [True] * 4,
})

DF_CP = pd.DataFrame({
    "parameter":     ["orders", "orders"],
    "value":         [10, 20],
    "seed":          [0, 0],
    "instance_id":   ["orders_10_0", "orders_20_0"],
    "cp_objective":  [55.0, 110.0],
    "cp_status":     ["Optimal", "Optimal"],
    "cp_solve_time": [0.04, 0.05],
})


def test_compute_cp_gap_returns_three_dfs():
    raw, agg, by_param = compute_cp_gap(DF_HEURISTICS, DF_CP)
    assert isinstance(raw, pd.DataFrame)
    assert isinstance(agg, pd.DataFrame)
    assert isinstance(by_param, pd.DataFrame)


def test_compute_cp_gap_values():
    raw, _, _ = compute_cp_gap(DF_HEURISTICS, DF_CP)
    greedy_10 = raw[(raw["module"] == "autostore_heuristic") & (raw["value"] == 10)]
    assert greedy_10["cp_gap"].iloc[0] == pytest.approx(100.0 * 10 / 55, rel=1e-3)


def test_compute_cp_gap_agg_columns():
    _, agg, _ = compute_cp_gap(DF_HEURISTICS, DF_CP)
    for col in ["module", "mean_cp_gap", "median_cp_gap", "std_cp_gap"]:
        assert col in agg.columns


def test_compute_cp_gap_empty_on_no_match():
    df_cp_no_match = DF_CP.copy()
    df_cp_no_match["seed"] = 99
    raw, agg, by_param = compute_cp_gap(DF_HEURISTICS, df_cp_no_match)
    assert raw.empty
