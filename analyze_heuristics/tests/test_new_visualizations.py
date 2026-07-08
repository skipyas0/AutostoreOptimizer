"""Tests for new visualization functions."""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))
from analyze_heuristics import metrics


def test_compute_parameter_scaling_empty():
    """Should return empty DataFrames when no data is present."""
    df_h = pd.DataFrame()
    df_cp = pd.DataFrame()
    merged, agg = metrics.compute_parameter_scaling(df_h, df_cp)
    assert merged.empty
    assert agg.empty


def test_compute_parameter_scaling_basic():
    """Should compute ratios and gaps correctly for basic data."""
    df_h = pd.DataFrame({
        "module": ["mod_a", "mod_a", "mod_a", "mod_b", "mod_b", "mod_b"],
        "parameter": ["orders", "orders", "orders", "orders", "orders", "orders"],
        "value": [10, 10, 20, 10, 10, 20],
        "seed": [0, 1, 0, 0, 1, 0],
        "objective_value": [100, 110, 200, 105, 115, 205],
        "solve_time": [1, 1.1, 2, 1.1, 1.2, 2.2],
    })
    df_cp = pd.DataFrame({
        "parameter": ["orders", "orders"],
        "value": [10, 20],
        "seed": [0, 0],
        "cp_objective": [90, 180],
    })
    merged, agg = metrics.compute_parameter_scaling(df_h, df_cp)

    assert not merged.empty
    assert not agg.empty

    # Check ratio for mod_a, value=10, seed=0
    row = merged[(merged["module"] == "mod_a") & (merged["value"] == 10)].iloc[0]
    expected_ratio = 100 / 90
    assert row["ratio"] == pytest.approx(expected_ratio)

    # Check cp_gap
    expected_gap = (100 - 90) / 90 * 100
    assert row["cp_gap"] == pytest.approx(expected_gap)

    # Check aggregated values: 2 modules × 2 values = 4 rows
    assert len(agg) == 4
    assert "obj_median" in agg.columns
    assert "time_median" in agg.columns


def test_compute_parameter_scaling_missing_cp():
    """Should return empty if CP has no matching rows."""
    df_h = pd.DataFrame({
        "module": ["mod_a"],
        "parameter": ["orders"],
        "value": [10],
        "seed": [0],
        "objective_value": [100],
        "solve_time": [1],
    })
    df_cp = pd.DataFrame({
        "parameter": ["stations"],
        "value": [1],
        "seed": [0],
        "cp_objective": [90],
    })
    merged, agg = metrics.compute_parameter_scaling(df_h, df_cp)
    assert merged.empty
    assert agg.empty
