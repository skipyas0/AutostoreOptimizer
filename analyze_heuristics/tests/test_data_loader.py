"""Tests for data_loader module."""
import json
from pathlib import Path
import pandas as pd
import pytest
import sys
sys.path.insert(0, str(Path(__file__).parents[2]))

from analyze_heuristics.data_loader import load_cp

CP_FILE_CONTENT = {
    "meta": {
        "mode": "warmstart",
        "heuristic_module": "autostore_heuristic",
        "model_version": "v5",
        "parameter": "orders",
        "value": 10,
        "seed": 0,
    },
    "config": {},
    "result": {
        "status": "Optimal",
        "solve_time": 0.04,
        "objective_value": 55,
    },
    "progress": [],
    "heuristic": {"status": "Feasible", "solve_time": 0.05, "objective_value": 59.0},
}


@pytest.fixture
def tmp_base(tmp_path):
    cp_dir = tmp_path / "logs" / "v6_CPGWS_2_newdatagen"
    cp_dir.mkdir(parents=True)
    (cp_dir / "warmstart_orders_val10_seed0.json").write_text(json.dumps(CP_FILE_CONTENT))
    return tmp_path


def test_load_cp_returns_dataframe(tmp_base):
    df = load_cp(tmp_base)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1


def test_load_cp_columns(tmp_base):
    df = load_cp(tmp_base)
    for col in ["parameter", "value", "seed", "instance_id",
                "cp_objective", "cp_status", "cp_solve_time"]:
        assert col in df.columns, f"Missing column: {col}"


def test_load_cp_values(tmp_base):
    df = load_cp(tmp_base)
    row = df.iloc[0]
    assert row["parameter"] == "orders"
    assert row["value"] == 10
    assert row["seed"] == 0
    assert row["instance_id"] == "orders_10_0"
    assert row["cp_objective"] == 55
    assert row["cp_status"] == "Optimal"


def test_load_cp_empty_on_missing_dir(tmp_path):
    df = load_cp(tmp_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


# ── load_heuristics tests ────────────────────────────────────────────────────

from analyze_heuristics.data_loader import load_heuristics

GREEDY_FILE_CONTENT = {
    "meta": {"mode": "heuristic_local", "module": "autostore_heuristic",
             "parameter": "orders", "value": 10, "seed": 0},
    "config": {},
    "result": {"status": "Feasible", "solve_time": 0.5, "objective_value": 65.0},
}

AMA_SGC_FILE_CONTENT = {
    "meta": {"mode": "heuristic_local", "module": "heuristic_ama_sgc",
             "parameter": "orders", "value": 10, "seed": 0},
    "config": {},
    "result": {"status": "Feasible", "solve_time": 12.3,
               "objective_value": 58.0, "total_moves": 45,
               "winning_config": ["max_rt", False, "demand_ratio", True]},
}

AMA_2PHASE_FILE_CONTENT = {
    "meta": {"mode": "heuristic_local", "module": "heuristic_ama_sgc_2phase",
             "parameter": "orders", "value": 10, "seed": 0},
    "config": {},
    "result": {"status": "Feasible", "solve_time": 3.1,
               "objective_value": 60.0, "total_moves": 48,
               "winning_config": ["max_rt", False, "demand_ratio", True],
               "all_runs": []},
}


@pytest.fixture
def tmp_base_heuristics(tmp_path):
    h_dir = tmp_path / "results" / "heuristic_local" / "full_final"
    h_dir.mkdir(parents=True)
    (h_dir / "autostore_heuristic_orders_val10_seed0.json").write_text(
        json.dumps(GREEDY_FILE_CONTENT))
    (h_dir / "heuristic_ama_sgc_orders_val10_seed0.json").write_text(
        json.dumps(AMA_SGC_FILE_CONTENT))
    (h_dir / "heuristic_ama_sgc_2phase_orders_val10_seed0.json").write_text(
        json.dumps(AMA_2PHASE_FILE_CONTENT))
    return tmp_path


def test_load_heuristics_count(tmp_base_heuristics):
    df = load_heuristics(tmp_base_heuristics)
    assert len(df) == 3


def test_load_heuristics_columns(tmp_base_heuristics):
    df = load_heuristics(tmp_base_heuristics)
    for col in ["module", "parameter", "value", "seed", "instance_id",
                "objective_value", "solve_time", "feasible"]:
        assert col in df.columns, f"Missing: {col}"


def test_load_heuristics_modules(tmp_base_heuristics):
    df = load_heuristics(tmp_base_heuristics)
    modules = set(df["module"].unique())
    assert "autostore_heuristic" in modules
    assert "heuristic_ama_sgc" in modules
    assert "heuristic_ama_sgc_2phase" in modules


def test_load_heuristics_no_2phase_in_full_mode(tmp_base_heuristics):
    df = load_heuristics(tmp_base_heuristics)
    full_mode = df[df["module"] == "heuristic_ama_sgc"]
    assert len(full_mode) == 1
    assert full_mode.iloc[0]["objective_value"] == 58.0


# ── load_configs + load_all tests ────────────────────────────────────────────

from analyze_heuristics.data_loader import load_configs, load_all

AMA_SGC_WITH_RUNS = {
    "meta": {"mode": "heuristic_local", "module": "heuristic_ama_sgc",
             "parameter": "movecap", "value": 10, "seed": 0},
    "config": {},
    "result": {
        "status": "Feasible", "solve_time": 5.0,
        "objective_value": 100.0, "total_moves": 50,
        "winning_config": ["max_rt", False, "demand_ratio", True],
        "all_runs": [
            {"phase": "full_grid", "order_attr": "max_rt", "order_desc": True,
             "bin_attr": "rt", "bin_desc": True, "feasible": True,
             "makespan": 120.0, "total_moves": 60, "objective": 120.0},
            {"phase": "full_grid", "order_attr": "max_rt", "order_desc": False,
             "bin_attr": "demand_ratio", "bin_desc": True, "feasible": True,
             "makespan": 100.0, "total_moves": 50, "objective": 100.0},
        ],
    },
}

AMA_SGC_NO_RUNS = {
    "meta": {"mode": "heuristic_local", "module": "heuristic_ama_sgc",
             "parameter": "stations", "value": 1, "seed": 1},
    "config": {},
    "result": {"status": "Feasible", "solve_time": 10.0,
               "objective_value": 800.0, "total_moves": 100,
               "winning_config": ["max_rt", False, "demand", False]},
}


@pytest.fixture
def tmp_base_configs(tmp_path):
    h_dir = tmp_path / "results" / "heuristic_local" / "full_final"
    h_dir.mkdir(parents=True)
    (h_dir / "heuristic_ama_sgc_movecap_val10_seed0.json").write_text(
        json.dumps(AMA_SGC_WITH_RUNS))
    (h_dir / "heuristic_ama_sgc_stations_val1_seed1.json").write_text(
        json.dumps(AMA_SGC_NO_RUNS))
    return tmp_path


def test_load_configs_full_mode_runs(tmp_base_configs):
    df = load_configs(tmp_base_configs)
    full = df[df["module"] == "heuristic_ama_sgc"]
    assert len(full) == 2


def test_load_configs_skips_no_all_runs(tmp_base_configs):
    df = load_configs(tmp_base_configs)
    stations = df[(df["module"] == "heuristic_ama_sgc") & (df["parameter"] == "stations")]
    assert len(stations) == 0


def test_load_configs_config_id_format(tmp_base_configs):
    df = load_configs(tmp_base_configs)
    row = df[(df["order_attr"] == "max_rt") & (df["order_desc"] == True) &
             (df["bin_attr"] == "rt") & (df["bin_desc"] == True)].iloc[0]
    assert row["config_id"] == "max_rt↓ / rt↓"


def test_load_configs_columns(tmp_base_configs):
    df = load_configs(tmp_base_configs)
    for col in ["module", "parameter", "value", "seed", "instance_id", "phase",
                "order_attr", "order_desc", "bin_attr", "bin_desc", "config_id",
                "feasible", "objective", "total_moves"]:
        assert col in df.columns, f"Missing: {col}"


def test_load_all_returns_three_dataframes(tmp_base_configs):
    cp_dir = tmp_base_configs / "logs" / "v6_CPGWS_2_newdatagen"
    cp_dir.mkdir(parents=True)
    (cp_dir / "warmstart_movecap_val10_seed0.json").write_text(json.dumps(CP_FILE_CONTENT))
    df_configs, df_heuristics, df_cp = load_all(tmp_base_configs)
    assert isinstance(df_configs, pd.DataFrame)
    assert isinstance(df_heuristics, pd.DataFrame)
    assert isinstance(df_cp, pd.DataFrame)
