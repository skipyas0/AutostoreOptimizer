"""data_loader.py — Load heuristic benchmark JSON files into pandas DataFrames."""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import pandas as pd
HEURISTIC_DIR = "results/heuristic_local/full_final"
CP_DIR = "logs/v6_CPGWS_2_newdatagen"


def _make_config_id(order_attr: str, order_desc: bool,
                    bin_attr: str, bin_desc: bool) -> str:
    """Format config as label, e.g. 'max_rt↓ / demand_ratio↑'."""
    od = "↓" if order_desc else "↑"
    bd = "↓" if bin_desc else "↑"
    return f"{order_attr}{od} / {bin_attr}{bd}"


def _make_instance_id(parameter: str, value: int | float, seed: int) -> str:
    return f"{parameter}_{int(value)}_{seed}"


def load_cp(base_dir: Path) -> pd.DataFrame:
    """Load CP reference results from warmstart JSON files.

    Returns DataFrame: parameter, value, seed, instance_id,
    cp_objective, cp_status, cp_solve_time.
    Missing files are skipped with a console warning.
    """
    pattern = str(Path(base_dir) / CP_DIR / "warmstart_*_seed*.json")
    files = glob.glob(pattern)

    records = []
    for fpath in sorted(files):
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
            meta = data["meta"]
            result = data["result"]
            param = meta["parameter"]
            value = int(meta["value"])
            seed = int(meta["seed"])
            records.append({
                "parameter": param,
                "value": value,
                "seed": seed,
                "instance_id": _make_instance_id(param, value, seed),
                "cp_objective": result.get("objective_value"),
                "cp_status": result.get("status"),
                "cp_solve_time": result.get("solve_time"),
            })
        except Exception as e:
            print(f"[WARNING] Skipping CP file {fpath}: {e}")

    if not records:
        print(f"[WARNING] No CP reference files found at {pattern}")
        return pd.DataFrame(columns=[
            "parameter", "value", "seed", "instance_id",
            "cp_objective", "cp_status", "cp_solve_time",
        ])

    df = pd.DataFrame(records)
    df["value"] = pd.to_numeric(df["value"])
    df["seed"] = pd.to_numeric(df["seed"])
    df["cp_objective"] = pd.to_numeric(df["cp_objective"], errors="coerce")
    print(f"  Loaded {len(df)} CP reference records.")
    return df


def load_heuristics(base_dir: Path) -> pd.DataFrame:
    """Load winner-level results from all heuristic modules.

    Returns DataFrame: module, parameter, value, seed, instance_id,
    objective_value, solve_time, feasible.
    """
    records = []
    heuristic_dir = Path(base_dir) / HEURISTIC_DIR
    all_json = glob.glob(str(heuristic_dir / "*.json"))

    for fpath in sorted(all_json):
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
            
            meta = data.get("meta", {})
            result = data.get("result", {})
            module = meta.get("module", "unknown_heuristic")
            
            param = meta["parameter"]
            value = int(meta["value"])
            seed = int(meta["seed"])
            records.append({
                "module": module,
                "parameter": param,
                "value": value,
                "seed": seed,
                "instance_id": _make_instance_id(param, value, seed),
                "objective_value": result.get("objective_value"),
                "solve_time": result.get("solve_time"),
                "feasible": result.get("status") in ("Feasible", "Optimal"),
            })
        except Exception as e:
            print(f"[WARNING] Skipping {fpath}: {e}")

    if not records:
        return pd.DataFrame(columns=[
            "module", "parameter", "value", "seed", "instance_id",
            "objective_value", "solve_time", "feasible",
        ])

    df = pd.DataFrame(records)
    df["value"] = pd.to_numeric(df["value"])
    df["seed"] = pd.to_numeric(df["seed"])
    df["objective_value"] = pd.to_numeric(df["objective_value"], errors="coerce")
    df["solve_time"] = pd.to_numeric(df["solve_time"], errors="coerce")
    for mod, n in df.groupby("module").size().items():
        print(f"  Loaded {n} records for {mod}.")
    return df


def load_configs(base_dir: Path) -> pd.DataFrame:
    """Load per-config all_runs from AMA-SGC full-mode and 2-phase files.

    Returns DataFrame: module, parameter, value, seed, instance_id, phase,
    order_attr, order_desc, bin_attr, bin_desc, config_id, feasible,
    objective, total_moves.
    Files without all_runs are skipped with a console warning.
    """
    records = []
    heuristic_dir = Path(base_dir) / HEURISTIC_DIR
    all_json = glob.glob(str(heuristic_dir / "*.json"))

    skipped = 0
    for fpath in sorted(all_json):
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
            
            result = data.get("result", {})
            all_runs = result.get("all_runs")
            if not all_runs:
                skipped += 1
                continue
                
            meta = data.get("meta", {})
            module = meta.get("module", "unknown_heuristic")
            
            param = meta["parameter"]
            value = int(meta["value"])
            seed = int(meta["seed"])
            instance_id = _make_instance_id(param, value, seed)
            
            for run in all_runs:
                records.append({
                    "module": module,
                    "parameter": param,
                    "value": value,
                    "seed": seed,
                    "instance_id": instance_id,
                    "phase": run.get("phase"),
                    "order_attr": run.get("order_attr"),
                    "order_desc": bool(run.get("order_desc", False)),
                    "bin_attr": run.get("bin_attr"),
                    "bin_desc": bool(run.get("bin_desc", False)),
                    "config_id": _make_config_id(
                        run.get("order_attr", ""), bool(run.get("order_desc", False)),
                        run.get("bin_attr", ""), bool(run.get("bin_desc", False))
                    ),
                    "feasible": bool(run.get("feasible", True)),
                    "objective": run.get("objective"),
                    "total_moves": run.get("total_moves"),
                })
        except Exception as e:
            print(f"[WARNING] Skipping {fpath}: {e}")

    if skipped:
        print(f"[WARNING] {skipped} files had no all_runs — skipped.")

    if not records:
        return pd.DataFrame(columns=[
            "module", "parameter", "value", "seed", "instance_id", "phase",
            "order_attr", "order_desc", "bin_attr", "bin_desc", "config_id",
            "feasible", "objective", "total_moves",
        ])

    df = pd.DataFrame(records)
    df["value"] = pd.to_numeric(df["value"])
    df["seed"] = pd.to_numeric(df["seed"])
    df["objective"] = pd.to_numeric(df["objective"], errors="coerce")
    df["total_moves"] = pd.to_numeric(df["total_moves"], errors="coerce")
    for mod, n in df.groupby("module").size().items():
        print(f"  Loaded {n} config-run records for {mod}.")
    return df


def load_all(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all data. Returns (df_configs, df_heuristics, df_cp)."""
    base_dir = Path(base_dir)
    print("=" * 60)
    print("Loading data...")
    df_cp = load_cp(base_dir)
    df_heuristics = load_heuristics(base_dir)
    df_configs = load_configs(base_dir)
    print("=" * 60)
    return df_configs, df_heuristics, df_cp
