#!/usr/bin/env python3
import argparse
import concurrent.futures
import os
import sys

import numpy as np
import pandas as pd

from autostore_heuristic import solve_heuristic_instance as base_solve
from datagen import generate_data
from heuristic_ama_sgc import (
    _compute_objective,
    precompute_attributes,
    run_sgc_parameterised,
)

ORDER_ATTRS = [
    "sum_rt",
    "order_size",
    "sum_cycle",
    "max_rt",
    "sku_rarity",
    "sku_contention",
    "sharing_degree",
    "min_copies",
]
BIN_ATTRS = ["rt", "p", "cycle", "demand_ratio", "copies", "demand"]
DIRECTIONS = [True, False]

# Pre-generate the 192 configs
ALL_CONFIGS = []
for oa in ORDER_ATTRS:
    for od in DIRECTIONS:
        for ba in BIN_ATTRS:
            for bd in DIRECTIONS:
                ALL_CONFIGS.append((oa, od, ba, bd))


def _process_combo(cfg):
    instance = generate_data(
        num_stations=cfg["stations"],
        lanes_per_station=cfg["lanes"],
        num_orders=cfg["orders"],
        num_skus=cfg["skus"],
        seed=cfg["seed"],
        pick_touch_time=cfg["pick"],
        movecap=cfg.get("movecap", max(10, int(cfg["skus"] / 1250))),
    )
    features = instance.get_features()

    # Baseline score
    res_base = base_solve({**cfg, "horizon": 10000}, return_raw=False)
    baseline_score = res_base.get("objective_value", float("inf"))
    if baseline_score is None:
        baseline_score = float("inf")

    # Evaluate all configurations
    order_attrs, sku_attrs = precompute_attributes(instance)
    config_scores = []
    for config in ALL_CONFIGS:
        oa, od, ba, bd = config
        sol = run_sgc_parameterised(
            instance,
            10000,
            cfg["movecap"],
            1.0,
            0.0,
            order_attrs,
            sku_attrs,
            oa,
            od,
            ba,
            bd,
        )
        obj = _compute_objective(sol, 1.0, 0.0, instance.S)
        config_scores.append(obj)

    return {
        "features": features,
        "baseline_score": baseline_score,
        "config_scores": config_scores,
        "config": cfg,
    }


def _regenerate_features(row: dict) -> dict:
    """Regenerate all feat_* columns for a cached row by recreating the instance.

    Uses the config columns (Config_Stations, Config_Lanes, etc.) to deterministically
    rebuild the Instance and compute fresh features. Returns the updated row dict.
    """
    import math

    def _safe_int(val, default=None):
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return default
        return int(val)

    cfg = {
        "stations": int(row["Config_Stations"]),
        "lanes": int(row["Config_Lanes"]),
        "orders": int(row["Config_Orders"]),
        "skus": int(row["Config_SKUs"]),
        "seed": int(row["Config_Seed"]),
        "pick": int(row["Config_Pick"]),
        # "movecap": _safe_int(row.get("Config_Movecap"))
        # or max(10, int(row["Config_SKUs"] / 1250)),
        "movecap": int(row.get("Config_Movecap")),
    }
    instance = generate_data(
        num_stations=cfg["stations"],
        lanes_per_station=cfg["lanes"],
        num_orders=cfg["orders"],
        num_skus=cfg["skus"],
        seed=cfg["seed"],
        pick_touch_time=cfg["pick"],
        movecap=cfg["movecap"],
    )
    new_features = instance.get_features()

    # Sanity check: verify that key features match between old and new
    for check_key in ["feat_rt_mean", "feat_p_mean", "feat_used_skus_ratio"]:
        old_val = row.get(check_key)
        new_val = new_features.get(check_key)
        if old_val is not None and new_val is not None:
            if abs(float(old_val) - float(new_val)) > 1e-9:
                print(
                    f"  WARNING: {check_key} mismatch: old={old_val:.6f}, new={new_val:.6f} "
                    f"(cfg={cfg})"
                )

    # Replace all feat_* columns
    for col in list(row.keys()):
        if col.startswith("feat_"):
            del row[col]
    row.update(new_features)
    return row


def generate_and_precompute_dataset(
    num_seeds: int = 15, cache_file: str = None, force_regenerate: bool = False
) -> pd.DataFrame:
    """
    Generate dataset and precompute features & configuration makespans under the parameterized AMA heuristic.
    Uses reference + param variation logic across all paradigms.
    """
    total_samples = 32 * num_seeds
    if cache_file is None:
        cache_file = f"ama_precomputed_data_{total_samples}.csv"

    if not force_regenerate and os.path.exists(cache_file):
        print(f"Loading precomputed dataset from {cache_file}")
        df = pd.read_csv(cache_file)
        rows = df.to_dict(orient="records")
        print(f"Recalculating features for {len(rows)} cached rows...")

        workers = os.cpu_count() - 1 or 4
        count = 0
        updated_rows = [None] * len(rows)
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(_regenerate_features, row): i
                for i, row in enumerate(rows)
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                count += 1
                try:
                    updated_rows[idx] = future.result()
                except Exception as e:
                    print(f"\nTask failed at row {idx}: {e}")
                    updated_rows[idx] = rows[idx]
                sys.stdout.write(f"\r  Recalculated features {count}/{len(rows)}")
                sys.stdout.flush()
        print()

        df = pd.DataFrame(updated_rows)
        df.to_csv(cache_file, index=False)
        print(f"Updated dataset saved to {cache_file}")
        return df

    dataset = []

    # Replicate benchmark configurations
    from constants import PARAM_LEVELS, PARAM_ORDER, REFERENCE_CONFIG

    # Seeds starting at 42
    SEEDS = list(range(42, 42 + num_seeds))

    combos = []
    for param in PARAM_ORDER:
        levels = PARAM_LEVELS[param]
        for val in levels:
            for seed in SEEDS:
                cfg = dict(REFERENCE_CONFIG)
                cfg["seed"] = seed

                if param == "stations":
                    cfg["stations"] = val
                    cfg["skus"] = val * 5000
                    cfg["movecap"] = max(10, int(cfg["skus"] / 1250))
                elif param == "lanes":
                    cfg["lanes"] = val
                elif param == "orders":
                    cfg["orders"] = val
                elif param == "movecap":
                    cfg["movecap"] = val

                combos.append(cfg)

    print(
        f"Generating and precomputing {len(combos)} samples using reference+param variation logic..."
    )

    workers = os.cpu_count() - 1 or 4
    print(f"Executing {len(combos)} tasks using {workers} workers...")

    count = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_cfg = {executor.submit(_process_combo, cfg): cfg for cfg in combos}

        for future in concurrent.futures.as_completed(future_to_cfg):
            count += 1
            try:
                res = future.result()
                dataset.append(res)
                sys.stdout.write(f"\rProcessed {count}/{len(combos)}")
                sys.stdout.flush()
            except Exception as e:
                print(f"\nTask failed: {e}")
    print()

    df = create_dataset_table(dataset, cache_file)
    return df


def create_dataset_table(dataset, output_file: str = None) -> pd.DataFrame:
    """
    Creates a comprehensive table (as a pandas DataFrame and optionally written to output_file)
    showing the instance config, its features, and the heuristic configuration chosen by the full AMA heuristic.
    """
    rows = []
    for i, data in enumerate(dataset):
        row = {}
        # Instance Config
        cfg = data.get("config", {})
        features = data.get("features", [])

        # # Fallback to reconstruct configuration from normalized features if config is missing
        # if not cfg and len(features) >= 4:
        #     cfg = {
        #         "stations": int(round(features["feat_num_stations"] * 20)),
        #         "lanes": int(round(features["feat_num_lanes"] * 20)),
        #         "orders": int(round(features["feat_num_orders"] * 1000)),
        #         "skus": int(round(features["feat_num_skus"] * 10000)),
        #         "pick": 4,  # default benchmark pick touch time
        #     }

        row["Sample_Index"] = i
        row["Config_Stations"] = cfg.get("stations", None)
        row["Config_Lanes"] = cfg.get("lanes", None)
        row["Config_Orders"] = cfg.get("orders", None)
        row["Config_SKUs"] = cfg.get("skus", None)
        row["Config_Movecap"] = cfg.get("movecap", None)
        row["Config_Seed"] = cfg.get("seed", None)
        row["Config_Pick"] = cfg.get("pick", None)

        # Baseline & best AMA config
        baseline = data.get("baseline_score", float("inf"))
        row["Baseline_Score"] = baseline

        scores = data.get("config_scores", [])
        if scores:
            best_idx = int(np.argmin(scores))
            best_score = scores[best_idx]
            best_config = ALL_CONFIGS[best_idx]
            row["AMA_Best_Score"] = best_score
            row["AMA_Config_Str"] = (
                f"{best_config[0]} ({best_config[1]}), {best_config[2]} ({best_config[3]})"
            )
            row["AMA_Order_Attr"] = best_config[0]
            row["AMA_Order_Dir"] = best_config[1]
            row["AMA_Bin_Attr"] = best_config[2]
            row["AMA_Bin_Dir"] = best_config[3]
            for idx, (oa, od, ba, bd) in enumerate(ALL_CONFIGS):
                col_name = f"AMA_{oa}_{od}_{ba}_{bd}_Score"
                row[col_name] = scores[idx] if idx < len(scores) else None
        else:
            row["AMA_Best_Score"] = None
            row["AMA_Config_Str"] = None
            row["AMA_Order_Attr"] = None
            row["AMA_Order_Dir"] = None
            row["AMA_Bin_Attr"] = None
            row["AMA_Bin_Dir"] = None
            for oa, od, ba, bd in ALL_CONFIGS:
                col_name = f"AMA_{oa}_{od}_{ba}_{bd}_Score"
                row[col_name] = None

        # Features
        features = data.get("features", {})
        if isinstance(features, dict):
            for feat_name, feat_val in features.items():
                row[feat_name] = feat_val
        else:
            raise ValueError

        rows.append(row)

    df = pd.DataFrame(rows)
    if output_file:
        if not output_file.endswith(".csv"):
            output_file = output_file + ".csv"
            print("Only supports .csv, renaming to", output_file)

        df.to_csv(output_file, index=False)
        print(f"Table saved to CSV: {output_file}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AutoStore Trainset Generator & Data Table Creator"
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=15,
        help="Number of seeds to vary configurations over",
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Force regeneration of the dataset even if cache exists",
    )
    parser.add_argument(
        "--cache-file", type=str, default=None, help="Path to cache file"
    )

    args = parser.parse_args()

    df = generate_and_precompute_dataset(
        num_seeds=args.num_seeds,
        cache_file=args.cache_file,
        force_regenerate=args.force_regenerate,
    )

    # if args.output_table != args.cache_file:
    #     if args.output_table.endswith(".csv"):
    #         df.to_csv(args.output_table, index=False)
    #     else:
    #         with open(args.output_table, "w") as f:
    #             f.write(df.to_string(index=False))
    #         print(f"Table saved to {args.output_table}")
