#!/usr/bin/env python3
import os
import pickle
import time
import argparse
import sys
import concurrent.futures
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import torch
from torch.utils.data import Dataset

from datagen import generate_data
from instance import Instance
from autostore_heuristic import solve_heuristic_instance as base_solve
from heuristic_ama_sgc import precompute_attributes, run_sgc_parameterised, _compute_objective

ORDER_ATTRS = [
    "sum_rt", "order_size", "sum_cycle", "max_rt",
    "sku_rarity", "sku_contention", "sharing_degree", "min_copies",
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
        pick_touch_time=cfg["pick"]
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
            instance, 10000, cfg["movecap"], 1.0, 0.0,
            order_attrs, sku_attrs, oa, od, ba, bd
        )
        obj = _compute_objective(sol, 1.0, 0.0, instance.S)
        config_scores.append(obj)
    
    return {
        "features": features,
        "baseline_score": baseline_score,
        "config_scores": config_scores,
        "config": cfg
    }

def generate_and_precompute_dataset(
    num_seeds: int = 15,
    cache_file: str = None,
    force_regenerate: bool = False
):
    """
    Generate dataset and precompute features & configuration makespans under the parameterized AMA heuristic.
    Uses reference + param variation logic across all paradigms.
    """
    total_samples = 32 * num_seeds
    if cache_file is None:
        cache_file = f"ama_precomputed_data_{total_samples}.pkl"

    if not force_regenerate and os.path.exists(cache_file):
        print(f"Loading precomputed dataset from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    dataset = []

    # Replicate benchmark configurations
    REFERENCE_CONFIG = {
        "stations": 4,
        "lanes": 4,
        "orders": 40,
        "pick": 4,
    }
    REFERENCE_CONFIG["skus"] = REFERENCE_CONFIG["stations"] * 5000
    REFERENCE_CONFIG["movecap"] = REFERENCE_CONFIG["skus"] // 1000

    PARAM_LEVELS = {
        "stations": [1, 2, 4, 6, 8, 10],
        "lanes": [1, 2, 4, 6, 8, 10],
        "orders": [10, 20, 40, 60, 80, 90, 100, 120, 140, 160, 180, 200],
        "movecap": [1, 2, 5, 10, 15, 20, 40, 60],
    }
    PARAM_ORDER = ["stations", "lanes", "orders", "movecap"]
    
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

    print(f"Generating and precomputing {len(combos)} samples using reference+param variation logic...")
    
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

    with open(cache_file, "wb") as f:
        pickle.dump(dataset, f)
    
    return dataset

class AMAConfigDataset(Dataset):
    def __init__(self, raw_data):
        self.X = []
        self.y = []
        for data in raw_data:
            features = data["features"]
            scores = data["config_scores"]
            
            # Label is the index of the best configuration (minimum makespan)
            best_idx = int(np.argmin(scores))
            
            self.X.append(features)
            self.y.append(best_idx)
            
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class AMASGCEnv(gym.Env):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.action_space = spaces.Discrete(len(ALL_CONFIGS))
        # 31 features from instance.get_features()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(31,), dtype=np.float32)
        self.current_idx = 0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx = np.random.randint(len(self.dataset))
        data = self.dataset[self.current_idx]
        return np.array(data["features"], dtype=np.float32), {}
    
    def step(self, action):
        data = self.dataset[self.current_idx]
        baseline = data["baseline_score"]
        config_score = data["config_scores"][action]
        
        # Reward based on relative improvement to baseline
        # We want to maximize this: if config_score < baseline, reward > 0
        if baseline == float("inf") and config_score == float("inf"):
            reward = 0.0
        elif baseline == float("inf"):
            reward = 1.0
        elif config_score == float("inf"):
            reward = -1.0
        else:
            reward = (baseline - config_score) / baseline
        
        # Terminate immediately (contextual bandit formulation)
        return np.array(data["features"], dtype=np.float32), float(reward), True, False, {}

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
        
        # Fallback to reconstruct configuration from normalized features if config is missing
        if not cfg and len(features) >= 4:
            cfg = {
                "stations": int(round(features[0] * 20)),
                "lanes": int(round(features[1] * 20)),
                "orders": int(round(features[2] * 1000)),
                "skus": int(round(features[3] * 10000)),
                "pick": 4, # default benchmark pick touch time
            }

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
            row["AMA_Config_Str"] = f"{best_config[0]} ({best_config[1]}), {best_config[2]} ({best_config[3]})"
            row["AMA_Order_Attr"] = best_config[0]
            row["AMA_Order_Dir"] = best_config[1]
            row["AMA_Bin_Attr"] = best_config[2]
            row["AMA_Bin_Dir"] = best_config[3]
        else:
            row["AMA_Best_Score"] = None
            row["AMA_Config_Str"] = None
            row["AMA_Order_Attr"] = None
            row["AMA_Order_Dir"] = None
            row["AMA_Bin_Attr"] = None
            row["AMA_Bin_Dir"] = None

        # Features (normalized list of 31 floats)
        features = data.get("features", [])
        feature_names = [
            "feat_num_stations", "feat_num_lanes", "feat_num_orders", "feat_num_skus",
            "feat_used_skus_ratio", "feat_order_size_min", "feat_order_size_max",
            "feat_order_size_mean", "feat_order_size_median", "feat_order_size_stdev",
            "feat_rt_min", "feat_rt_max", "feat_rt_mean", "feat_rt_stdev",
            "feat_p_min", "feat_p_max", "feat_p_mean", "feat_p_stdev",
            "feat_n_min", "feat_n_max", "feat_n_mean", "feat_n_stdev",
            "feat_sku_freq_min", "feat_sku_freq_max", "feat_sku_freq_mean", "feat_sku_freq_stdev",
            "feat_pareto_ratio", "feat_avg_jaccard", "feat_total_pick_lines",
            "feat_supply_demand_ratio", "feat_supply_demand_ratio_stdev"
        ]
        for f_idx, f_val in enumerate(features):
            if f_idx < len(feature_names):
                row[feature_names[f_idx]] = f_val
            else:
                row[f"feat_unmapped_{f_idx}"] = f_val

        rows.append(row)

    df = pd.DataFrame(rows)
    if output_file:
        if output_file.endswith(".csv"):
            df.to_csv(output_file, index=False)
            print(f"Table saved to CSV: {output_file}")
        elif output_file.endswith(".md"):
            with open(output_file, "w") as f:
                f.write(df.to_markdown(index=False))
            print(f"Table saved to Markdown: {output_file}")
        elif output_file.endswith(".html"):
            df.to_html(output_file, index=False)
            print(f"Table saved to HTML: {output_file}")
        else:
            with open(output_file, "w") as f:
                f.write(df.to_string(index=False))
            print(f"Table saved to Text: {output_file}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoStore Trainset Generator & Data Table Creator")
    parser.add_argument("--num-seeds", type=int, default=15, help="Number of seeds to vary configurations over")
    parser.add_argument("--force-regenerate", action="store_true", help="Force regeneration of the dataset even if cache exists")
    parser.add_argument("--cache-file", type=str, default=None, help="Path to cache file")
    parser.add_argument("--output-table", type=str, default=None, help="Path to save the generated table (e.g. dataset_info.md, dataset_info.csv)")
    
    args = parser.parse_args()
    
    dataset = generate_and_precompute_dataset(
        num_seeds=args.num_seeds,
        cache_file=args.cache_file,
        force_regenerate=args.force_regenerate
    )
    
    if args.output_table:
        create_dataset_table(dataset, args.output_table)
    else:
        df = create_dataset_table(dataset)
        print("\nDataset Table Preview (First 5 Rows):")
        print(df.head().to_string())
