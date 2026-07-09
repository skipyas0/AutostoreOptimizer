#!/usr/bin/env python3
import os
import json
import pickle
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datagen import generate_data
from instance import Instance
from autostore_heuristic import solve_heuristic_instance as base_solve
from heuristic_ama_sgc import precompute_attributes, run_sgc_parameterised, _compute_objective

from trainset_generator import ALL_CONFIGS, generate_and_precompute_dataset, AMASGCEnv

MODEL_FILE = "ama_rl_model"

def train_rl_agent(num_seeds=15, total_timesteps=50000):
    dataset = generate_and_precompute_dataset(num_seeds=num_seeds)
    env = AMASGCEnv(dataset)
    
    model = PPO("MlpPolicy", env, verbose=1, policy_kwargs={"net_arch": [64, 64]})
    print("Training RL agent...")
    model.learn(total_timesteps=total_timesteps)
    model.save(MODEL_FILE)
    print(f"Saved model to {MODEL_FILE}")

# Cache the loaded model to avoid reloading on every call
_global_model = None

def solve_heuristic_instance(config: dict, return_raw: bool = False):
    """
    Wrapper function matching other heuristic signatures.
    """
    global _global_model
    if _global_model is None:
        if not os.path.exists(MODEL_FILE + ".zip"):
            raise FileNotFoundError(f"Model {MODEL_FILE}.zip not found. Train it first!")
        _global_model = PPO.load(MODEL_FILE)
    
    # Generate the instance from config to get features
    num_stations = config.get("stations", 1)
    lanes_per_station = config.get("lanes", 2)
    num_orders = config.get("orders", 10)
    num_skus = config.get("skus", 20)
    seed = config.get("seed", 42)
    pick_touch_time = config.get("pick", 4)
    horizon = config.get("horizon", 10000)
    move_cap = config.get("movecap", None)
    alpha = config.get("alpha", 1.0)
    beta = config.get("beta", 0.0)

    t0 = time.perf_counter()
    instance = generate_data(
        num_stations=num_stations,
        lanes_per_station=lanes_per_station,
        num_orders=num_orders,
        num_skus=num_skus,
        seed=seed,
        pick_touch_time=pick_touch_time
    )
    
    features = np.array(instance.get_features(), dtype=np.float32)
    action, _states = _global_model.predict(features, deterministic=True)
    
    oa, od, ba, bd = ALL_CONFIGS[int(action)]
    
    # TODO: don't precompute everything, only the chosen config
    order_attrs, sku_attrs = precompute_attributes(instance)
    sol = run_sgc_parameterised(
        instance,
        horizon,
        move_cap,
        alpha, beta,
        order_attrs, sku_attrs,
        oa, od, ba, bd
    )
    elapsed = time.perf_counter() - t0
    
    res = {
        "status": "Feasible" if sol.feasible else "Infeasible",
        "solve_time": elapsed,
        "objective_value": float(sol.makespan) if sol.feasible else None,
        "num_vars": 0,
        "progress": [],
        "total_moves": sol.total_moves if sol.feasible else None,
        "winning_config": (oa, od, ba, bd)
    }
    return (res, sol) if return_raw else res

if __name__ == "__main__":
    train_rl_agent(num_seeds=15, total_timesteps=50000)
