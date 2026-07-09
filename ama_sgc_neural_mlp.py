#!/usr/bin/env python3
import os
import json
import pickle
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datagen import generate_data
from instance import Instance
from autostore_heuristic import solve_heuristic_instance as base_solve
from heuristic_ama_sgc import precompute_attributes, run_sgc_parameterised, _compute_objective

from trainset_generator import ALL_CONFIGS, generate_and_precompute_dataset, AMAConfigDataset

MODEL_FILE = "ama_mlp_model.pt"

class AMAConfigClassifier(nn.Module):
    def __init__(self, input_dim=31, num_classes=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def train_mlp_agent(epochs=1000, batch_size=32, lr=1e-3):
    raw_data = generate_and_precompute_dataset()
    dataset = AMAConfigDataset(raw_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = AMAConfigClassifier(input_dim=31, num_classes=len(ALL_CONFIGS))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("Training MLP classifier...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
            
    torch.save(model.state_dict(), MODEL_FILE)
    print(f"Saved model to {MODEL_FILE}")

# Cache the loaded model
_global_model = None

def solve_heuristic_instance(config: dict, return_raw: bool = False):
    """
    Wrapper function matching other heuristic signatures.
    """
    global _global_model
    if _global_model is None:
        if not os.path.exists(MODEL_FILE):
            raise FileNotFoundError(f"Model {MODEL_FILE} not found. Train it first!")
        
        _global_model = AMAConfigClassifier(input_dim=31, num_classes=len(ALL_CONFIGS))
        _global_model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device('cpu')))
        _global_model.eval()
    
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
    features_tensor = torch.tensor(features).unsqueeze(0)  # Add batch dim
    
    with torch.no_grad():
        outputs = _global_model(features_tensor)
        action = torch.argmax(outputs, dim=1).item()
        
    oa, od, ba, bd = ALL_CONFIGS[action]
    
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
    train_mlp_agent(epochs=1000)
