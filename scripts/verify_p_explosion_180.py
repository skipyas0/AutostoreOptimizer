import sys
from order_station_assign.code.benchmarking.benchmark_v4_single import build_config
from datagen import generate_data
from cp_model import _old_generate_data

cfg = build_config("orders", 180, 42, 1800)

S, L, K, orders_req, rt, p, N = generate_data(
    num_stations=cfg["stations"], 
    lanes_per_station=cfg["lanes"], 
    num_orders=cfg["orders"], 
    num_skus=cfg["skus"], 
    seed=cfg["seed"],
    pick_touch_time=cfg["pick"]
)

U = {}
for o, reqs in orders_req.items():
    for k in reqs:
        U[k] = U.get(k, 0) + 1
        
total_p = sum(len(S) * count * count for count in U.values())
print(f"New Datagen (180 orders) -> Total P variables: {total_p}")
print(f"Max U[k]: {max(U.values())}")

S, L, K, orders_req_old, rt, p, N = _old_generate_data(
    num_stations=cfg["stations"], 
    lanes_per_station=cfg["lanes"], 
    num_orders=cfg["orders"], 
    num_skus=cfg["skus"], 
    seed=cfg["seed"],
    pick_touch_time=cfg["pick"],
    order_size_dist="poisson2_to_1_6"
)

U_old = {}
for o, reqs in orders_req_old.items():
    for k in reqs:
        U_old[k] = U_old.get(k, 0) + 1
        
total_p_old = sum(len(S) * count * count for count in U_old.values())
print(f"Old Datagen (180 orders) -> Total P variables: {total_p_old}")
print(f"Max U[k] old: {max(U_old.values())}")

