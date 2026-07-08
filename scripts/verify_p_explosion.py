import sys
from order_station_assign.code.benchmarking.benchmark_v4_single import build_config
from datagen import generate_data

cfg = build_config("orders", 40, 42, 1800)

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

print("Top SKUs by request count:")
for k, count in sorted(U.items(), key=lambda x: -x[1])[:5]:
    P_vars_for_k = len(S) * count * count
    print(f"SKU {k}: Requested {count} times -> creates {P_vars_for_k} P variables")
    
total_p = sum(len(S) * count * count for count in U.values())
print(f"\nTotal P variables in new datagen: {total_p}")

