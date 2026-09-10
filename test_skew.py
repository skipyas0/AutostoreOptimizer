import numpy as np

from datagen import generate_data


def theoretical_catalogue_pareto_20(num_skus: int = 20000, skew: float = 1.0) -> float:
    ranks = np.arange(1, num_skus + 1)
    weights = 1.0 / (ranks**skew)
    weights /= weights.sum()

    top_20_count = int(0.20 * num_skus)
    return float(weights[:top_20_count].sum())


skew_vals = [0.9, 1.0, 1.1, 1.15, 1.20, 1.25, 1.30]
for num_skus in [5000, 10000, 15000, 20000, 25000, 30000]:
    ratios_theory = []
    ratios_empirical = []
    for skew in skew_vals:
        ratios_theory.append(theoretical_catalogue_pareto_20(num_skus, skew))
        inst = generate_data(
            num_stations=3,
            lanes_per_station=4,
            num_orders=300,
            num_skus=num_skus,
            seed=42,
            sku_popularity_skew=skew,
            movecap=20,
        )
        ratios_empirical.append(inst.stats["pareto_ratios"][20])
    print(
        f"SKUS {num_skus:5d}: | ",
        *[
            f"skew {s}: theory {100 * t:.02f}% empirical {100 * e:.02f}%"
            for s, t, e in zip(skew_vals, ratios_theory, ratios_empirical)
        ],
    )
