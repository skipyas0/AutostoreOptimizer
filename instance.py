from typing import Dict, List, Tuple, Any

class Instance:
    """Represents an order scheduling instance for a robotic compact storage system (AutoStore)."""
    def __init__(
        self,
        S: List[int],
        L: List[int],
        K: List[int],
        orders_requirements: Dict[int, List[int]],
        rt: Dict[int, int],
        p: Dict[int, int],
        N: Dict[int, int],
        rt_ret: Dict[int, int] = None,
    ):
        self.S = S
        self.L = L
        self.K = K
        self._orders_requirements = orders_requirements
        self.rt = rt
        self.p = p
        self.N = N
        self._rt_ret = rt_ret if rt_ret is not None else dict(rt)

    @property
    def orders_requirements(self) -> Dict[int, List[int]]:
        return self._orders_requirements

    @orders_requirements.setter
    def orders_requirements(self, value: Dict[int, List[int]]):
        self._orders_requirements = value

    @property
    def orders_req(self) -> Dict[int, List[int]]:
        return self._orders_requirements

    @orders_req.setter
    def orders_req(self, value: Dict[int, List[int]]):
        self._orders_requirements = value

    @property
    def O(self) -> List[int]:
        """Sorted list of order IDs."""
        return sorted(self._orders_requirements.keys())

    @property
    def rt_ret(self) -> Dict[int, int]:
        """A dictionary mapping SKU to its return time, defaulting to its retrieval time."""
        return self._rt_ret

    @rt_ret.setter
    def rt_ret(self, value: Dict[int, int]):
        self._rt_ret = value

    def get_statistics(self) -> Dict[str, Any]:
        """Calculate and return key statistics about the instance."""
        import itertools as _itertools

        num_orders = len(self._orders_requirements)
        order_sizes = [len(v) for v in self._orders_requirements.values()]
        rt_vals = list(self.rt.values())
        p_vals = list(self.p.values())
        n_vals = list(self.N.values())

        # SKU frequency across orders
        sku_freq: Dict[int, int] = {}
        for reqs in self._orders_requirements.values():
            for k in reqs:
                sku_freq[k] = sku_freq.get(k, 0) + 1
        used_skus = [k for k in self.K if sku_freq.get(k, 0) > 0]
        unused_skus = len(self.K) - len(used_skus)

        # Pareto check: what fraction of picks come from top 20% of used SKUs
        if used_skus:
            sorted_by_freq = sorted(used_skus, key=lambda k: sku_freq[k], reverse=True)
            top20_count = max(1, len(sorted_by_freq) // 5)
            top20_picks = sum(sku_freq[k] for k in sorted_by_freq[:top20_count])
            total_picks = sum(sku_freq.values())
            pareto_ratio = top20_picks / total_picks if total_picks else 0
        else:
            pareto_ratio = 0.0

        # SKU overlap: average Jaccard similarity between order pairs (sample first 50 orders)
        order_keys = list(self._orders_requirements.keys())
        pairs = list(_itertools.combinations(order_keys[:50], 2))
        if pairs:
            jaccards = []
            for i, j in pairs:
                si = set(self._orders_requirements[i])
                sj = set(self._orders_requirements[j])
                inter = len(si & sj)
                union = len(si | sj)
                jaccards.append(inter / union if union else 0)
            avg_jaccard = sum(jaccards) / len(jaccards)
        else:
            avg_jaccard = 0.0

        # Supply (number of bins per SKU) and demand (number of orders per SKU)
        supply = [n_vals[k] for k in used_skus]
        demand = [sku_freq[k] for k in used_skus]
        supply_demand_ratio = sum(supply) / sum(demand) if demand else 0
        supply_demand_ratio_stdev = sum((s - d)**2 for s, d in zip(supply, demand)) / len(supply) if supply else 0

        return {
            "num_stations": len(self.S),
            "num_lanes": len(self.L),
            "num_orders": num_orders,
            "num_skus": len(self.K),
            "used_skus": len(used_skus),
            "unused_skus": unused_skus,
            "order_size_min": min(order_sizes) if order_sizes else 0,
            "order_size_max": max(order_sizes) if order_sizes else 0,
            "order_size_mean": sum(order_sizes) / len(order_sizes) if order_sizes else 0.0,
            "order_size_median": sorted(order_sizes)[len(order_sizes)//2] if order_sizes else 0,
            "order_size_stdev": sum((x - sum(order_sizes) / len(order_sizes))**2 for x in order_sizes) / len(order_sizes) if order_sizes else 0.0,
            
            "rt_min": min(rt_vals) if rt_vals else 0,
            "rt_max": max(rt_vals) if rt_vals else 0,
            "rt_mean": sum(rt_vals) / len(rt_vals) if rt_vals else 0.0,
            "rt_stdev": sum((x - sum(rt_vals) / len(rt_vals))**2 for x in rt_vals) / len(rt_vals) if rt_vals else 0.0,
            
            "p_min": min(p_vals) if p_vals else 0,
            "p_max": max(p_vals) if p_vals else 0,
            "p_mean": sum(p_vals) / len(p_vals) if p_vals else 0.0,
            "p_stdev": sum((x - sum(p_vals) / len(p_vals))**2 for x in p_vals) / len(p_vals) if p_vals else 0.0,
            
            "n_min": min(n_vals) if n_vals else 0,
            "n_max": max(n_vals) if n_vals else 0,
            "n_mean": sum(n_vals) / len(n_vals) if n_vals else 0.0,
            "n_stdev": sum((x - sum(n_vals) / len(n_vals))**2 for x in n_vals) / len(n_vals) if n_vals else 0.0,

            "sku_freq_min": min(sku_freq.values()) if sku_freq.values() else 0,
            "sku_freq_max": max(sku_freq.values()) if sku_freq.values() else 0,
            "sku_freq_mean": sum(sku_freq.values()) / len(sku_freq.values()) if sku_freq.values() else 0.0,
            "sku_freq_stdev": sum((x - sum(sku_freq.values()) / len(sku_freq.values()))**2 for x in sku_freq.values()) / len(sku_freq.values()) if sku_freq.values() else 0.0,
            
            "pareto_ratio": pareto_ratio,
            "avg_jaccard": avg_jaccard,
            "total_pick_lines": sum(order_sizes),

            "supply_demand_ratio": supply_demand_ratio,
            "supply_demand_ratio_stdev": supply_demand_ratio_stdev,
        }

    def get_features(self) -> List[float]:
        """
        Convert statistics dict to a list of numeric features and normalize to around 0..1.
        """
        stats = self.get_statistics()
        return [
            stats["num_stations"] / 20,
            stats["num_lanes"] / 20,
            stats["num_orders"] / 1000,
            stats["num_skus"] / 10000,
            stats["used_skus"] / stats["num_skus"],
            stats["order_size_min"] / 10,
            stats["order_size_max"] / 10,
            stats["order_size_mean"] / 10,
            stats["order_size_median"] / 10,
            stats["order_size_stdev"] / 10,
            stats["rt_min"] / 100,
            stats["rt_max"] / 100,
            stats["rt_mean"] / 100,
            stats["rt_stdev"] / 100,
            stats["p_min"] / 10,
            stats["p_max"] / 10,
            stats["p_mean"] / 10,
            stats["p_stdev"] / 10,
            stats["n_min"] / 5,
            stats["n_max"] / 5,
            stats["n_mean"] / 5,
            stats["n_stdev"] / 5,
            stats["sku_freq_min"] / stats["num_orders"],
            stats["sku_freq_max"] / stats["num_orders"],
            stats["sku_freq_mean"] / stats["num_orders"],
            stats["sku_freq_stdev"] / stats["num_orders"],
            stats["pareto_ratio"],
            stats["avg_jaccard"],
            stats["total_pick_lines"] / 5,
            stats["supply_demand_ratio"] / 2,
            stats["supply_demand_ratio_stdev"] / 2,
        ]


    def print_summary(self) -> None:
        """Print a human-readable summary of the instance."""
        stats = self.get_statistics()
        print("=" * 60)
        print("  AutoStore Instance Summary")
        print("=" * 60)
        print(f"  Stations:  {stats['num_stations']}")
        print(f"  Lanes/st:  {stats['num_lanes']}")
        print(f"  Orders:    {stats['num_orders']}")
        print(f"  SKUs:      {stats['num_skus']}  (used: {stats['used_skus']}, unused: {stats['unused_skus']})")
        print()
        print(f"  Order size:  min={stats['order_size_min']}, max={stats['order_size_max']}, "
              f"mean={stats['order_size_mean']:.2f}, "
              f"median={stats['order_size_median']}")
        print(f"  rt (sec):    min={stats['rt_min']}, max={stats['rt_max']}, "
              f"mean={stats['rt_mean']:.1f}")
        print(f"  p  (sec):    min={stats['p_min']}, max={stats['p_max']}, "
              f"mean={stats['p_mean']:.1f}")
        print(f"  N  (bins):   min={stats['n_min']}, max={stats['n_max']}, "
              f"mean={stats['n_mean']:.1f}")
        print()
        print(f"  Pareto check: top 20% of used SKUs account for "
              f"{stats['pareto_ratio']:.1%} of all picks")
        print(f"  Avg Jaccard overlap (first 50 orders): {stats['avg_jaccard']:.3f}")
        print(f"  Total pick lines: {stats['total_pick_lines']}")
        print("=" * 60)

    # Support unpacking (e.g. S, L, K, orders_req, rt, p, N = instance)
    def __iter__(self):
        yield self.S
        yield self.L
        yield self.K
        yield self._orders_requirements
        yield self.rt
        yield self.p
        yield self.N

    def __getitem__(self, index):
        return [self.S, self.L, self.K, self._orders_requirements, self.rt, self.p, self.N][index]

    def __len__(self):
        return 7
