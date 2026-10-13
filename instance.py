from itertools import product
from manim.animation import transform_matching_parts
from collections import defaultdict
import pickle
from typing import Any
import numpy as np
from torch_geometric.data import HeteroData
import torch

from constants import PARAM_LEVELS, REFERENCE_CONFIG
from jaccard_similarity import (
    compute_jaccard_weighted_and_unweighted,
)


class Instance:
    """Represents an order scheduling instance for a robotic compact storage system (AutoStore)."""

    def __init__(
        self,
        S: list[int],
        L: list[int],
        K: list[int],
        orders_requirements: dict[int, list[int]],
        rt: dict[int, int],
        p: dict[int, int],
        N: dict[int, int],
        movecap: int,
        seed: int,
        rt_ret: dict[int, int] | None = None,
        pickcap: int = 1,
    ):
        self.S = S
        self.L = L
        self.K = K
        self._orders_requirements = orders_requirements
        self.rt = rt
        self.p = p
        self.N = N
        self._rt_ret = rt_ret if rt_ret is not None else dict(rt)
        self._movecap = movecap
        self._pickcap = pickcap
        self._seed = seed

        all_skus = np.concatenate(
            [np.array(reqs) for reqs in self._orders_requirements.values()]
        )
        self.unique_skus, self.sku_freq_vals = np.unique(all_skus, return_counts=True)

        self.sku_to_orders_map = defaultdict(list)
        for o, req in self.orders_requirements.items():
            for k in req:
                self.sku_to_orders_map[k].append(o)

        self.stats = self.get_statistics()
        self.features = self.get_features()

    @property
    def orders_requirements(self) -> dict[int, list[int]]:
        return self._orders_requirements

    @orders_requirements.setter
    def orders_requirements(self, value: dict[int, list[int]]):
        self._orders_requirements = value

    @property
    def orders_req(self) -> dict[int, list[int]]:
        return self._orders_requirements

    @orders_req.setter
    def orders_req(self, value: dict[int, list[int]]):
        self._orders_requirements = value

    @property
    def O(self) -> list[int]:
        """Sorted list of order IDs."""
        return sorted(self._orders_requirements.keys())

    @property
    def rt_ret(self) -> dict[int, int]:
        """A dictionary mapping SKU to its return time, defaulting to its retrieval time."""
        return self._rt_ret

    @rt_ret.setter
    def rt_ret(self, value: dict[int, int]):
        self._rt_ret = value

    @property
    def movecap(self) -> int:
        return self._movecap

    @property
    def pickcap(self) -> int:
        return self._pickcap if hasattr(self, "_pickcap") else len(self.L)

    @property
    def seed(self) -> int:
        return self._seed

    def get_statistics(self) -> dict[str, Any]:
        """Calculate and return key statistics about the instance."""
        import itertools as _itertools

        num_orders = len(self._orders_requirements)

        # Convert to numpy arrays for vectorized operations
        order_sizes = np.array([len(v) for v in self._orders_requirements.values()])
        rt_vals = np.array(list(self.rt.values()))
        rt_ret_vals = np.array(list(self.rt_ret.values()))
        p_vals = np.array(list(self.p.values()))
        n_vals = np.array(list(self.N.values()))

        # SKU frequency across orders (demand for SKUs) via flat array
        

        # Order time intensities (how long would unparallelized orders take)
        rt2p = rt_vals + rt_ret_vals + p_vals
        order_times_list = np.array(
            [rt2p[list(reqs)].sum() for reqs in self._orders_requirements.values()]
        )

        used_skus_mask = np.isin(np.array(self.K), self.unique_skus)
        used_skus = list(np.array(self.K)[used_skus_mask])
        unused_skus = len(self.K) - len(used_skus)

        # Pareto checks for top x% of SKUs
        total_picks = self.sku_freq_vals.sum()
        sku_freq_sorted = np.sort(self.sku_freq_vals)[::-1]  # descending
        cumsum = np.cumsum(sku_freq_sorted)

        percentages = [1, 5, 10, 20, 30, 40, 50, 75]
        pareto_ratios = {}
        for perc in percentages:
            top_count = max(1, len(used_skus) * perc // 100)
            pareto_ratios[perc] = (
                float(cumsum[top_count - 1] / total_picks) if total_picks else 0.0
            )

        # SKU overlap: average Jaccard similarity between order pairs (sample first 50 orders)
        order_keys = list(self._orders_requirements.keys())
        pairs = list(_itertools.combinations(order_keys[:50], 2))
        if pairs:
            jaccards_w = []
            jaccards_unw = []
            for i, j in pairs:
                j_w, j_uw = compute_jaccard_weighted_and_unweighted(
                    i, j, self._orders_requirements, rt=self.rt, rt_ret=self.rt_ret
                )
                jaccards_w.append(j_w)
                jaccards_unw.append(j_uw)
            avg_jaccard_w = sum(jaccards_w) / len(jaccards_w)
            avg_jaccard_unw = sum(jaccards_unw) / len(jaccards_unw)
        else:
            avg_jaccard_w = 0.0
            avg_jaccard_unw = 0.0

        # Supply (number of bins per SKU) and demand (number of orders per SKU)
        supply = n_vals[np.array(used_skus, dtype=int)]
        demand = self.sku_freq_vals

        supply_demand_ratios = supply / demand
        supply_demand_ratio_min = float(supply_demand_ratios.min())
        supply_demand_ratio_max = float(supply_demand_ratios.max())
        supply_demand_ratio_median = float(np.median(supply_demand_ratios))
        supply_demand_ratio_mean = float(supply_demand_ratios.mean())
        supply_demand_ratio_stdev = float(np.sqrt(((supply - demand) ** 2).mean()))

        # Movecap
        movecap_norm = (self.movecap / 100) if self.movecap is not None else 0.0
        movecap_norm_per_lane = movecap_norm / (len(self.S) * len(self.L))

        return {
            "num_stations": len(self.S),
            "num_lanes": len(self.L),
            "num_orders": num_orders,
            "num_skus": len(self.K),
            "used_skus": len(used_skus),
            "unused_skus": unused_skus,
            "total_order_volume": float(order_sizes.sum()),
            "order_size_min": float(order_sizes.min()) if len(order_sizes) else 0,
            "order_size_max": float(order_sizes.max()) if len(order_sizes) else 0,
            "order_size_mean": float(order_sizes.mean()) if len(order_sizes) else 0.0,
            "order_size_median": float(np.median(order_sizes))
            if len(order_sizes)
            else 0.0,
            "order_size_stdev": float(order_sizes.std(ddof=0))
            if len(order_sizes)
            else 0.0,
            "order_times_min": float(order_times_list.min()),
            "order_times_max": float(order_times_list.max()),
            "order_times_mean": float(order_times_list.mean()),
            "order_times_median": float(np.median(order_times_list)),
            "order_times_stdev": float(order_times_list.std(ddof=0)),
            "rt_min": float(rt_vals.min()) if len(rt_vals) else 0,
            "rt_max": float(rt_vals.max()) if len(rt_vals) else 0,
            "rt_mean": float(rt_vals.mean()) if len(rt_vals) else 0.0,
            "rt_median": float(np.median(rt_vals)) if len(rt_vals) else 0.0,
            "rt_stdev": float(rt_vals.std(ddof=0)) if len(rt_vals) else 0.0,
            "p_min": float(p_vals.min()) if len(p_vals) else 0,
            "p_max": float(p_vals.max()) if len(p_vals) else 0,
            "p_mean": float(p_vals.mean()) if len(p_vals) else 0.0,
            "p_median": float(np.median(p_vals)) if len(p_vals) else 0.0,
            "p_stdev": float(p_vals.std(ddof=0)) if len(p_vals) else 0.0,
            "n_min": float(n_vals.min()) if len(n_vals) else 0,
            "n_max": float(n_vals.max()) if len(n_vals) else 0,
            "n_mean": float(n_vals.mean()) if len(n_vals) else 0.0,
            "n_median": float(np.median(n_vals)) if len(n_vals) else 0.0,
            "n_stdev": float(n_vals.std(ddof=0)) if len(n_vals) else 0.0,
            "sku_freq_min": float(demand.min()) if len(demand) else 0,
            "sku_freq_max": float(demand.max()) if len(demand) else 0,
            "sku_freq_mean": float(demand.mean()) if len(demand) else 0.0,
            "sku_freq_median": float(np.median(demand))
            if len(demand)
            else 0.0,
            "sku_freq_stdev": float(demand.std(ddof=0))
            if len(demand)
            else 0.0,
            "pareto_ratios": pareto_ratios,
            "avg_jaccard_weighted": avg_jaccard_w,
            "avg_jaccard_unweighted": avg_jaccard_unw,
            "supply_demand_ratio_min": supply_demand_ratio_min,
            "supply_demand_ratio_max": supply_demand_ratio_max,
            "supply_demand_ratio_mean": supply_demand_ratio_mean,
            "supply_demand_ratio_median": supply_demand_ratio_median,
            "supply_demand_ratio_stdev": supply_demand_ratio_stdev,
            "movecap_norm": movecap_norm,
            "movecap_norm_per_lane": movecap_norm_per_lane,
        }

    def get_features(self) -> dict[str, float]:
        """
        Convert statistics dict to a dict of numeric features keyed by feature name, normalized to around 0..1.
        """
        stats = self.stats
        return {
            # General instance config
            "feat_num_stations": stats["num_stations"] / max(PARAM_LEVELS["stations"]),
            "feat_num_lanes": stats["num_lanes"] / max(PARAM_LEVELS["lanes"]),
            "feat_num_orders": stats["num_orders"] / max(PARAM_LEVELS["orders"]),
            "feat_num_skus": stats["num_skus"] / (5000 * max(PARAM_LEVELS["orders"])),
            # Order size stats
            "feat_order_size_max": stats["order_size_max"] / 10,
            "feat_order_size_min": stats["order_size_min"] / stats["order_size_max"],
            "feat_order_size_mean": stats["order_size_mean"] / stats["order_size_max"],
            "feat_order_size_median": stats["order_size_median"]
            / stats["order_size_max"],
            "feat_order_size_cv": stats["order_size_stdev"] / stats["order_size_mean"],
            # Order completion times without parallelization
            "feat_order_times_max": stats["order_times_max"]
            / (stats["p_mean"] * 1000),  # relate to 1000 pick times
            "feat_order_times_min": stats["order_times_min"] / stats["order_times_max"],
            "feat_order_times_mean": stats["order_times_mean"]
            / stats["order_times_max"],
            "feat_order_times_median": stats["order_times_median"]
            / stats["order_times_max"],
            "feat_order_times_cv": stats["order_size_stdev"]
            / stats["order_times_mean"],
            # Retrieval times
            "feat_rt_min": stats["rt_min"] / stats["rt_max"],
            "feat_rt_mean": stats["rt_mean"] / stats["rt_max"],
            "feat_rt_cv": stats["rt_stdev"] / stats["rt_mean"],
            # Picking times
            "feat_p_mean": stats["p_mean"] / stats["p_max"],
            "feat_p_cv": stats["p_stdev"] / stats["p_mean"],
            # Bin counts
            "feat_n_mean": stats["n_mean"] / stats["n_max"],
            "feat_n_cv": stats["n_stdev"] / stats["n_mean"],
            "feat_retrieval_picking_ratio": stats["p_mean"] / stats["rt_mean"],
            # SKU frequency stats
            "feat_used_skus_ratio": stats["used_skus"] / REFERENCE_CONFIG["skus"],
            "feat_sku_freq_min": stats["sku_freq_min"] / stats["num_orders"],
            "feat_sku_freq_max": stats["sku_freq_max"] / stats["num_orders"],
            "feat_sku_freq_mean": stats["sku_freq_mean"] / stats["num_orders"],
            "feat_sku_freq_cv": stats["sku_freq_stdev"] / stats["num_orders"],
            # SKU overlap
            "feat_avg_jaccard_weighted": stats["avg_jaccard_weighted"],
            "feat_avg_jaccard_unweighted": stats["avg_jaccard_unweighted"],
            # Pareto ratios
            **{
                f"feat_pareto_ratio_{perc}": stats["pareto_ratios"][perc]
                for perc in [5, 10, 20, 30, 40, 50]
            },
            # Supply and Demand for SKUs
            # "feat_supply_demand_ratio_max": stats["supply_demand_ratio_max"] / 10,
            "feat_supply_demand_ratio_min": stats["supply_demand_ratio_min"]
            / stats["supply_demand_ratio_max"],
            "feat_supply_demand_ratio_mean": stats["supply_demand_ratio_mean"]
            / stats["supply_demand_ratio_max"],
            "feat_supply_demand_ratio_median": stats["supply_demand_ratio_median"]
            / stats["supply_demand_ratio_max"],
            "feat_supply_demand_ratio_cv": stats["supply_demand_ratio_stdev"]
            / (stats["supply_demand_ratio_mean"] * 2),  # halve it so it's <1.0
            # Movecap stats
            "feat_movecap": stats["movecap_norm"],
            "feat_movecap_per_lane": stats["movecap_norm_per_lane"],
            "feat_movecap_per_used_sku": stats["movecap_norm"]
            / (stats["used_skus"] * 2),
        }

    def print_summary(self) -> None:
        """Print a human-readable summary of the instance."""
        stats = self.stats
        print("=" * 60)
        print("  AutoStore Instance Summary")
        print("=" * 60)
        print(f"  Stations:  {stats['num_stations']}")
        print(f"  Lanes/st:  {stats['num_lanes']}")
        print(f"  Orders:    {stats['num_orders']}")
        print(
            f"  SKUs:      {stats['num_skus']}  (used: {stats['used_skus']}, unused: {stats['unused_skus']})"
        )
        print()
        print(
            f"  Order size:  min={stats['order_size_min']}, max={stats['order_size_max']}, "
            f"mean={stats['order_size_mean']:.2f}, "
            f"median={stats['order_size_median']}"
        )
        print(
            f"  rt (sec):    min={stats['rt_min']}, max={stats['rt_max']}, "
            f"mean={stats['rt_mean']:.1f}"
        )
        print(
            f"  p  (sec):    min={stats['p_min']}, max={stats['p_max']}, "
            f"mean={stats['p_mean']:.1f}"
        )
        print(
            f"  N  (bins):   min={stats['n_min']}, max={stats['n_max']}, "
            f"mean={stats['n_mean']:.1f}"
        )
        print()
        print(
            f"  Pareto check: top 20% of used SKUs account for "
            f"{stats['pareto_ratios'][20]:.1%} of all picks"
        )
        print(
            f"  Avg Jaccard overlap (first 50 orders): {stats['avg_jaccard_weighted']:.3f}"
        )
        print("=" * 60)

    def to_pickle(self, path: str) -> None:
        """Serialize this Instance to a pickle file at *path*."""
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_pickle(cls, path: str) -> "Instance":
        """Deserialize an Instance from a pickle file at *path*."""
        with open(path, "rb") as f:
            return pickle.load(f)

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
        return [
            self.S,
            self.L,
            self.K,
            self._orders_requirements,
            self.rt,
            self.p,
            self.N,
        ][index]

    def __len__(self):
        return 7

    
    def create_order_graph(self): 
        from jaccard_similarity import build_similarity_matrix, compute_demand_count
        
        O = self.O
        K = self.K
        
        o_to_idx = {o: i for i, o in enumerate(O)}
        k_to_idx = {k: i for i, k in enumerate(K)}
        
        sim_weighted, sim_unweighted = build_similarity_matrix(
            O, self.orders_req, self.rt, self.rt_ret, normalize=False
        )
        demand_count = compute_demand_count(self.orders_req, K)
        
        # Order features
        order_features = []
        for o in O:
            reqs = self.orders_req[o]
            num_items = len(reqs)
            
            overlaps_w = []
            overlaps_unw = []
            for other_o in O:
                if o == other_o:
                    continue
                key = (o, other_o) if o < other_o else (other_o, o)
                overlaps_w.append(sim_weighted[key])
                overlaps_unw.append(sim_unweighted[key])
                
            overlaps_w = np.array(overlaps_w)
            overlaps_unw = np.array(overlaps_unw)
            
            non_zero_overlap = (overlaps_unw > 0).sum()
            avg_w = overlaps_w.mean() if len(overlaps_w) > 0 else 0.0
            avg_unw = overlaps_unw.mean() if len(overlaps_unw) > 0 else 0.0
            std_w = overlaps_w.std(ddof=0) if len(overlaps_w) > 0 else 0.0
            std_unw = overlaps_unw.std(ddof=0) if len(overlaps_unw) > 0 else 0.0
            
            total_cycle_time = sum(self.rt[k] + self.p[k] + self.rt_ret[k] for k in reqs)
            
            order_features.append([
                num_items,
                non_zero_overlap,
                avg_w,
                avg_unw,
                std_w,
                std_unw,
                total_cycle_time
            ])
            
        order_x = torch.tensor(order_features, dtype=torch.float)
        order_x = torch.log1p(order_x)
        order_mean = order_x.mean(dim=0, keepdim=True)
        order_std = order_x.std(dim=0, keepdim=True)
        order_std[order_std == 0] = 1.0
        order_x = (order_x - order_mean) / order_std
        
        # Bin (SKU) features
        bin_features = []
        for k in K:
            demand = demand_count.get(k, 0)
            supply = self.N.get(k, 0)
            sup_dem_ratio = supply / demand if demand > 0 else float(supply)
            rt = self.rt.get(k, 0)
            p = self.p.get(k, 0)
            rt_ret = self.rt_ret.get(k, 0)
            
            bin_features.append([
                demand,
                supply,
                sup_dem_ratio,
                rt,
                p,
                rt_ret
            ])
            
        bin_x = torch.tensor(bin_features, dtype=torch.float)
        bin_x = torch.log1p(bin_x)
        bin_mean = bin_x.mean(dim=0, keepdim=True)
        bin_std = bin_x.std(dim=0, keepdim=True)
        bin_std[bin_std == 0] = 1.0
        bin_x = (bin_x - bin_mean) / bin_std
        
        # Global node features
        global_features = [list(self.features.values())]
        global_x = torch.tensor(global_features, dtype=torch.float)
        
        # Edges
        order_bin_src = []
        order_bin_dst = []
        for o in O:
            for k in self.orders_req[o]:
                order_bin_src.append(o_to_idx[o])
                order_bin_dst.append(k_to_idx[k])
                
        edge_index_order_bin = torch.tensor([order_bin_src, order_bin_dst], dtype=torch.long)
        edge_index_bin_order = torch.tensor([order_bin_dst, order_bin_src], dtype=torch.long)
        
        edge_index_global_order = torch.tensor(
            [[0] * len(O), list(range(len(O)))], dtype=torch.long
        )
        edge_index_order_global = torch.tensor(
            [list(range(len(O))), [0] * len(O)], dtype=torch.long
        )
        
        edge_index_global_bin = torch.tensor(
            [[0] * len(K), list(range(len(K)))], dtype=torch.long
        )
        edge_index_bin_global = torch.tensor(
            [list(range(len(K))), [0] * len(K)], dtype=torch.long
        )
        
        data = HeteroData()
        data['order'].x = order_x
        data['bin'].x = bin_x
        data['global'].x = global_x
        
        data['order', 'requires', 'bin'].edge_index = edge_index_order_bin
        data['bin', 'required_by', 'order'].edge_index = edge_index_bin_order
        
        data['global', 'connected_to', 'order'].edge_index = edge_index_global_order
        data['order', 'connected_to', 'global'].edge_index = edge_index_order_global
        
        data['global', 'connected_to', 'bin'].edge_index = edge_index_global_bin
        data['bin', 'connected_to', 'global'].edge_index = edge_index_bin_global
        
        return data
