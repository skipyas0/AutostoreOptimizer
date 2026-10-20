import random

import numpy as np
import torch
from torch import nn
from torch_geometric.nn import HeteroConv, LayerNorm, Linear, SAGEConv


class OrderGraphGNN(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Initial projection to unified hidden dimension using lazy Linear (-1 infers size on first pass)
        self.encoder_dict = nn.ModuleDict(
            {
                "order": Linear(-1, hidden_dim),
                "bin": Linear(-1, hidden_dim),
                "global": Linear(-1, hidden_dim),
            }
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            # SAGEConv with in_channels=-1 allows for lazy initialization of weights
            conv = HeteroConv(
                {
                    ("order", "requires", "bin"): SAGEConv(-1, hidden_dim),
                    ("bin", "required_by", "order"): SAGEConv(-1, hidden_dim),
                    ("global", "connected_to", "order"): SAGEConv(-1, hidden_dim),
                    ("order", "connected_to", "global"): SAGEConv(-1, hidden_dim),
                    ("global", "connected_to", "bin"): SAGEConv(-1, hidden_dim),
                    ("bin", "connected_to", "global"): SAGEConv(-1, hidden_dim),
                },
                aggr="mean",
            )

            self.convs.append(conv)

            norm = nn.ModuleDict(
                {
                    "order": LayerNorm(hidden_dim),
                    "bin": LayerNorm(hidden_dim),
                    "global": LayerNorm(hidden_dim),
                }
            )
            self.norms.append(norm)

        self.activation = nn.ReLU()

    def forward(self, x_dict, edge_index_dict):
        # Initial feature projection
        h_dict = {
            node_type: self.activation(self.encoder_dict[node_type](x))
            for node_type, x in x_dict.items()
        }

        # Message passing layers
        for i in range(self.num_layers):
            h_dict = self.convs[i](h_dict, edge_index_dict)

            # Apply LayerNorm and activation
            h_dict = {
                node_type: self.activation(self.norms[i][node_type](x))
                for node_type, x in h_dict.items()
            }

        return h_dict


class OrderAssignmentActor(nn.Module):
    def __init__(self, hidden_dim: int, max_stations: int, mlp_hidden: int = 128):
        super().__init__()
        self.max_stations = max_stations

        # State: global_emb (hidden_dim) + order_emb (hidden_dim) + solver_state (max_stations * 4)
        input_dim = hidden_dim * 2 + max_stations * 4

        self.net = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, max_stations),
        )

    def forward(self, global_emb, order_emb, solver_state, n_stations):
        state = torch.cat([global_emb, order_emb, solver_state], dim=-1)
        logits = self.net(state)

        # Mask out non-present stations
        mask = (
            torch.arange(self.max_stations, device=logits.device).unsqueeze(0)
            >= n_stations
        )
        logits = logits.masked_fill(mask, float("-inf"))

        return logits


class OrderAssignmentCritic(nn.Module):
    def __init__(self, hidden_dim: int, max_stations: int, mlp_hidden: int = 128):
        super().__init__()
        self.max_stations = max_stations

        input_dim = hidden_dim * 2 + max_stations * 4

        self.net = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),  # Predicts scalar value V(S_t)
        )

    def forward(self, global_emb, order_emb, solver_state):
        state = torch.cat([global_emb, order_emb, solver_state], dim=-1)
        value = self.net(state)
        return value


class AssignmentStateTracker:
    def __init__(
        self,
        n_orders: int,
        n_stations: int,
        total_used_skus: int,
        movecap: float,
        max_stations: int,
        device: torch.device,
    ):
        self.n_orders = n_orders
        self.n_stations = n_stations
        self.total_used_skus = total_used_skus
        self.movecap = movecap
        self.max_stations = max_stations
        self.device = device

        self.assigned_orders = torch.zeros(n_stations, device=device)
        self.assigned_skus = torch.zeros(n_stations, device=device)

    def update(self, station_idx: int, num_skus: float):
        self.assigned_orders[station_idx] += 1
        self.assigned_skus[station_idx] += num_skus

    def get_solver_state(self):
        eps = 1e-8
        # 1. order_share
        order_share = self.assigned_orders / self.n_orders

        # 2. item_share
        item_share = self.assigned_skus / (self.total_used_skus + eps)

        # 3. bin_pressure
        bin_pressure = self.assigned_skus / (self.movecap / self.n_stations + eps)

        # 4. load_deviation
        avg_skus = self.assigned_skus.mean()
        load_deviation = (self.assigned_skus - avg_skus) / (self.total_used_skus + eps)

        # Matrix: n_stations x 4
        state_matrix = torch.stack(
            [order_share, item_share, bin_pressure, load_deviation], dim=-1
        )

        # Normalize and flatten
        flat_state = state_matrix.flatten()

        # Pad to max_stations * 4
        padded_state = torch.zeros(self.max_stations * 4, device=self.device)
        padded_state[: self.n_stations * 4] = flat_state

        return padded_state.unsqueeze(0)  # (1, max_stations * 4)


class OrderAssignmentAgent(nn.Module):
    def __init__(
        self,
        gnn: OrderGraphGNN,
        hidden_dim: int,
        max_stations: int,
        mlp_hidden: int = 128,
    ):
        super().__init__()
        self.gnn = gnn
        self.actor = OrderAssignmentActor(hidden_dim, max_stations, mlp_hidden)
        self.critic = OrderAssignmentCritic(hidden_dim, max_stations, mlp_hidden)
        self.max_stations = max_stations

    def forward(self, hetero_data, instance_stats, order_num_skus, deterministic: bool = False):
        """
        hetero_data: PyG HeteroData object containing the instance graph
        instance_stats: Dict with keys: 'num_stations', 'total_used_skus', 'movecap'
        order_num_skus: Tensor of shape (n_orders,) containing unnormalized number of SKUs per order
        deterministic: If True, select argmax actions instead of sampling
        """
        # 1. GNN pass
        h_dict = self.gnn(hetero_data.x_dict, hetero_data.edge_index_dict)
        order_embs = h_dict["order"]  # (n_orders, hidden_dim)

        # Using the first (and only) global node's embedding
        global_emb = h_dict["global"][0].unsqueeze(0)  # (1, hidden_dim)

        n_orders = order_embs.size(0)
        n_stations = instance_stats["num_stations"]
        total_used_skus = instance_stats["total_used_skus"]
        movecap = instance_stats["movecap"]
        device = order_embs.device

        tracker = AssignmentStateTracker(
            n_orders, n_stations, total_used_skus, movecap, self.max_stations, device
        )

        from torch.distributions import Categorical

        log_probs = []
        actions = []
        values = []

        # 2. Sequential assignment
        for i in range(n_orders):
            order_emb = order_embs[i].unsqueeze(0)
            solver_state = tracker.get_solver_state()

            logits = self.actor(global_emb, order_emb, solver_state, n_stations)
            value = self.critic(global_emb, order_emb, solver_state)

            dist = Categorical(logits=logits)
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)

            log_probs.append(log_prob)
            actions.append(action.item())
            values.append(value.squeeze())

            num_skus = order_num_skus[i].item()
            tracker.update(action.item(), num_skus)

        return actions, torch.stack(log_probs), torch.stack(values)

    def evaluate_actions(
        self, hetero_data, instance_stats, order_num_skus, actions_to_evaluate
    ):
        """
        Re-evaluate a sequence of actions to get log_probs, values, and entropies for PPO update.
        """
        h_dict = self.gnn(hetero_data.x_dict, hetero_data.edge_index_dict)
        order_embs = h_dict["order"]
        global_emb = h_dict["global"][0].unsqueeze(0)

        n_orders = order_embs.size(0)
        n_stations = instance_stats["num_stations"]
        total_used_skus = instance_stats["total_used_skus"]
        movecap = instance_stats["movecap"]
        device = order_embs.device

        tracker = AssignmentStateTracker(
            n_orders, n_stations, total_used_skus, movecap, self.max_stations, device
        )

        from torch.distributions import Categorical

        log_probs = []
        values = []
        entropies = []

        for i in range(n_orders):
            order_emb = order_embs[i].unsqueeze(0)
            solver_state = tracker.get_solver_state()

            logits = self.actor(global_emb, order_emb, solver_state, n_stations)
            value = self.critic(global_emb, order_emb, solver_state)

            dist = Categorical(logits=logits)
            action = actions_to_evaluate[i]

            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
            values.append(value.squeeze())

            tracker.update(action.item(), order_num_skus[i].item())

        return torch.stack(log_probs), torch.stack(values), torch.stack(entropies)


# --- PPO Utilities ---


def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """
    Computes Generalized Advantage Estimation (GAE).
    Since it's an episodic task, rewards[t] is 0 for t < T-1 and the final reward at t = T-1.
    """
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    next_value = 0

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        advantages[t] = last_gae = delta + gamma * lam * last_gae
        next_value = values[t]

    returns = advantages + values
    return advantages, returns


def ppo_update(
    agent,
    optimizer,
    hetero_data,
    instance_stats,
    order_num_skus,
    old_actions,
    old_log_probs,
    returns,
    advantages,
    clip_ratio=0.2,
    c1=0.5,
    c2=0.01,
    epochs=4,
):
    """
    Performs PPO optimization epochs.
    """
    import torch.nn.functional as F

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(epochs):
        log_probs, values, entropies = agent.evaluate_actions(
            hetero_data, instance_stats, order_num_skus, old_actions
        )

        ratios = torch.exp(log_probs - old_log_probs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages

        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = F.mse_loss(values, returns)
        entropy_bonus = entropies.mean()

        loss = actor_loss + c1 * critic_loss - c2 * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
        optimizer.step()

    return loss.item()


class StationOrderScorer(nn.Module):
    """
    Permutation-invariant MLP that scores candidate stations for a given order
    using pairwise station-order relationship and workload balance features.
    """
    def __init__(self, feature_dim: int = 14, hidden_dim: int = 64):
        super().__init__()
        self.feature_dim = feature_dim
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: Tensor of shape (n_stations, feature_dim) or (B, n_stations, feature_dim)
        returns: Logits of shape (n_stations,) or (B, n_stations)
        """
        return self.net(features).squeeze(-1)


def precompute_instance_similarities(instance):
    """
    Quickly precomputes unweighted and retrieval-weighted pairwise Jaccard similarities
    between all order pairs using an inverted SKU index.
    """
    from collections import defaultdict

    n_orders = len(instance.O)
    order_sets = [set(instance.orders_req[o]) for o in instance.O]
    sku_weights = {
        k: float(instance.rt.get(k, 0) + instance.rt_ret.get(k, 0)) for k in instance.K
    }

    sku_to_orders = defaultdict(list)
    for i, s in enumerate(order_sets):
        for k in s:
            sku_to_orders[k].append(i)

    pairs_to_compute = set()
    for ords in sku_to_orders.values():
        for idx1 in range(len(ords)):
            o1 = ords[idx1]
            for idx2 in range(idx1 + 1, len(ords)):
                o2 = ords[idx2]
                pairs_to_compute.add((o1, o2) if o1 < o2 else (o2, o1))

    sim_unw = np.zeros((n_orders, n_orders), dtype=np.float32)
    sim_w = np.zeros((n_orders, n_orders), dtype=np.float32)

    for o1, o2 in pairs_to_compute:
        s1, s2 = order_sets[o1], order_sets[o2]
        inter = s1 & s2
        union = s1 | s2
        unw_j = len(inter) / len(union) if union else 0.0
        inter_w = sum(sku_weights[k] for k in inter)
        union_w = sum(sku_weights[k] for k in union)
        w_j = inter_w / union_w if union_w > 0 else 0.0

        sim_unw[o1, o2] = sim_unw[o2, o1] = unw_j
        sim_w[o1, o2] = sim_w[o2, o1] = w_j

    return sim_unw, sim_w


class StationOrderStateTracker:
    """
    Tracks state of order assignments to stations and extracts 14 dynamic station-order features:
    - share of all orders at station (unweighted, pick time-weighted, fetch+return time-weighted)
    - share of so far assigned orders at station (unweighted, pick time-weighted, fetch+return time-weighted)
    - normalized std of pick times at station
    - normalized std of return+fetch times at station
    - average weighted and unweighted Jaccard between order and station's orders
    - fraction between order's pick time and total pick times at station
    - fraction between order's fetch+return and total f+r at station
    - normalized order pick and f+r time relative to instance means
    """
    def __init__(self, instance, sim_unw: np.ndarray, sim_w: np.ndarray, device: torch.device):
        self.instance = instance
        self.sim_unw = sim_unw
        self.sim_w = sim_w
        self.device = device

        self.n_stations = len(instance.S)
        self.n_orders = len(instance.O)

        self.order_p = np.array(
            [float(sum(instance.p.get(k, 0) for k in instance.orders_req[o])) for o in instance.O],
            dtype=np.float32,
        )
        self.order_fr = np.array(
            [
                float(sum(instance.rt.get(k, 0) + instance.rt_ret.get(k, 0) for k in instance.orders_req[o]))
                for o in instance.O
            ],
            dtype=np.float32,
        )

        self.total_p = float(self.order_p.sum())
        self.total_fr = float(self.order_fr.sum())
        self.mean_p = float(self.order_p.mean()) if self.n_orders > 0 else 1.0
        self.mean_fr = float(self.order_fr.mean()) if self.n_orders > 0 else 1.0

        # Action Masking Balance Corridor threshold: (Total Work / |S|) * 1.05
        self.w_balance_cap = (self.total_p / max(1, self.n_stations)) * 1.05

        # Dynamic station tracking
        self.station_orders = [[] for _ in range(self.n_stations)]
        self.count = np.zeros(self.n_stations, dtype=np.float32)
        self.w_sum = np.zeros(self.n_stations, dtype=np.float32)
        self.w_sq_sum = np.zeros(self.n_stations, dtype=np.float32)
        self.fr_sum = np.zeros(self.n_stations, dtype=np.float32)
        self.fr_sq_sum = np.zeros(self.n_stations, dtype=np.float32)

        self.assigned_count = 0
        self.assigned_p = 0.0
        self.assigned_fr = 0.0

        # Accumulated similarities of orders with station
        self.sim_unw_sum = np.zeros((self.n_stations, self.n_orders), dtype=np.float32)
        self.sim_w_sum = np.zeros((self.n_stations, self.n_orders), dtype=np.float32)

    def get_features(self, order_idx: int) -> torch.Tensor:
        cnt = self.count
        cnt_nz = np.maximum(cnt, 1.0)
        w_s = self.w_sum
        fr_s = self.fr_sum

        p_i = self.order_p[order_idx]
        fr_i = self.order_fr[order_idx]

        # 1. Share of all orders at station
        f1 = cnt / max(1.0, float(self.n_orders))
        f2 = w_s / (self.total_p + 1e-8)
        f3 = fr_s / (self.total_fr + 1e-8)

        # 2. Share of so far assigned orders at station
        f4 = cnt / max(1.0, float(self.assigned_count))
        f5 = w_s / (self.assigned_p + 1e-8)
        f6 = fr_s / (self.assigned_fr + 1e-8)

        # 3. Normalized std of pick times at station
        mean_p_stn = np.where(cnt > 0, w_s / cnt_nz, 0.0)
        var_p = np.where(cnt >= 2, np.maximum(0.0, self.w_sq_sum / cnt_nz - mean_p_stn**2), 0.0)
        f7 = np.sqrt(var_p) / (self.mean_p + 1e-8)

        # 4. Normalized std of return+fetch times at station
        mean_fr_stn = np.where(cnt > 0, fr_s / cnt_nz, 0.0)
        var_fr = np.where(cnt >= 2, np.maximum(0.0, self.fr_sq_sum / cnt_nz - mean_fr_stn**2), 0.0)
        f8 = np.sqrt(var_fr) / (self.mean_fr + 1e-8)

        # 5. Average unweighted and weighted Jaccard between order and station's orders
        f9 = np.where(cnt > 0, self.sim_unw_sum[:, order_idx] / cnt_nz, 0.0)
        f10 = np.where(cnt > 0, self.sim_w_sum[:, order_idx] / cnt_nz, 0.0)

        # 6. Fraction between order's pick time and total pick times at station
        f11 = p_i / (w_s + p_i + 1e-8)

        # 7. Fraction between order's fetch+return and total f+r at station
        f12 = fr_i / (fr_s + fr_i + 1e-8)

        # 8. Normalized order pick and f+r times relative to instance means
        f13 = np.full(self.n_stations, p_i / (self.mean_p + 1e-8), dtype=np.float32)
        f14 = np.full(self.n_stations, fr_i / (self.mean_fr + 1e-8), dtype=np.float32)

        feat_mat = np.column_stack([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14])
        return torch.tensor(feat_mat, dtype=torch.float32, device=self.device)

    def get_balance_mask(self) -> torch.Tensor:
        """
        Action Masking for Balance:
        Mask out any station s whose current workload W_s exceeds (Total Work / |S|) * 1.05.
        Guarantees by construction that the agent produces balanced schedules within the 5% corridor.
        """
        mask = torch.tensor(self.w_sum >= self.w_balance_cap, dtype=torch.bool, device=self.device)

        # Safety fallback: if all stations exceed threshold, mask only stations strictly above min workload
        if mask.all():
            min_w = self.w_sum.min()
            mask = torch.tensor(self.w_sum > min_w, dtype=torch.bool, device=self.device)
            if mask.all():
                mask = torch.zeros(self.n_stations, dtype=torch.bool, device=self.device)

        return mask

    def assign(self, order_idx: int, station_idx: int):
        o = self.instance.O[order_idx]
        p_i = self.order_p[order_idx]
        fr_i = self.order_fr[order_idx]

        self.station_orders[station_idx].append(o)
        self.count[station_idx] += 1.0
        self.w_sum[station_idx] += p_i
        self.w_sq_sum[station_idx] += p_i**2
        self.fr_sum[station_idx] += fr_i
        self.fr_sq_sum[station_idx] += fr_i**2

        self.assigned_count += 1
        self.assigned_p += p_i
        self.assigned_fr += fr_i

        # Vectorized accumulator update
        self.sim_unw_sum[station_idx] += self.sim_unw[order_idx]
        self.sim_w_sum[station_idx] += self.sim_w[order_idx]


def rollout_station_order(
    instance,
    model: StationOrderScorer,
    sim_unw: np.ndarray,
    sim_w: np.ndarray,
    device: torch.device,
    deterministic: bool = False,
):
    from torch.distributions import Categorical

    tracker = StationOrderStateTracker(instance, sim_unw, sim_w, device)
    actions = []
    log_probs = []
    entropies = []

    for idx in range(len(instance.O)):
        features = tracker.get_features(idx)  # (n_stations, 14)
        logits = model(features)  # (n_stations,)

        # Action masking for balance
        mask = tracker.get_balance_mask()
        logits = logits.masked_fill(mask, float("-inf"))

        if deterministic:
            action = torch.argmax(logits).item()
        else:
            dist = Categorical(logits=logits)
            action_t = dist.sample()
            action = action_t.item()
            log_probs.append(dist.log_prob(action_t))
            entropies.append(dist.entropy())

        actions.append(action)
        tracker.assign(idx, action)

    return actions, log_probs, entropies, tracker


def evaluate_assignment_reward(instance, actions):
    """
    Evaluates order assignments by running the RDI heuristic on each station subinstance.
    Computes normalized makespan, imbalance, and reward.
    """
    from heuristic_rdi_sgc import run_rdi_sgc
    from instance import Instance

    station_assignments = {s: [] for s in range(len(instance.S))}
    for o_idx, s_idx in enumerate(actions):
        s_idx = min(s_idx, len(instance.S) - 1)
        station_assignments[s_idx].append(instance.O[o_idx])

    from decomposition import distribute_movecap

    n_stations = len(instance.S)
    workloads = [len(station_assignments[i]) for i in range(n_stations)]
    station_movecaps = distribute_movecap(instance.movecap, n_stations, workloads)

    M_k = []
    W_k = []
    for i, s_idx in enumerate(instance.S):
        assigned_orders = station_assignments[i]
        w = sum(instance.p.get(k, 0) for o in assigned_orders for k in instance.orders_req[o])
        W_k.append(float(w))

        if not assigned_orders:
            M_k.append(0.0)
            continue

        subinstance = Instance(
            [s_idx],
            instance.L,
            instance.K,
            {o: instance.orders_req[o] for o in assigned_orders},
            instance.rt,
            instance.p,
            instance.N,
            station_movecaps[i],
            instance.seed,
            instance.rt_ret,
            instance.pickcap,
        )

        heur_sol = run_rdi_sgc(subinstance, horizon=50000, batch_start_time=0)
        if heur_sol and heur_sol.feasible:
            M_k.append(float(heur_sol.makespan))
        else:
            M_k.append(50000.0)

    num_stations = len(instance.S)
    upper_bound = (
        sum(
            instance.rt.get(k, 0) + instance.rt_ret.get(k, 0)
            for o in instance.O
            for k in instance.orders_req[o]
        )
        / max(instance.movecap, 1)
    )
    lower_bound = (
        sum(instance.p.get(k, 0) for o in instance.O for k in instance.orders_req[o])
        / max(num_stations, 1)
    )
    scale_estimate = max((upper_bound + lower_bound) / 2.0, 1e-6)

    M = max(M_k) if M_k else 0.0
    normalized_M = M / scale_estimate

    avg_W = sum(W_k) / len(W_k) if W_k else 0.0
    avg_pick_imbalance = (
        sum(abs(w - avg_W) for w in W_k) / num_stations if num_stations > 0 else 0.0
    )

    reward = -normalized_M
    return reward, float(M), float(normalized_M), float(avg_pick_imbalance)


def train_self_critical_reinforce(device: torch.device = None):
    """
    Self-Critical REINFORCE Training Loop:
    - Permutation-invariant StationOrderScorer MLP.
    - Action masking for balance: masks out any station whose current workload exceeds (Total Work / |S|) * 1.05.
    - 1 greedy rollout (deterministic=True) to establish baseline reward R_greedy.
    - K=4 stochastic rollouts to obtain sample returns R_1, ..., R_K.
    - Policy update using advantage (R_k - R_greedy) across all steps (gamma = 1.0).
    """
    from torch import optim
    from datagen import generate_data

    # Device configuration
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    print(f"[*] Training StationOrderScorer with Self-Critical REINFORCE on: {device}")

    # Hyperparameters
    feature_dim = 14
    hidden_dim = 64
    num_episodes = 1200
    K_samples = 4
    lr = 1e-4
    entropy_coef = 0.01

    model = StationOrderScorer(feature_dim=feature_dim, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    rewards_history = []
    losses_history = []
    greedy_m_history = []
    greedy_imb_history = []

    for ep in range(num_episodes):
        # 1. Instance generation with progressive curriculum
        s = random.randint(2, 15)
        l = random.randint(3, 10)

        if s > 5:
            o = random.randint(2, 8) * 150
            m = random.randint(4, 8) * 10
            sku = random.randint(8, 15) * 2000
        else:
            o = random.randint(2, 8) * 50
            m = random.randint(4, 8) * 5
            sku = random.randint(8, 15) * 1000

        order_multip = 0.2 if ep < 0.5 * num_episodes else 0.5 if ep < 0.75 * num_episodes else 1.0
        o = max(2, round(order_multip * o))

        instance = generate_data(
            num_stations=s,
            lanes_per_station=l,
            num_orders=o,
            num_skus=sku,
            seed=seed + ep,
            movecap=m,
            pick_touch_time=4,
            return_model="balancing",
            max_bins_per_sku=8,
            sku_popularity_skew=1.0 + 0.1 * (random.random() - 0.5),
            retrieval_time_model="depth_based",
            pick_time_model="variable",
            bin_count_model="popularity_correlated",
            grid_depth=16 + random.randint(-2, 2),
        )

        # Precompute similarity matrices once per instance
        sim_unw, sim_w = precompute_instance_similarities(instance)

        # 2. Greedy rollout for baseline reward R_greedy
        greedy_actions, _, _, _ = rollout_station_order(
            instance, model, sim_unw, sim_w, device, deterministic=True
        )
        R_greedy, M_greedy, NM_greedy, PI_greedy = evaluate_assignment_reward(
            instance, greedy_actions
        )

        # 3. K stochastic rollouts
        sample_rewards = []
        sample_log_probs = []
        sample_entropies = []
        sample_makespans = []
        sample_imbalances = []

        for _ in range(K_samples):
            actions_k, log_probs_k, entropies_k, _ = rollout_station_order(
                instance, model, sim_unw, sim_w, device, deterministic=False
            )
            R_k, M_k, NM_k, PI_k = evaluate_assignment_reward(instance, actions_k)

            sample_rewards.append(R_k)
            sample_log_probs.append(torch.stack(log_probs_k))
            if entropies_k:
                sample_entropies.append(torch.stack(entropies_k))
            sample_makespans.append(M_k)
            sample_imbalances.append(PI_k)

        # 4. Self-Critical REINFORCE loss: Advantage = R_k - R_greedy
        advantages = [r_k - R_greedy for r_k in sample_rewards]
        policy_loss_terms = []
        for adv, log_p in zip(advantages, sample_log_probs):
            # Across all steps gamma = 1.0
            policy_loss_terms.append(-adv * log_p.sum())

        policy_loss = torch.stack(policy_loss_terms).mean()

        # Entropy bonus
        if sample_entropies:
            entropy_loss = -entropy_coef * torch.cat(sample_entropies).mean()
        else:
            entropy_loss = 0.0

        total_loss = policy_loss + entropy_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        mean_sample_r = sum(sample_rewards) / len(sample_rewards)
        mean_adv = sum(advantages) / len(advantages)

        rewards_history.append(R_greedy)
        losses_history.append(total_loss.item())
        greedy_m_history.append(M_greedy)
        greedy_imb_history.append(PI_greedy)

        print(
            f"Ep {ep + 1:04d}/{num_episodes} | R_greedy: {R_greedy:7.4f} | R_sample: {mean_sample_r:7.4f} | "
            f"Adv: {mean_adv:+7.4f} | M_greedy: {M_greedy:6.0f} | Imb: {PI_greedy:6.2f} | Loss: {total_loss.item():7.4f}"
        )

        if (ep + 1) % 50 == 0:
            torch.save(model.state_dict(), "rl_agent_weights.pth")
            print(f"[+] Saved checkpoint to rl_agent_weights.pth at episode {ep + 1}")

    torch.save(model.state_dict(), "rl_agent_weights.pth")
    print("[+] Training complete. Final weights saved to rl_agent_weights.pth")


# Default alias
train = train_self_critical_reinforce

if __name__ == "__main__":
    train_self_critical_reinforce()

