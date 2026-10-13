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

    def forward(self, hetero_data, instance_stats, order_num_skus):
        """
        hetero_data: PyG HeteroData object containing the instance graph
        instance_stats: Dict with keys: 'num_stations', 'total_used_skus', 'movecap'
        order_num_skus: Tensor of shape (n_orders,) containing unnormalized number of SKUs per order
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


def train_ppo(device: torch.device = None):
    from torch import optim

    from datagen import generate_data
    from heuristic_rdi_sgc import run_rdi_sgc
    from instance import Instance

    # Device configuration
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    print(f"Using device for training: {device}")

    # Hyperparameters
    hidden_dim = 64
    mlp_hidden = 128
    num_episodes = 500
    epochs_per_episode = 4
    lr = 1e-4
    gamma = 0.99
    lam = 0.95
    penalty_lambda = 10.0

    max_stations = 20

    # Initialize networks
    gnn = OrderGraphGNN(hidden_dim=hidden_dim, num_layers=2)
    agent = OrderAssignmentAgent(
        gnn, hidden_dim=hidden_dim, max_stations=max_stations, mlp_hidden=mlp_hidden
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    rewards_list = []
    losses = []
    for ep in range(num_episodes):
        # 1. Generate instance
        s = random.randint(2, 12)
        l = random.randint(3, 10)
        if s > 5:
            o = random.randint(2, 8) * 100
            m = random.randint(4, 8) * 10
            sku = random.randint(8, 15) * 2000
        else:
            o = random.randint(2, 8) * 30
            m = random.randint(4, 8) * 5
            sku = random.randint(8, 15) * 1000

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

        # 2. Get HeteroData and move to device
        hetero_data = instance.create_order_graph().to(device)

        instance_stats = {
            "num_stations": len(instance.S),
            "total_used_skus": instance.stats["used_skus"],
            "movecap": instance.movecap,
        }

        # Prepare order_num_skus (unnormalized) on device
        order_num_skus = torch.tensor(
            [len(instance.orders_req[o]) for o in instance.O],
            dtype=torch.float,
            device=device,
        )

        # 3. Agent forward pass
        actions, log_probs, values = agent(hetero_data, instance_stats, order_num_skus)

        # Extract station assignments
        station_assignments = {s: [] for s in range(len(instance.S))}
        for o_idx, s_idx in enumerate(actions):
            # Safety clamp for stations
            s_idx = min(s_idx, len(instance.S) - 1)
            station_assignments[s_idx].append(instance.O[o_idx])

        # 4. Compute reward via heuristic
        M_k = []
        W_k = []  # weighted picks
        P_k = []  # unweighted picks
        Be_k = []  # bin_events
        Rt_k = []  # robot_time
        for i, s_idx in enumerate(instance.S):
            assigned_orders = station_assignments[i]

            # Workload W_k: picking time weighted sum of SKUs
            w = 0
            for o in assigned_orders:
                for k in instance.orders_req[o]:
                    w += instance.p.get(k, 0)
            W_k.append(w)

            if not assigned_orders:
                M_k.append(0)
                continue

            per_station_movecap = (
                instance.movecap // len(instance.S) if instance.movecap else None
            )

            subinstance = Instance(
                [s_idx],
                instance.L,
                instance.K,
                {o: instance.orders_req[o] for o in assigned_orders},
                instance.rt,
                instance.p,
                instance.N,
                per_station_movecap,
                instance.seed,
                instance.rt_ret,
                instance.pickcap,
            )

            heur_sol = run_rdi_sgc(subinstance, horizon=50000, batch_start_time=0)
            if heur_sol and heur_sol.feasible:
                M_k.append(heur_sol.makespan)
                # print(f"Solution for stations {s_idx} has {len(assigned_orders)} orders {len(heur_sol.bin_events[s_idx])} bin_events and {len(heur_sol.pick_events)} pick_events")
                Be_k.append(len(heur_sol.bin_events[s_idx]))
                P_k.append(len(heur_sol.pick_events))
                Rt_k.append(
                    sum(
                        [
                            be.fetch_end
                            - be.fetch_start
                            + be.return_end
                            - be.return_start
                            for be in heur_sol.bin_events[s_idx]
                        ]
                    )
                )
            else:
                M_k.append(50000)  # Penalty makespan for failure

        num_stations = len(instance.S)
        # Develop an estimate for makespan to normalize instance scale
        # 1. Upper bound: If each bin presence served just one pick, how long would it take
        # for the robots to fetch and return everything
        upper_bound = (
            sum(
                instance.rt.get(k, 0) + instance.rt_ret.get(k, 0)
                for o in instance.O
                for k in instance.orders_req[o]
            )
            / instance.movecap
        )
        # 2. Lower bound: If all pickfaces we saturated, how long would it take for pickers
        # to pick everything
        lower_bound = (
            sum(
                instance.p.get(k, 0) for o in instance.O for k in instance.orders_req[o]
            )
            / num_stations
        )

        scale_estimate = (upper_bound + lower_bound) / 2

        # Calculate final reward
        M = max(M_k)
        avg_W = sum(W_k) / len(W_k) if W_k else 0
        avg_M = sum(M_k) / len(M_k)
        min_M = min(M_k)

        movecap_utilization = sum(Rt_k) / (M * instance.movecap)

        avg_pick_imbalance = sum(abs(w - avg_W) for w in W_k) / num_stations
        shift_length_coeff = (avg_M + num_stations * min_M) / M
        picks_from_bin_presence = sum(P_k) / sum(Be_k) - 1
        normalized_M = M / scale_estimate

        # reward = -(normalized_M + penalty_lambda * normalized_avg_imbalance)
        # reward2 = -(normalized_M + 0.05 * avg_pick_imbalance)
        reward3 = (
            5 * movecap_utilization
            + 1000 * picks_from_bin_presence
            + shift_length_coeff
            - (3 * normalized_M + 0.1 * avg_pick_imbalance)
        )
        # 5. PPO Update
        reward = reward3
        rewards = torch.zeros(len(actions), device=device)
        rewards[-1] = reward

        advantages, returns = compute_gae(rewards, values.detach(), gamma, lam)
        old_actions = torch.tensor(actions, dtype=torch.long, device=device)

        loss = ppo_update(
            agent,
            optimizer,
            hetero_data,
            instance_stats,
            order_num_skus,
            old_actions,
            log_probs.detach(),
            returns,
            advantages,
            epochs=epochs_per_episode,
        )

        print(
            f"Episode {ep + 1:03d}/{num_episodes} - Reward: {reward:9.4f}) - Loss: {loss:7.4f}"
        )
        print(
            f"  - Instance | S: {len(instance.S)}, L: {len(instance.L)}, O: {len(instance.O)}, K: {len(instance.K)}, Movecap: {instance.movecap}"
        )
        print(
            f"  - Components | MU: {5 * movecap_utilization:7.4f}, PBP: {1000 * picks_from_bin_presence:7.4f}, SLC: {shift_length_coeff:7.4f}, NM: {-3 * normalized_M:7.4f}, PI: {-0.1 * avg_pick_imbalance:7.4f}"
        )
        rewards_list.append(reward)
        losses.append(loss)
    torch.save(agent.state_dict(), "rl_agent_weights.pth")
    print("Training complete. Weights saved to rl_agent_weights.pth")

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    x = range(1, num_episodes + 1)
    weights = np.ones(5) / 5
    rewards_ma = np.convolve(rewards_list, weights, mode="same")
    losses_ma = np.convolve(losses, weights, mode="same")
    ax1.plot(x, rewards_list, color="r")
    ax1.plot(x, rewards_ma, color="g")
    ax1.set_title("Rewards")

    ax2.plot(x, losses, color="b")
    ax2.plot(x, losses_ma, color="g")

    ax2.set_title("Losses")
    plt.tight_layout()
    plt.savefig("progress.jpg")


if __name__ == "__main__":
    train_ppo()
