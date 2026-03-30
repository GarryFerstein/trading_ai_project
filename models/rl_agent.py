from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:  # pragma: no cover
    torch = None
    nn = None
    optim = None


@dataclass
class Transition:
    state: np.ndarray
    action: int
    log_prob: float
    reward: float
    done: bool


class PolicyNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128, action_dim: int = 3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.model(x)


class PPOAgent:
    """
    Небольшая PPO-реализация:
    actions: 0=hold, 1=buy, 2=sell.
    """

    def __init__(self, state_dim: int, lr: float = 1e-3, gamma: float = 0.99, clip_eps: float = 0.2):
        if torch is None:
            raise ImportError("PyTorch не установлен. RL-агент требует torch.")
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.policy = PolicyNet(state_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.memory: List[Transition] = []

    def act(self, state: np.ndarray) -> Tuple[int, float]:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        logits = self.policy(state_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return int(action.item()), float(dist.log_prob(action).item())

    def remember(self, transition: Transition):
        self.memory.append(transition)

    def _returns(self):
        returns = []
        running = 0.0
        for tr in reversed(self.memory):
            running = tr.reward + self.gamma * running * (1.0 - float(tr.done))
            returns.append(running)
        returns.reverse()
        rets = torch.tensor(returns, dtype=torch.float32)
        return (rets - rets.mean()) / (rets.std() + 1e-8)

    def update(self, epochs: int = 4):
        if not self.memory:
            return
        states = torch.tensor(np.array([t.state for t in self.memory]), dtype=torch.float32)
        actions = torch.tensor([t.action for t in self.memory], dtype=torch.long)
        old_log_probs = torch.tensor([t.log_prob for t in self.memory], dtype=torch.float32)
        returns = self._returns()

        for _ in range(epochs):
            logits = self.policy(states)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)

            unclipped = ratio * returns
            clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * returns
            loss = -torch.min(unclipped, clipped).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.memory.clear()
