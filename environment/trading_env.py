from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class EnvConfig:
    initial_cash: float = 100000.0
    transaction_cost: float = 0.0005
    max_position_size: float = 1.0


class TradingEnv:
    """
    Упрощенная среда:
    - action 0: hold
    - action 1: buy/increase long
    - action 2: sell/increase short
    """

    def __init__(self, df: pd.DataFrame, features: np.ndarray, config: EnvConfig):
        self.df = df.reset_index(drop=True)
        self.features = features
        self.config = config
        self.n_steps = len(df)
        self.reset()

    def reset(self):
        self.step_idx = 0
        self.cash = self.config.initial_cash
        self.position = 0.0
        self.equity = self.config.initial_cash
        self.prev_equity = self.equity
        return self._state()

    def _state(self):
        p = float(self.df.loc[self.step_idx, "close"])
        base = np.array([self.cash, self.position, p], dtype=np.float32)
        return np.concatenate([base, self.features[self.step_idx]], axis=0)

    def _mark_to_market(self, price: float):
        return self.cash + self.position * price

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        price = float(self.df.loc[self.step_idx, "close"])
        target_pos = self.position
        pos_limit = self.config.max_position_size

        if action == 1:
            target_pos = min(pos_limit, self.position + 0.1)
        elif action == 2:
            target_pos = max(-pos_limit, self.position - 0.1)

        delta = target_pos - self.position
        trade_value = delta * price
        fee = abs(trade_value) * self.config.transaction_cost

        self.cash -= trade_value + fee
        self.position = target_pos

        self.equity = self._mark_to_market(price)
        reward = self.equity - self.prev_equity
        self.prev_equity = self.equity

        self.step_idx += 1
        done = self.step_idx >= self.n_steps - 1
        next_state = self._state() if not done else np.zeros_like(self._state())
        info = {"equity": self.equity, "cash": self.cash, "position": self.position}
        return next_state, float(reward), done, info
