from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from config_ml import DATA, TRAIN, TRANSFORMER
from ml.data_processor import add_handcrafted_features
from models.transformer import NumpyFeatureExtractor, PriceTransformer

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


class HybridPredictor:
    """
    Инференс-модуль:
    - готовит последнее окно OHLCV+фич;
    - извлекает вектор через transformer;
    - возвращает action RL: hold/buy/sell.
    """

    def __init__(self, save_dir: str = TRAIN.save_dir):
        self.save_dir = save_dir
        self.scaler, self.feature_cols = self._load_scaler()
        self.transformer = self._load_transformer()
        self.rl_policy = self._load_policy()

    def _load_scaler(self):
        path = os.path.join(self.save_dir, "scaler.json")
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        scaler = {"mean": np.array(d["mean"], dtype=np.float32), "std": np.array(d["std"], dtype=np.float32)}
        return scaler, d["feature_cols"]

    def _load_transformer(self):
        if torch is None:
            return NumpyFeatureExtractor()
        model = PriceTransformer(
            input_dim=len(self.feature_cols),
            d_model=TRANSFORMER.d_model,
            n_heads=TRANSFORMER.n_heads,
            n_layers=TRANSFORMER.n_layers,
            ff_dim=TRANSFORMER.ff_dim,
            dropout=TRANSFORMER.dropout,
        )
        model.load_state_dict(torch.load(os.path.join(self.save_dir, "transformer.pth"), map_location="cpu"))
        model.eval()
        return model

    def _load_policy(self):
        if torch is None:
            return None
        from models.rl_agent import PolicyNet

        # state = [cash, position, price] + transformer_feature_dim
        state_dim = 3 + TRANSFORMER.d_model
        policy = PolicyNet(state_dim=state_dim)
        policy.load_state_dict(torch.load(os.path.join(self.save_dir, "rl_agent.pth"), map_location="cpu"))
        policy.eval()
        return policy

    def _build_window(self, df: pd.DataFrame) -> np.ndarray:
        prepared = add_handcrafted_features(df)
        if len(prepared) < DATA.sequence_length:
            raise ValueError(f"Недостаточно данных: нужно >= {DATA.sequence_length} строк после preprocessing.")
        arr = prepared.loc[:, self.feature_cols].values.astype(np.float32)[-DATA.sequence_length :]
        x = np.expand_dims(arr, axis=0)
        x = (x - self.scaler["mean"]) / self.scaler["std"]
        return x

    def _extract_feature(self, x: np.ndarray) -> np.ndarray:
        if torch is None or isinstance(self.transformer, NumpyFeatureExtractor):
            return self.transformer.transform(x[0]).features
        with torch.no_grad():
            h, _ = self.transformer(torch.tensor(x, dtype=torch.float32))
        return h[0].numpy()

    def predict_action(self, df: pd.DataFrame, cash: float, position: float) -> Dict:
        x = self._build_window(df)
        feat = self._extract_feature(x)
        price = float(df["close"].iloc[-1])
        state = np.concatenate([np.array([cash, position, price], dtype=np.float32), feat], axis=0)

        if self.rl_policy is None:
            # fallback: простой rule-based
            action = 1 if feat.mean() > 0 else 2
            confidence = float(abs(feat.mean()))
        else:
            with torch.no_grad():
                logits = self.rl_policy(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
                action = int(np.argmax(probs))
                confidence = float(np.max(probs))

        action_name = {0: "hold", 1: "buy", 2: "sell"}.get(action, "hold")
        return {"action": action, "action_name": action_name, "confidence": confidence, "price": price}
