from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


@dataclass
class TransformerOutput:
    features: np.ndarray
    prediction: Optional[np.ndarray] = None


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class PriceTransformer(nn.Module):
    """
    Transformer-энкодер:
    - извлекает скрытые признаки из OHLCV-окон;
    - дает прогноз следующего return.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ff_dim: int,
        dropout: float = 0.1,
    ):
        if torch is None or nn is None:
            raise ImportError("PyTorch не установлен. Установите torch для использования Transformer.")
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = _PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        # x: [B, T, F]
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h)
        h = self.norm(h)
        cls_like = h[:, -1, :]
        pred = self.head(cls_like).squeeze(-1)
        return cls_like, pred


class NumpyFeatureExtractor:
    """
    Fallback без torch: статистические признаки по окну.
    """

    def transform(self, window: np.ndarray) -> TransformerOutput:
        last = window[-1]
        mean = window.mean(axis=0)
        std = window.std(axis=0) + 1e-8
        features = np.concatenate([last, mean, std], axis=0).astype(np.float32)
        return TransformerOutput(features=features, prediction=None)
