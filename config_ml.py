"""
ML/RL конфигурация для офлайн-исследований рынка.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    csv_path: str = "data/historical_data.csv"
    timestamp_col: str = "timestamp"
    feature_columns: tuple = ("open", "high", "low", "close", "volume")
    train_split: float = 0.7
    val_split: float = 0.15
    sequence_length: int = 64
    target_horizon: int = 1


@dataclass(frozen=True)
class TransformerConfig:
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    ff_dim: int = 128
    dropout: float = 0.1


@dataclass(frozen=True)
class RLConfig:
    initial_cash: float = 100_000.0
    transaction_cost: float = 0.0005
    max_position_size: float = 1.0
    gamma: float = 0.99
    learning_rate: float = 1e-3
    entropy_coef: float = 1e-3
    episodes: int = 20
    max_steps_per_episode: int = 10_000


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 64
    transformer_epochs: int = 12
    transformer_lr: float = 1e-3
    device: str = "cpu"
    save_dir: str = "saved_models"


DATA = DataConfig()
TRANSFORMER = TransformerConfig()
RL = RLConfig()
TRAIN = TrainConfig()
