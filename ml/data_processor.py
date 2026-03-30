from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from config_ml import DATA


@dataclass
class SplitData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def load_ohlcv(csv_path: str = DATA.csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {DATA.timestamp_col, *DATA.feature_columns}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"В CSV нет колонок: {sorted(missing)}")
    df[DATA.timestamp_col] = pd.to_datetime(df[DATA.timestamp_col], utc=True, errors="coerce")
    df = df.dropna(subset=[DATA.timestamp_col]).sort_values(DATA.timestamp_col).reset_index(drop=True)
    return df


def add_handcrafted_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_return"] = np.log(out["close"] / out["close"].shift(1)).fillna(0.0)
    out["hl_spread"] = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)
    out["oc_change"] = (out["close"] - out["open"]) / out["open"].replace(0, np.nan)
    out["vol_z"] = (out["volume"] - out["volume"].rolling(20).mean()) / (out["volume"].rolling(20).std() + 1e-8)
    out["target_return"] = out["close"].shift(-DATA.target_horizon) / out["close"] - 1.0
    out = out.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return out


def split_df(df: pd.DataFrame) -> SplitData:
    n = len(df)
    train_end = int(n * DATA.train_split)
    val_end = int(n * (DATA.train_split + DATA.val_split))
    return SplitData(train=df.iloc[:train_end], val=df.iloc[train_end:val_end], test=df.iloc[val_end:])


def make_sequences(df: pd.DataFrame, feature_cols: Tuple[str, ...], seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    feats = df.loc[:, list(feature_cols)].values.astype(np.float32)
    target = df["target_return"].values.astype(np.float32)

    xs, ys = [], []
    for i in range(seq_len, len(df)):
        xs.append(feats[i - seq_len : i])
        ys.append(target[i])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def fit_standardizer(x_train: np.ndarray) -> Dict[str, np.ndarray]:
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True) + 1e-8
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def transform_with_standardizer(x: np.ndarray, scaler: Dict[str, np.ndarray]) -> np.ndarray:
    return (x - scaler["mean"]) / scaler["std"]
