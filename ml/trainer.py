from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import asdict
from typing import Dict

import numpy as np

from config_ml import DATA, RL, TRAIN, TRANSFORMER
from environment.trading_env import EnvConfig, TradingEnv
from ml.data_processor import (
    add_handcrafted_features,
    fit_standardizer,
    load_ohlcv,
    make_sequences,
    split_df,
    transform_with_standardizer,
)
from models.transformer import NumpyFeatureExtractor, PriceTransformer

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None

logger = logging.getLogger(__name__)


def _ensure_save_dir():
    os.makedirs(TRAIN.save_dir, exist_ok=True)


def train_transformer(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray):
    if torch is None:
        raise ImportError("Для обучения transformer нужен torch.")

    model = PriceTransformer(
        input_dim=x_train.shape[-1],
        d_model=TRANSFORMER.d_model,
        n_heads=TRANSFORMER.n_heads,
        n_layers=TRANSFORMER.n_layers,
        ff_dim=TRANSFORMER.ff_dim,
        dropout=TRANSFORMER.dropout,
    ).to(TRAIN.device)
    opt = torch.optim.Adam(model.parameters(), lr=TRAIN.transformer_lr)

    xtr = torch.tensor(x_train, dtype=torch.float32, device=TRAIN.device)
    ytr = torch.tensor(y_train, dtype=torch.float32, device=TRAIN.device)
    xva = torch.tensor(x_val, dtype=torch.float32, device=TRAIN.device)
    yva = torch.tensor(y_val, dtype=torch.float32, device=TRAIN.device)

    best_val = float("inf")
    best_state = None

    logger.info(
        "Transformer: эпохи=%s, device=%s, train=%s val=%s",
        TRAIN.transformer_epochs,
        TRAIN.device,
        x_train.shape,
        x_val.shape,
    )
    t0 = time.perf_counter()
    for epoch in range(1, TRAIN.transformer_epochs + 1):
        model.train()
        _, pred = model(xtr)
        loss = F.mse_loss(pred, ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            _, val_pred = model(xva)
            val_loss = F.mse_loss(val_pred, yva).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        logger.info(
            "Transformer: эпоха %s/%s train_mse=%.6f val_mse=%.6f best_val=%.6f",
            epoch,
            TRAIN.transformer_epochs,
            float(loss.item()),
            val_loss,
            best_val,
        )
        sys.stdout.flush()

    if best_state is not None:
        model.load_state_dict(best_state)
    logger.info("Transformer: готово за %.1f с", time.perf_counter() - t0)
    return model


def extract_features(model, x: np.ndarray, label: str = "features") -> np.ndarray:
    n = len(x)
    if n == 0:
        return np.zeros((0, TRANSFORMER.d_model), dtype=np.float32)

    if torch is None:
        logger.info("%s: режим без torch, numpy-фичи для %s окон...", label, n)
        fallback = NumpyFeatureExtractor()
        feats = []
        log_every = max(1, n // 20)
        t0 = time.perf_counter()
        for i, win in enumerate(x):
            feats.append(fallback.transform(win).features)
            if (i + 1) % log_every == 0 or i + 1 == n:
                logger.info("%s: обработано %s / %s окон", label, i + 1, n)
                sys.stdout.flush()
        logger.info("%s: numpy готово за %.1f с", label, time.perf_counter() - t0)
        return np.stack(feats, axis=0)

    model.eval()
    batch_size = max(1, int(TRAIN.batch_size))
    chunks = []
    t0 = time.perf_counter()
    logger.info(
        "%s: извлечение признаков батчами batch_size=%s, всего окон=%s, device=%s",
        label,
        batch_size,
        n,
        TRAIN.device,
    )
    sys.stdout.flush()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            chunk = x[start:end]
            xt = torch.tensor(chunk, dtype=torch.float32, device=TRAIN.device)
            h, _ = model(xt)
            chunks.append(h.cpu().numpy())
            if end == n or end % max(batch_size * 10, 1) == 0:
                logger.info("%s: извлечено %s / %s окон", label, end, n)
                sys.stdout.flush()
    out = np.concatenate(chunks, axis=0)
    logger.info("%s: готово, shape=%s, за %.1f с", label, out.shape, time.perf_counter() - t0)
    return out


def train_rl_agent(df_for_env, features: np.ndarray):
    from models.rl_agent import PPOAgent, Transition

    env = TradingEnv(
        df=df_for_env,
        features=features,
        config=EnvConfig(
            initial_cash=RL.initial_cash,
            transaction_cost=RL.transaction_cost,
            max_position_size=RL.max_position_size,
        ),
    )
    state_dim = env.reset().shape[0]
    agent = PPOAgent(state_dim=state_dim, lr=RL.learning_rate, gamma=RL.gamma)

    max_steps = min(RL.max_steps_per_episode, max(0, env.n_steps - 1))
    logger.info(
        "RL (PPO): эпизодов=%s, до %s шагов на эпизод, строк в env=%s, state_dim=%s",
        RL.episodes,
        max_steps,
        env.n_steps,
        state_dim,
    )
    sys.stdout.flush()
    t_all = time.perf_counter()
    for ep in range(1, RL.episodes + 1):
        state = env.reset()
        ep_reward = 0.0
        steps = 0
        t_ep = time.perf_counter()
        for _ in range(max_steps):
            action, log_prob = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(
                Transition(
                    state=state,
                    action=action,
                    log_prob=log_prob,
                    reward=reward,
                    done=done,
                )
            )
            ep_reward += reward
            steps += 1
            state = next_state
            if done:
                break
        agent.update()
        logger.info(
            "RL: эпизод %s/%s шагов=%s суммарный reward=%.4f за %.1f с",
            ep,
            RL.episodes,
            steps,
            ep_reward,
            time.perf_counter() - t_ep,
        )
        sys.stdout.flush()
    logger.info("RL: все эпизоды за %.1f с", time.perf_counter() - t_all)
    return agent


def run_full_training() -> Dict[str, float]:
    _ensure_save_dir()
    logger.info("=== Обучение: старт ===")
    logger.info("CSV: %s", os.path.abspath(DATA.csv_path))
    sys.stdout.flush()

    t_load = time.perf_counter()
    raw = load_ohlcv(DATA.csv_path)
    logger.info("Загружено строк OHLCV: %s (%.2f с)", len(raw), time.perf_counter() - t_load)
    sys.stdout.flush()

    t_feat = time.perf_counter()
    feat_df = add_handcrafted_features(raw)
    logger.info(
        "После фичей: строк=%s (%.2f с), период примерно %s .. %s",
        len(feat_df),
        time.perf_counter() - t_feat,
        feat_df[DATA.timestamp_col].iloc[0] if len(feat_df) else None,
        feat_df[DATA.timestamp_col].iloc[-1] if len(feat_df) else None,
    )
    sys.stdout.flush()

    splits = split_df(feat_df)
    logger.info(
        "Сплит train/val/test: %s / %s / %s строк",
        len(splits.train),
        len(splits.val),
        len(splits.test),
    )
    sys.stdout.flush()

    feature_cols = tuple(DATA.feature_columns) + ("log_return", "hl_spread", "oc_change", "vol_z")
    t_seq = time.perf_counter()
    x_train, y_train = make_sequences(splits.train, feature_cols, DATA.sequence_length)
    x_val, y_val = make_sequences(splits.val, feature_cols, DATA.sequence_length)
    x_test, y_test = make_sequences(splits.test, feature_cols, DATA.sequence_length)
    logger.info(
        "Окна (sequence_length=%s): train=%s val=%s test=%s за %.2f с",
        DATA.sequence_length,
        x_train.shape,
        x_val.shape,
        x_test.shape,
        time.perf_counter() - t_seq,
    )
    sys.stdout.flush()
    if min(len(x_train), len(x_val), len(x_test)) == 0:
        raise ValueError(
            "Недостаточно данных после сплита/окон. "
            "Увеличьте размер historical_data.csv или уменьшите sequence_length в config_ml.py."
        )

    scaler = fit_standardizer(x_train)
    x_train = transform_with_standardizer(x_train, scaler)
    x_val = transform_with_standardizer(x_val, scaler)
    x_test = transform_with_standardizer(x_test, scaler)
    logger.info("Нормализация (scaler) применена к train/val/test")

    model = train_transformer(x_train, y_train, x_val, y_val)
    train_features = extract_features(model, x_train, label="train_features")
    test_features = extract_features(model, x_test, label="test_features")

    df_env_train = splits.train.iloc[DATA.sequence_length :].reset_index(drop=True)
    df_env_test = splits.test.iloc[DATA.sequence_length :].reset_index(drop=True)
    logger.info(
        "Среда RL: df_env_train=%s строк, df_env_test=%s строк",
        len(df_env_train),
        len(df_env_test),
    )
    if len(df_env_train) != len(train_features):
        logger.warning(
            "Размер train_features (%s) != len(df_env_train) (%s) — проверьте сплит/окна",
            len(train_features),
            len(df_env_train),
        )
    rl_agent = train_rl_agent(df_env_train, train_features)

    logger.info("Оценка на тесте: прогон среды...")
    sys.stdout.flush()
    env_test = TradingEnv(
        df=df_env_test,
        features=test_features,
        config=EnvConfig(RL.initial_cash, RL.transaction_cost, RL.max_position_size),
    )
    state = env_test.reset()
    done = False
    step_eval = 0
    while not done:
        action, _ = rl_agent.act(state)
        state, _, done, _ = env_test.step(action)
        step_eval += 1
    logger.info("Оценка: шагов=%s, итоговая equity=%.2f", step_eval, env_test.equity)

    final_equity = env_test.equity
    stats = {"final_equity": float(final_equity), "pnl_pct": float((final_equity / RL.initial_cash - 1) * 100.0)}

    logger.info("Сохранение в %s ...", os.path.abspath(TRAIN.save_dir))
    sys.stdout.flush()
    with open(os.path.join(TRAIN.save_dir, "scaler.json"), "w", encoding="utf-8") as f:
        payload = {"mean": scaler["mean"].tolist(), "std": scaler["std"].tolist(), "feature_cols": list(feature_cols)}
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if torch is not None:
        torch.save(model.state_dict(), os.path.join(TRAIN.save_dir, "transformer.pth"))
        torch.save(rl_agent.policy.state_dict(), os.path.join(TRAIN.save_dir, "rl_agent.pth"))

    with open(os.path.join(TRAIN.save_dir, "training_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    with open(os.path.join(TRAIN.save_dir, "config_snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "DATA": asdict(DATA),
                "TRANSFORMER": asdict(TRANSFORMER),
                "RL": asdict(RL),
                "TRAIN": asdict(TRAIN),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info("=== Обучение: финиш, pnl_pct=%.2f%% ===", stats["pnl_pct"])
    sys.stdout.flush()
    return stats
