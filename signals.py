# Сигналы

import pandas as pd
import logging
from config import (
    STOP_LOSS_LONG_PERCENT, STOP_LOSS_SHORT_PERCENT,
    ADX_THRESHOLDS, ADX_DEFAULT_THRESHOLD,
    MOMENTUM_CHAIKIN_THRESHOLDS, MOMENTUM_CHAIKIN_DEFAULT
)
import time
from datetime import datetime, timezone

# Настройка логирования
logger = logging.getLogger(__name__)
_last_no_signal_time = {}

def _percent_to_fraction(p):
    try:
        return float(p) / 100.0
    except:
        logger.warning(f"Некорректный стоп-лосс процент: {p}, используется 1%")
        return 0.01

_SL_LONG_FRAC = _percent_to_fraction(STOP_LOSS_LONG_PERCENT)
_SL_SHORT_FRAC = _percent_to_fraction(STOP_LOSS_SHORT_PERCENT)

# === ФУНКЦИЯ расчёта P&L при закрытии позиции ===
def calculate_pnl(entry_price: float, exit_price: float, position_type: str) -> float:
    """Рассчитывает процент прибыли или убытка."""
    if position_type == "long":
        return (exit_price - entry_price) / entry_price * 100
    elif position_type == "short":
        return (entry_price - exit_price) / entry_price * 100
    else:
        return 0.0

# Функция проверки фиксации прибыли
def check_profit_take(df: pd.DataFrame, figi: str, virtual_positions: dict) -> dict:
    pos = virtual_positions.get(figi)
    if not pos or pos.get("type") not in ("long", "short"):
        return {"triggered": False}
    entry_price = pos["entry_price"]
    entry_time = pos.get("entry_time")
    if entry_time is None:
        return {"triggered": False}
    # Конвертируем entry_time в Timestamp
    try:
        entry_time = pd.to_datetime(entry_time, utc=True)
    except Exception:
        logger.error(f"Некорректное entry_time для {figi}: {entry_time}")
        return {"triggered": False}
    if entry_time not in df.index:
        # Ищем ближайшую свечу
        time_diffs = (df.index - entry_time).abs()
        if time_diffs.min().total_seconds() > 300:
            logger.warning(f"entry_time слишком далеко от данных для {figi}")
            return {"triggered": False}
        entry_time = df.index[time_diffs.argmin()]
    df_slice = df.loc[entry_time:]
    
    # Фиксация при long
    if pos["type"] == "long":
        # Ищем первую свечу, где цена достигла Bollinger High
        touch = df_slice[df_slice["high"] >= df_slice["bollinger_high"]]
        if not touch.empty:
            idx = touch.index[0]
            exit_price = df_slice.loc[idx, "high"]
            profit_pct = calculate_pnl(entry_price, exit_price, "long")
            return {
                "triggered": True,
                "reason": "достижение Bollinger High",
                "exit_price": exit_price,
                "profit_pct": profit_pct
            }
        # Или прибыль >= 1.2%
        for idx, row in df_slice.iterrows():
            if (row["high"] - entry_price) / entry_price * 100 >= 1.2:
                exit_price = row["high"]
                profit_pct = calculate_pnl(entry_price, exit_price, "long")
                return {
                    "triggered": True,
                    "reason": "прибыль >= 1.2%",
                    "exit_price": exit_price,
                    "profit_pct": profit_pct
                }
    else:  # Фиксация при  short
        # Ищем первую свечу, где цена достигла Bollinger Low
        touch = df_slice[df_slice["low"] <= df_slice["bollinger_low"]]
        if not touch.empty:
            idx = touch.index[0]
            exit_price = df_slice.loc[idx, "low"]
            profit_pct = calculate_pnl(entry_price, exit_price, "short")
            return {
                "triggered": True,
                "reason": "достижение Bollinger Low",
                "exit_price": exit_price,
                "profit_pct": profit_pct
            }
        # Или прибыль >= 1.2%    
        for idx, row in df_slice.iterrows():
            if (entry_price - row["low"]) / entry_price * 100 >= 1.2:
                exit_price = row["low"]
                profit_pct = calculate_pnl(entry_price, exit_price, "short")
                return {
                    "triggered": True,
                    "reason": "прибыль >= 1.2%",
                    "exit_price": exit_price,
                    "profit_pct": profit_pct
                }
    return {"triggered": False}

# Функция генерации сигналов
def generate_signals(
    df: pd.DataFrame,
    virtual_positions: dict,
    figi: str,
    ml_action: str | None = None,
    ml_confidence: float = 0.0,
    min_ml_confidence: float = 0.50,
) -> pd.DataFrame:
    required_cols = ["close", "open", "high", "low", "bollinger_low", "bollinger_high",
                     "rsi", "adx", "momentum", "chaikin_volatility"]
    if not all(col in df.columns for col in required_cols):
        logger.error(f"Отсутствуют колонки для {figi}: {set(required_cols) - set(df.columns)}")
        return df
    if len(df) < 12:
        logger.warning(f"Мало данных для {figi}: {len(df)}")
        return df
    df = df.copy()
    i = len(df) - 1
    current_index = df.index[i]
    df["buy_signal"] = False
    df["sell_signal"] = False
    df["stop_loss"] = False
    pos_state = virtual_positions.get(figi, {})
    position = pos_state.get("type")
    entry_price = pos_state.get("entry_price", float('nan'))
    sl_breach_count = int(pos_state.get("sl_breach_count", 0))
    entry_time = pos_state.get("entry_time")  
    adx_thresholds = ADX_THRESHOLDS.get(figi, {"buy": ADX_DEFAULT_THRESHOLD, "sell": ADX_DEFAULT_THRESHOLD})
    mom_ch_th = MOMENTUM_CHAIKIN_THRESHOLDS.get(figi, MOMENTUM_CHAIKIN_DEFAULT)

    # === ВХОДЫ ===
    if i >= 2 and position is None:
        current = df.iloc[i]
        # BUY
        try:
            prev_red = all(
                df["close"].iloc[j] < df["open"].iloc[j] and df["low"].iloc[j] <= df["bollinger_low"].iloc[j]
                for j in [i-2, i-1]
            )
            rsi_cond = any(df.loc[df.index[j], "rsi"] < 35 for j in [i-2, i-1])
            adx_cond = current["adx"] >= adx_thresholds["buy"]
            mom_cond = current["momentum"] <= mom_ch_th["buy"][0]
            chaikin_cond = current["chaikin_volatility"] >= mom_ch_th["buy"][1]
            ml_buy_ok = (ml_action in (None, "buy")) or (ml_action == "hold" and ml_confidence < min_ml_confidence)
            if prev_red and rsi_cond and adx_cond and mom_cond and chaikin_cond and ml_buy_ok:
                df.loc[current_index, "buy_signal"] = True
                entry_price = float(current["close"])
                virtual_positions[figi] = {
                    "type": "long",
                    "entry_price": entry_price,
                    "sl_breach_count": 0,
                    "entry_time": current_index.isoformat(),
                }
                logger.info(f"Сигнал BUY для {figi}: {entry_price:.2f} | Mom={current['momentum']:.2f}, CHV={current['chaikin_volatility']:.1f}")
        except Exception as e:
            logger.warning(f"Ошибка BUY для {figi}: {e}")
        # SELL
        try:
            prev_green = all(
                df["close"].iloc[j] > df["open"].iloc[j] and df["high"].iloc[j] >= df["bollinger_high"].iloc[j]
                for j in [i-2, i-1]
            )
            rsi_cond = any(df.loc[df.index[j], "rsi"] > 65 for j in [i-2, i-1])
            adx_cond = current["adx"] >= adx_thresholds["sell"]
            mom_cond = current["momentum"] >= mom_ch_th["sell"][0]
            chaikin_cond = current["chaikin_volatility"] <= mom_ch_th["sell"][1]
            ml_sell_ok = (ml_action in (None, "sell")) or (ml_action == "hold" and ml_confidence < min_ml_confidence)
            if prev_green and rsi_cond and adx_cond and mom_cond and chaikin_cond and ml_sell_ok:
                df.loc[current_index, "sell_signal"] = True
                entry_price = float(current["close"])
                virtual_positions[figi] = {
                    "type": "short",
                    "entry_price": entry_price,
                    "sl_breach_count": 0,
                    "entry_time": current_index.isoformat(),
                }
                logger.info(f"Сигнал SELL для {figi}: {entry_price:.2f} | Mom={current['momentum']:.2f}, CHV={current['chaikin_volatility']:.1f}")
        except Exception as e:
            logger.warning(f"Ошибка SELL для {figi}: {e}")

    # === СТОП-ЛОСС ===
    if figi in virtual_positions and position is not None and not pd.isna(entry_price):
        try:
            close_i = float(df["close"].iloc[i])
            if position == "long":
                stop_level = entry_price * (1 - _SL_LONG_FRAC)
                breached = close_i <= stop_level
            else:
                stop_level = entry_price * (1 + _SL_SHORT_FRAC)
                breached = close_i >= stop_level
            if breached:
                sl_breach_count += 1
            else:
                sl_breach_count = 0
            virtual_positions[figi]["sl_breach_count"] = sl_breach_count
            if sl_breach_count >= 2:
                df.loc[current_index, "stop_loss"] = True
                
                # УДАЛЕНИЕ ПОЗИЦИИ 
                loss_pct = calculate_pnl(entry_price, close_i, position)
                logger.info(f"СТОП-ЛОСС {figi} ({position}): {close_i:.2f}, убыток {loss_pct:.2f}%")
        except Exception as e:
            logger.error(f"Ошибка стоп-лосс для {figi}: {e}")

    # Логирование отсутствия сигналов
    if not df["buy_signal"].iloc[-1] and not df["sell_signal"].iloc[-1] and not df["stop_loss"].iloc[-1]:
        logger.debug(f"Для {figi}: нет сигналов на свече {df.index[-1]} — ADX={df['adx'].iloc[-1]:.1f}, RSI={df['rsi'].iloc[-1]:.1f}, Mom={df['momentum'].iloc[-1]:.2f}, Chaikin={df['chaikin_volatility'].iloc[-1]:.1f}")
    return df