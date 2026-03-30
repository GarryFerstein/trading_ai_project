# Индикаторы

import pandas as pd
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator, ROCIndicator 
from ta.trend import ADXIndicator
from config import (
    BOLLINGER_WINDOW, BOLLINGER_WINDOW_DEV, RSI_WINDOW, ADX_WINDOW,
    MOMENTUM_WINDOW, CHAIKIN_VOLATILITY_WINDOW, CHAIKIN_VOLATILITY_ROC_PERIOD
)
import logging
import time

# Настройка логирования
logger = logging.getLogger(__name__)
_last_warning_count = {}
_last_warning_time = {}

# Функция Индикатора волатильности Чайкина 
def chaikin_volatility(high, low, window=10, roc_period=10):
    hl = high - low
    ema_hl = hl.ewm(span=window, adjust=False).mean()
    chv = ema_hl.pct_change(periods=roc_period) * 100
    return chv

# Функция расчета индикаторов
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
    MIN_REQUIRED = max(
        BOLLINGER_WINDOW, RSI_WINDOW, ADX_WINDOW, MOMENTUM_WINDOW,
        CHAIKIN_VOLATILITY_WINDOW + CHAIKIN_VOLATILITY_ROC_PERIOD
    )
    
    OPTIMAL_BARS = MIN_REQUIRED + 10
    now = time.time()
    if len(df) < OPTIMAL_BARS:
        key = id(df)
        last_count = _last_warning_count.get(key)
        last_time = _last_warning_time.get(key, 0)
        if last_count != len(df) or (now - last_time) > 600:
            if len(df) < MIN_REQUIRED:
                logger.warning(f"Мало данных для индикаторов: {len(df)} свечей (нужно ≥ {MIN_REQUIRED})")
            _last_warning_count[key] = len(df)
            _last_warning_time[key] = now
    if len(df) < MIN_REQUIRED:
        return df
    
    # Bollinger Bands
    bb = BollingerBands(close=df["close"], window=BOLLINGER_WINDOW, window_dev=BOLLINGER_WINDOW_DEV)
    df["bollinger_high"] = bb.bollinger_hband()
    df["bollinger_low"] = bb.bollinger_lband()
    df["bollinger_mid"] = bb.bollinger_mavg()
    logger.debug(f"Bollinger рассчитаны: window={BOLLINGER_WINDOW}, dev={BOLLINGER_WINDOW_DEV}")
    
    # RSI
    rsi = RSIIndicator(close=df["close"], window=RSI_WINDOW)
    df["rsi"] = rsi.rsi()
    logger.debug(f"RSI рассчитан: window={RSI_WINDOW}, последнее значение={df['rsi'].iloc[-1]:.2f}")
    
    # ADX
    adx_ind = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=ADX_WINDOW)
    df["adx"] = adx_ind.adx()
    logger.debug(f"ADX рассчитан: window={ADX_WINDOW}, последнее значение={df['adx'].iloc[-1]:.2f}")
    
    # Momentum → теперь ROCIndicator
    roc = ROCIndicator(close=df["close"], window=MOMENTUM_WINDOW)
    df["momentum"] = roc.roc()
    logger.debug(f"Momentum (ROC) рассчитан: window={MOMENTUM_WINDOW}, значение={df['momentum'].iloc[-1]:.2f}")
    
    # Chaikin Volatility
    df["chaikin_volatility"] = chaikin_volatility(
        high=df["high"],
        low=df["low"],
        window=CHAIKIN_VOLATILITY_WINDOW,
        roc_period=CHAIKIN_VOLATILITY_ROC_PERIOD
    )
    logger.debug(f"Chaikin Volatility рассчитана: win={CHAIKIN_VOLATILITY_WINDOW}, roc={CHAIKIN_VOLATILITY_ROC_PERIOD}, значение={df['chaikin_volatility'].iloc[-1]:.2f}")

    # Удаляем NaN
    df.dropna(subset=[
        "bollinger_high", "bollinger_low", "rsi", "adx",
        "momentum", "chaikin_volatility"
    ], inplace=True)
    logger.info(f"Индикаторы рассчитаны (ROC=Momentum, Chaikin): {len(df)} свечей")
    return df