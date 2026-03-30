# Получение исторических свечей для инициализации стримовой обработки

from t_tech.invest import AsyncClient, CandleInterval
import pandas as pd
from datetime import datetime, timedelta
from config import API_TOKEN, FIGI_TO_TICKER, TIMEFRAME
import logging

# Настройка логирования
logger = logging.getLogger(__name__)

# Словарь длительности интервалов в минутах
INTERVAL_DURATION = {
    "1m": 1, "5m": 5, "15m": 15, "1h": 60, "1d": 1440,
}

# Маппинг таймфреймов на enum CandleInterval из Tinkoff API
INTERVAL_MAPPING = {
    "1m": CandleInterval.CANDLE_INTERVAL_1_MIN,
    "5m": CandleInterval.CANDLE_INTERVAL_5_MIN,
    "15m": CandleInterval.CANDLE_INTERVAL_15_MIN,
    "1h": CandleInterval.CANDLE_INTERVAL_HOUR,
    "1d": CandleInterval.CANDLE_INTERVAL_DAY,
}

# Установка интервала свечей и его длительности
CANDLE_INTERVAL = INTERVAL_MAPPING.get(TIMEFRAME, CandleInterval.CANDLE_INTERVAL_15_MIN)
INTERVAL_MINUTES = INTERVAL_DURATION.get(TIMEFRAME, 15)

async def fetch_initial_candles(figi, n=100):
    """
    Получает последние n свечей для заданного FIGI из Tinkoff API.
    Возвращает DataFrame с колонками: timestamp, open, high, low, close, volume
    """
    try:
        # Асинхронный клиент для запроса к API
        async with AsyncClient(API_TOKEN) as client:
            end_time = datetime.utcnow()
            # Запрашиваем двойной диапазон для надежности
            start_time = end_time - timedelta(minutes=n * INTERVAL_MINUTES * 2)
            ticker = FIGI_TO_TICKER.get(figi, "Unknown Ticker")
            logger.info(f"[fetch_initial_candles] Запрос исторических данных для {ticker} ({figi}): {start_time} - {end_time}, n={n}, interval={INTERVAL_MINUTES} минут")
            
            # Запрос свечей через MarketDataService.GetCandles
            candles = (await client.market_data.get_candles(
                figi=figi,
                from_=start_time,
                to=end_time,
                interval=CANDLE_INTERVAL
            )).candles

            logger.info(f"[fetch_initial_candles] Получено {len(candles)} свечей для {ticker} ({figi})")
            
            # Логирование временного диапазона свечей
            if candles:
                first_candle_time = candles[0].time
                last_candle_time = candles[-1].time
                logger.info(f"[fetch_initial_candles] {ticker}: диапазон свечей: {first_candle_time} - {last_candle_time}")
            else:
                logger.warning(f"[fetch_initial_candles] {ticker}: НЕТ СВЕЧЕЙ в запрошенном диапазоне!")
            
            # Преобразование данных в DataFrame
            df = pd.DataFrame([{
                "timestamp": candle.time,
                "open": candle.open.units + candle.open.nano / 1e9,
                "high": candle.high.units + candle.high.nano / 1e9,
                "low": candle.low.units + candle.low.nano / 1e9,
                "close": candle.close.units + candle.close.nano / 1e9,
                "volume": candle.volume
            } for candle in candles])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            # Установка timestamp как индекса и сортировка
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)
            
            # Ограничение до последних n свечей
            if len(df) > n:
                df = df.iloc[-n:]
            logger.info(f"[fetch_initial_candles] После обрезки: {len(df)} свечей для {ticker} ({figi})")
            return df
    except Exception as e:
        logger.error(f"Ошибка для FIGI {figi}: {str(e)}")
        return None
