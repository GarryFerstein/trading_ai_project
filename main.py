# Главный модуль запуска бота для уведомлений.
# Этот модуль управляет подключением к Tinkoff Invest API, обработкой рыночных данных,
# генерацией торговых сигналов и отправкой уведомлений в Telegram.

import argparse
import asyncio
import logging
import random
from datetime import datetime, timedelta, timezone
import pandas as pd
from t_tech.invest import AsyncClient, CandleInterval, CandleInstrument, InfoInstrument
from t_tech.invest.market_data_stream.async_market_data_stream_manager import AsyncMarketDataStreamManager
from telegram_notifier import TelegramNotifier
from data_fetcher import fetch_initial_candles
from indicators import calculate_indicators
from signals import generate_signals, check_profit_take  
from config import (
    API_TOKEN, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, FIGI_TO_TICKER,
    TIMEFRAME, BOLLINGER_WINDOW, RSI_WINDOW, ADX_WINDOW,
    MOMENTUM_WINDOW, CHAIKIN_VOLATILITY_WINDOW,
    STOP_LOSS_LONG_PERCENT, STOP_LOSS_SHORT_PERCENT
)
import numpy as np
import warnings
import time
import asyncio as _asyncio
import hashlib
import os
import sys
import json
import socket
import re
try:
    import grpc
except Exception:
    grpc = None

# Игнорирование предупреждений protobuf
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")

# Настройка логирования
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("tinkoff.invest.logging").setLevel(logging.WARNING)
    logging.getLogger("indicators").setLevel(logging.DEBUG)
    logging.getLogger("signals").setLevel(logging.DEBUG)

setup_logging()
logger = logging.getLogger(__name__)

# Инициализация Telegram notifier
telegram_notifier = None  # будет инициализирована в main с обработчиком команд

# Глобальные структуры
candles_data = {}  # Хранение свечей по FIGI
virtual_positions = {}  # Хранение активных виртуальных позиций
sent_signals = set()  # Отправленные сигналы
sent_profit_notifications = set()  # Уведомления о фиксации прибыли
sent_stoploss_notifications = set()  # Уведомления о стоп-лоссе
last_status = {}  # Последние статусы торгов
last_trade_ts = {}  # Timestamps последних торгов
error_notification_timestamps = {}  # Таймеры для уведомлений об ошибках
processed_candles = set()  # Обработанные свечи
instrument_names = {}  # FIGI -> полное имя инструмента
hybrid_predictor = None  # ML+RL фильтр для сигналов (опционально)

# Файлы
POSITIONS_FILE = "virtual_positions.json"  # Файл для сохранения виртуальных позиций
PNL_FILE = "total_pnl.json"                # Файл для сохранения сделок
SENT_SIGNALS_FILE = "sent_signals.json"    # Файл для сохранения сигналов

# === СОХРАНЕНИЕ ===
def load_virtual_positions():
    global virtual_positions
    try:
        with open(POSITIONS_FILE, 'r') as f:
            virtual_positions = json.load(f)
        # Удаляем entry_index (он больше не используется)
        for figi, data in virtual_positions.items():
            if "entry_index" in data:
                del data["entry_index"]
        logger.info(f"Загружены позиции: {len(virtual_positions)}")
    except FileNotFoundError:
        virtual_positions = {}
    except Exception as e:
        logger.error(f"Ошибка загрузки позиций: {e}")

# Сохранение виртуальных позиций
def save_virtual_positions():
    try:
        with open(POSITIONS_FILE, 'w') as f:
            json.dump(virtual_positions, f)
    except Exception as e:
        logger.error(f"Ошибка сохранения позиций: {e}")

def load_pnl():
    try:
        with open(PNL_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

# Сохранение сделок
def save_pnl(trades):
    try:
        with open(PNL_FILE, 'w') as f:
            json.dump(trades, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Ошибка сохранения PnL: {e}")

def load_sent_signals():
    global sent_signals, sent_profit_notifications, sent_stoploss_notifications
    try:
        with open(SENT_SIGNALS_FILE, 'r') as f:
            data = json.load(f)
            sent_signals = {tuple(x) for x in data.get("signals", [])}
            sent_profit_notifications = {tuple(x) for x in data.get("profit", [])}
            sent_stoploss_notifications = {tuple(x) for x in data.get("stoploss", [])}
    except:
        pass

# Сохранение сигналов
def save_sent_signals():
    try:
        data = {
            "signals": [list(x) for x in sent_signals],
            "profit": [list(x) for x in sent_profit_notifications],
            "stoploss": [list(x) for x in sent_stoploss_notifications]
        }
        with open(SENT_SIGNALS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Ошибка сохранения сигналов: {e}")

# gRPC keepalive опции для стабильного соединения
GRPC_OPTIONS = [
    ("grpc.keepalive_time_ms", 30000),
    ("grpc.keepalive_timeout_ms", 15000),
    ("grpc.keepalive_permit_without_calls", 1),
    ("grpc.http2.max_pings_without_data", 0),
    ("grpc.http2.min_time_between_pings_ms", 10000),
    ("grpc.http2.min_ping_interval_without_data_ms", 5000),
    ("grpc.max_connection_idle_ms", 600000),
    ("grpc.max_connection_age_ms", 3600000),
]

# Функция проверки интернета
def check_internet():
    try:
        socket.create_connection(("ya.ru", 443), timeout=5)
        return True
    except OSError:
        return False

# Проверка, можно ли отправить уведомление об ошибке
def can_send_error_notification(msg):
    now = time.time()
    last = error_notification_timestamps.get(msg, 0)
    if now - last > 300:
        error_notification_timestamps[msg] = now
        return True
    return False

# Создание хэша свечи
def get_candle_hash(figi, ts, row):
    data = f"{figi}{ts}{row['open']}{row['high']}{row['low']}{row['close']}{row['volume']}"
    return hashlib.md5(data.encode()).hexdigest()

# Очистка сигналов старше 48 часов
def clean_old_signals():
    threshold = datetime.now(timezone.utc) - timedelta(hours=48)
    def filter_set(s):
        filtered = set()
        for item in s:
            if len(item) != 4: continue
            figi, ts_str, _, _ = item
            try:
                ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                if ts > threshold or figi in virtual_positions:
                    filtered.add(item)
            except: continue
        return filtered
    global sent_signals, sent_profit_notifications, sent_stoploss_notifications
    sent_signals = filter_set(sent_signals)
    sent_profit_notifications = filter_set(sent_profit_notifications)
    sent_stoploss_notifications = filter_set(sent_stoploss_notifications)

# Периодическое логирование активности бота (каждые 300 секунд)
async def heartbeat_task():
    while True:
        logger.info(f"HEARTBEAT: бот жив {datetime.now(timezone.utc).strftime('%H:%M:%S')}")
        await asyncio.sleep(300)

# =============================================
# ДОБАВЛЕНО: Фоновая задача — пересканирование каждые 15 минут
# =============================================
async def periodic_full_rescan():
    logger.info("Запущена фоновая задача: пересканирование каждые 15 минут")
    while True:
        await asyncio.sleep(900)  # 15 минут
        logger.info("Запуск периодического пересканирования всех инструментов...")
        for figi, ticker in FIGI_TO_TICKER.items():
            df = candles_data.get(figi)
            if df is not None and len(df) >= 30:
                try:
                    await process_signals(figi, ticker, None)
                except Exception as e:
                    logger.warning(f"Ошибка пересканирования {ticker}: {e}")
        logger.info("Периодическое пересканирование завершено")

# =============================================
# Функция process_signals — с корректной обработкой (без дублирования вызова generate_signals)
# =============================================
async def process_signals(figi, ticker, client):
    df = candles_data.get(figi)
    if df is None or df.empty:
        return
    min_req = max(BOLLINGER_WINDOW, RSI_WINDOW, ADX_WINDOW, MOMENTUM_WINDOW, CHAIKIN_VOLATILITY_WINDOW)
    if len(df) < min_req:
        return
    df = calculate_indicators(df)
    if df.empty:
        return
    ml_action = None
    ml_confidence = 0.0
    if hybrid_predictor is not None:
        try:
            tmp_df = df.reset_index().rename(columns={"index": "timestamp"})
            pred = hybrid_predictor.predict_action(
                df=tmp_df[["timestamp", "open", "high", "low", "close", "volume"]],
                cash=100000.0,
                position=0.0,
            )
            ml_action = pred.get("action_name")
            ml_confidence = float(pred.get("confidence", 0.0))
        except Exception as e:
            logger.debug(f"ML-предиктор недоступен для {ticker}: {e}")

    # ЕДИНСТВЕННЫЙ вызов generate_signals
    signals_df = generate_signals(
        df,
        virtual_positions,
        figi,
        ml_action=ml_action,
        ml_confidence=ml_confidence,
    )
    if len(signals_df) == 0:
        return
    # Обрабатываем только последнюю свечу
    ts = signals_df.index[-1]
    row = signals_df.iloc[-1]
    candle_key = (figi, ts.isoformat(), get_candle_hash(figi, ts.isoformat(), row))
    # Проверка актуальности свечи
    tf_sec = {"1m":60,"5m":300,"15m":900,"1h":3600,"1d":86400}.get(TIMEFRAME,900)
    if (datetime.now(timezone.utc) - ts).total_seconds() > tf_sec * 1.5:
        return
    if candle_key in processed_candles:
        return
    display_name = instrument_names.get(figi, FIGI_TO_TICKER.get(figi, figi))
    paren_name = FIGI_TO_TICKER.get(figi, figi)
    pos_type = virtual_positions.get(figi, {}).get("type")
    entry_sent = False

    # Покупка
    buy_key = (figi, ts.isoformat(), "buy", round(row.get("close", 0), 2))
    if not entry_sent and row.get("buy_signal") and buy_key not in sent_signals:
        msg = f"🟢Покупка: {display_name} ({paren_name})\nЦена: {row['close']:.2f} RUB\nВремя: {ts}\nПричина: Технический"
        await telegram_notifier.send_message(msg)
        sent_signals.add(buy_key)
        logger.info(f"🟢Покупка {display_name}")
        entry_sent = True

    # Продажа
    sell_key = (figi, ts.isoformat(), "sell", round(row.get("close", 0), 2))
    if not entry_sent and row.get("sell_signal") and sell_key not in sent_signals:
        msg = f"🔴Продажа: {display_name} ({paren_name})\nЦена: {row['close']:.2f} RUB\nВремя: {ts}\nПричина: Технический"
        await telegram_notifier.send_message(msg)
        sent_signals.add(sell_key)
        logger.info(f"🔴Продажа {display_name}")
        entry_sent = True

    # === СТОП-ЛОСС: 🚫 + сохранение в PnL ===
    stop_key = (figi, ts.isoformat(), "stop_loss", round(row.get("close", 0), 2))
    if row.get("stop_loss") and stop_key not in sent_stoploss_notifications:
        pos = virtual_positions.get(figi)
        if pos:
            entry_price = pos["entry_price"]
            exit_price = float(row["close"])
            # Импортируем из signals.py корректную функцию расчёта
            from signals import calculate_pnl
            loss_pct = calculate_pnl(entry_price, exit_price, pos["type"])
            display_name = instrument_names.get(figi, FIGI_TO_TICKER.get(figi, figi))
            paren_name = FIGI_TO_TICKER.get(figi, figi)
            # 🚫 Стандартное уведомление
            msg_stop = f"🚫Стоп-лосс: {display_name} ({paren_name})\nЦена: {exit_price:.2f} RUB\nВремя: {ts}"
            await telegram_notifier.send_message(msg_stop)
            # 🚫 Уведомление с расчётом убытка
            emoji = "🚫" if loss_pct < 0 else "💰"
            msg_loss = (
                f"{emoji}Закрытие ({'long' if pos['type']=='long' else 'short'}): {display_name} ({paren_name})\n"
                f"Вход: {entry_price:.2f} RUB\n"
                f"Выход: {exit_price:.2f} RUB\n"
                f"Причина: Стоп-лосс\n"
                f"{'Убыток' if loss_pct < 0 else 'Прибыль'}: {loss_pct:.2f}%\n"
                f"Время: {ts}"
            )
            await telegram_notifier.send_message(msg_loss)
            sent_stoploss_notifications.add(stop_key)
            logger.info(f"🚫Стоп-лосс {display_name}: {loss_pct:.2f}%")

            # Сохраняем как сделку
            trade = {
                "figi": figi,
                "ticker": paren_name,
                "type": pos["type"],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "profit_pct": loss_pct,
                "reason": "Стоп-лосс",
                "entry_time": pos["entry_time"],
                "exit_time": ts.isoformat()
            }
            pnl = load_pnl()
            pnl.append(trade)
            save_pnl(pnl)
            
        # Удаляем позицию
        virtual_positions.pop(figi, None)

    # === ФИКСАЦИЯ ПРИБЫЛИ ===
    profit = check_profit_take(signals_df, figi, virtual_positions)
    if profit["triggered"] and figi in virtual_positions:
        pos = virtual_positions[figi]
        exit_price = profit["exit_price"]
        key = (figi, ts.isoformat(), f"profit_{pos['type']}", round(exit_price, 2))
        if key not in sent_profit_notifications:
            msg = (
                f"💰Прибыль ({'long' if pos['type']=='long' else 'short'}): {display_name} ({paren_name})\n"
                f"Выход: {exit_price:.2f} RUB\n"
                f"Причина: {profit['reason']}\n"
                f"💰Прибыль: {profit['profit_pct']:.2f}%\n"
                f"Время: {ts}"
            )
            await telegram_notifier.send_message(msg)
            sent_profit_notifications.add(key)
            logger.info(f"💰Прибыль {display_name}: {profit['profit_pct']:.2f}%")

            # Сохраняем сделку
            trade = {
                "figi": figi,
                "ticker": paren_name,
                "type": pos["type"],
                "entry_price": pos["entry_price"],
                "exit_price": exit_price,
                "profit_pct": profit['profit_pct'],
                "reason": profit["reason"],
                "entry_time": pos["entry_time"],
                "exit_time": ts.isoformat()
            }
            pnl = load_pnl()
            pnl.append(trade)
            save_pnl(pnl)
            virtual_positions.pop(figi, None)
    processed_candles.add(candle_key)
    save_virtual_positions()
    save_sent_signals()

# Безопасная отписка и повторная подписка для избежания дубликатов
async def _resubscribe(stream_manager, candle_instruments, info_instruments):    
    try:
        stream_manager.candles.unsubscribe(candle_instruments)
        stream_manager.info.unsubscribe(info_instruments)
        logger.info("Отписка при переподключении")
    except Exception as e:
        logger.debug(f"Отписка при переподключении дала ошибку (можно игнорировать): {e}")
    await asyncio.sleep(0.2)
    stream_manager.candles.subscribe(candle_instruments)
    logger.info("Повторная подписка на свечи выполнена")
    stream_manager.info.subscribe(info_instruments)
    logger.info("Повторная подписка на информацию выполнена")

# Проверка, является ли ошибка gRPC INTERNAL
def _is_internal_error(exc: Exception) -> bool:
    try:
        if grpc is not None and isinstance(exc, grpc.RpcError):
            return exc.code() == grpc.StatusCode.INTERNAL
    except Exception:
        pass
    s = str(exc)
    return ("StatusCode.INTERNAL" in s) or ("internal error" in s.lower()) or ("Internal error from Core" in s)

# Проверка на UNAVAILABLE ошибки (включая timeout и connection issues)
def _is_unavailable_error(exc: Exception) -> bool:
    s = str(exc).lower()
    return "unavailable" in s or "timed out" in s or "no route to host" in s

# Проверка на CANCELLED ошибки
def _is_cancelled_error(exc: Exception) -> bool:
    try:
        if grpc is not None and isinstance(exc, grpc.RpcError):
            return exc.code() == grpc.StatusCode.CANCELLED
    except Exception:
        pass
    s = str(exc).lower()
    return "cancelled" in s or "rst_stream" in s

# Обработчик стрима рыночных данных с реконнектом
async def market_data_stream_handler(stream_manager, client, candle_instruments, info_instruments):
    backoff = 60  
    max_backoff = 300
    last_status_notified = None
    last_data = datetime.now(timezone.utc)
    while True:
        try:
            async for resp in stream_manager:
                last_data = datetime.now(timezone.utc)
                backoff = 10  
                if resp.candle:
                    figi = resp.candle.figi
                    ticker = FIGI_TO_TICKER.get(figi, "Unknown")
                    candle = resp.candle
                    ts_val = (candle.last_trade_ts.replace(tzinfo=timezone.utc).timestamp()
                              if candle.last_trade_ts else datetime.now(timezone.utc).timestamp())
                    if figi in last_trade_ts and ts_val <= last_trade_ts[figi]:
                        continue
                    last_trade_ts[figi] = ts_val
                    row = {
                        "timestamp": candle.time.replace(tzinfo=timezone.utc),
                        "open": candle.open.units + candle.open.nano / 1e9,
                        "high": candle.high.units + candle.high.nano / 1e9,
                        "low": candle.low.units + candle.low.nano / 1e9,
                        "close": candle.close.units + candle.close.nano / 1e9,
                        "volume": candle.volume
                    }
                    df = candles_data.get(figi)
                    new_ts = row["timestamp"]
                    process = False
                    if df is None or df.empty:
                        df = pd.DataFrame([row]).set_index("timestamp")
                        process = True
                    else:
                        if new_ts in df.index:
                            if any(df.loc[new_ts][k] != row[k] for k in row if k != "timestamp"):
                                process = True
                            df.loc[new_ts] = row
                        else:
                            process = True
                            df.loc[new_ts] = row
                        if len(df) > 100:
                            df = df.iloc[-100:]
                    candles_data[figi] = df
                    if last_status.get(figi, "") not in ("NORMAL_TRADING", "SECURITY_TRADING_STATUS_NORMAL_TRADING"):
                        continue
                    if process:
                        clean_old_signals()
                        await process_signals(figi, ticker, client)
                elif resp.trading_status:
                    figi = resp.trading_status.figi
                    ticker = FIGI_TO_TICKER.get(figi, "Unknown")
                    status = resp.trading_status.trading_status.name
                    last_status[figi] = status
                    OPEN = ("NORMAL_TRADING", "SECURITY_TRADING_STATUS_NORMAL_TRADING", "DEALER_NORMAL_TRADING", "SESSION_OPEN")
                    all_st = list(last_status.values())
                    if all(s in OPEN for s in all_st) and last_status_notified != 'open':
                        await telegram_notifier.send_message("✅Рынок полностью открыт")
                        last_status_notified = 'open'
                    elif all(s not in OPEN for s in all_st) and last_status_notified != 'suspended':
                        await telegram_notifier.send_message("⛔️Рынок приостановлен")
                        last_status_notified = 'suspended'
                if (datetime.now(timezone.utc) - last_data).total_seconds() > 300 and last_status_notified != 'suspended':
                    msg = "Нет данных >5 мин — переподключение"
                    logger.warning(msg)
                    await telegram_notifier.send_message(msg)
                    raise RuntimeError("Dead stream")
        except Exception as e:
            if _is_internal_error(e) or _is_cancelled_error(e):
                await _resubscribe(stream_manager, candle_instruments, info_instruments)
                await asyncio.sleep(random.uniform(0, backoff))
                backoff = min(backoff * 2, max_backoff)
                continue
            if 'RESOURCE_EXHAUSTED' in str(e):
                await asyncio.sleep(60)
                continue
            if _is_unavailable_error(e):
                raise
            else:
                await _resubscribe(stream_manager, candle_instruments, info_instruments)
                await asyncio.sleep(random.uniform(0, backoff))
                backoff = min(backoff * 2, max_backoff)

TIMEFRAME_TO_ENUM = {
    "1m": CandleInterval.CANDLE_INTERVAL_1_MIN,
    "5m": CandleInterval.CANDLE_INTERVAL_5_MIN,
    "15m": CandleInterval.CANDLE_INTERVAL_15_MIN,
    "1h": CandleInterval.CANDLE_INTERVAL_HOUR,
    "1d": CandleInterval.CANDLE_INTERVAL_DAY,
}

# Получение названий инструментов через API
async def fetch_instrument_names(client):
    names = {}
    methods = ["shares", "bonds", "etfs", "futures", "currencies"]
    for method_name in methods:
        func = getattr(client.instruments, method_name, None)
        if not func:
            continue
        try:
            result = func()
            if _asyncio.iscoroutine(result):
                result = await result
            instruments = getattr(result, "instruments", []) or []
            for item in instruments:
                try:
                    names[item.figi] = item.name
                except:
                    continue
        except Exception as e:
            logger.debug(f"Ошибка загрузки {method_name}: {e}")
    return names

# === ФУНКЦИИ ФОРМИРОВАНИЯ ОТЧЁТОВ ===

def format_positions(virtual_positions: dict) -> str:
    if not virtual_positions:
        return "Нет активных позиций."
    lines = ["<b>Активные виртуальные позиции:</b>"]
    for figi, pos in virtual_positions.items():
        ticker = FIGI_TO_TICKER.get(figi, figi)
        entry = pos.get('entry_price', 'N/A')
        typ = pos.get('type', 'N/A')
        lines.append(f"• {ticker} ({typ}) @ {entry:.2f}")
    return "\n".join(lines)

def format_trades(pnl: list) -> str:
    if not pnl:
        return "Сделок пока нет."
    lines = ["<b>Последние сделки (макс. 10):</b>"]
    for trade in reversed(pnl[-10:]):
        ticker = trade.get('ticker', 'N/A')
        typ = trade.get('type', 'N/A')
        entry = trade.get('entry_price', 0)
        exit_p = trade.get('exit_price', 0)
        profit = trade.get('profit_pct', 0)
        reason = trade.get('reason', 'N/A')
        lines.append(
            f"• {ticker} ({typ}) | Вход: {entry:.2f} | Выход: {exit_p:.2f} | "
            f"{'✅' if profit >= 0 else '❌'} {profit:.2f}% | {reason}"
        )
    return "\n".join(lines)

def format_signals(sent_signals_set: set) -> str:
    sigs = list(sent_signals_set)
    if not sigs:
        return "Сигналы за последние 48ч не найдены."
    lines = ["<b>Последние сигналы (макс. 10):</b>"]
    for item in sigs[-10:]:
        figi, ts_str, sig_type, price = item
        ticker = FIGI_TO_TICKER.get(figi, figi)
        lines.append(f"• {ticker} | {sig_type.upper()} @ {price:.2f} | {ts_str.split('T')[1][:5]}")
    return "\n".join(lines)

# === ГЛОБАЛЬНОЕ СОБЫТИЕ для graceful shutdown ===
shutdown_event = asyncio.Event()

# === ОБРАБОТЧИК КОМАНД ===
async def handle_telegram_command(command: str, message):
    global shutdown_event
    logger.info(f"Получена команда: {command} от {message.chat.id}")

    # Только авторизованный чат
    if str(message.chat.id) != str(TELEGRAM_CHAT_ID):
        logger.warning(f"Отклонена команда от неразрешённого chat_id: {message.chat.id}")
        return

    if command == "/status":

        # Загружаем актуальные данные
        load_virtual_positions()
        pnl = load_pnl()
        load_sent_signals()

        msg = (
            "<b>📊 Статус торгового робота:</b>\n\n"
            + format_positions(virtual_positions) + "\n\n"
            + format_trades(pnl) + "\n\n"
            + format_signals(sent_signals)
        )
        await telegram_notifier.send_message(msg)

    elif command == "/stop_bot":
        await telegram_notifier.send_message("🛑 Получена команда на остановку. Завершаю работу...")
        shutdown_event.set()

    elif command == "/restart_bot":
        await telegram_notifier.send_message("🔄 Получена команда на перезапуск. Сохраняю состояние...")
        save_virtual_positions()
        save_sent_signals()
        await telegram_notifier.close()
        os.execv(sys.executable, [sys.executable] + sys.argv)

    elif command == "/start_bot":
        await telegram_notifier.send_message("✅ Бот уже активен. Используйте /status для проверки состояния.")

    else:
        await telegram_notifier.send_message("Неизвестная команда. Доступны: /status, /stop_bot, /restart_bot")

# Основная функция запуска live-бота
async def live_main():
    logger.info(f"🚀Бот запущен. Таймфрейм: {TIMEFRAME}")

    # Передаём обработчик команд в TelegramNotifier
    global telegram_notifier
    telegram_notifier = TelegramNotifier(
        TELEGRAM_TOKEN,
        TELEGRAM_CHAT_ID,
        command_handler=handle_telegram_command
    )
    await telegram_notifier.start()
    await telegram_notifier.send_message(
        f"🚀Бот запущен в {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}\n"
        "Доступные команды:\n"
        "• /status — состояние робота\n"
        "• /stop_bot — остановить\n"
        "• /restart_bot — перезапустить"
    )

    global hybrid_predictor
    if os.path.exists("saved_models/transformer.pth") and os.path.exists("saved_models/rl_agent.pth"):
        try:
            from ml.predictor import HybridPredictor

            hybrid_predictor = HybridPredictor()
            logger.info("ML+RL предиктор подключен к сигналам.")
        except Exception as e:
            logger.warning(f"Не удалось загрузить ML+RL предиктор: {e}")

    load_virtual_positions()
    load_sent_signals()

    while not shutdown_event.is_set():
        try:
            if not check_internet():
                await telegram_notifier.send_message("Нет интернета — ожидание")
                await asyncio.sleep(5)
                continue
            try:
                client_ctx = AsyncClient(API_TOKEN, options=GRPC_OPTIONS)
            except TypeError:
                client_ctx = AsyncClient(API_TOKEN)
            async with client_ctx as client:
                global instrument_names
                instrument_names = await fetch_instrument_names(client)
                logger.info(f"Загружено имён: {len(instrument_names)}")
                for figi, ticker in FIGI_TO_TICKER.items():
                    df = await fetch_initial_candles(figi, n=100)
                    last_trade_ts[figi] = 0.0
                    if df is not None and not df.empty:
                        if not isinstance(df.index, pd.DatetimeIndex):
                            df.set_index("timestamp", inplace=True)
                        df.index = df.index.tz_localize('UTC') if df.index.tz is None else df.index.tz_convert('UTC')
                        candles_data[figi] = df
                    else:
                        candles_data[figi] = pd.DataFrame().set_index(pd.DatetimeIndex([]))
                interval = TIMEFRAME_TO_ENUM.get(TIMEFRAME, CandleInterval.CANDLE_INTERVAL_15_MIN)
                candle_instr = [CandleInstrument(figi=f, interval=interval) for f in FIGI_TO_TICKER]
                info_instr = [InfoInstrument(figi=f) for f in FIGI_TO_TICKER]
                stream_mgr = AsyncMarketDataStreamManager(client.market_data_stream)
                try:
                    stream_mgr.candles.unsubscribe(candle_instr)
                    stream_mgr.info.unsubscribe(info_instr)
                except: pass
                stream_mgr.candles.subscribe(candle_instr)
                stream_mgr.info.subscribe(info_instr)
                tasks = [
                    asyncio.create_task(market_data_stream_handler(stream_mgr, client, candle_instr, info_instr)),
                    asyncio.create_task(heartbeat_task()),
                    asyncio.create_task(periodic_full_rescan()),
                    asyncio.create_task(shutdown_event.wait())  # ждём остановку
                ]
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
        except Exception as e:
            if shutdown_event.is_set():
                break
            msg = f"Критическая ошибка: {e}. Перезапуск..."
            logger.critical(msg)
            if can_send_error_notification(msg):
                await telegram_notifier.send_message(msg)
            await asyncio.sleep(random.uniform(1, 5))
        except KeyboardInterrupt:
            logger.info("Получен сигнал остановки (Ctrl+C)")
            stop_msg = "🛑 Бот остановлен (вручную)"
            try:
                await telegram_notifier.send_message(stop_msg)
            except Exception as e:
                logger.error(f"Не удалось отправить уведомление об остановке: {e}")
            shutdown_event.set()
            break
        finally:
            logger.info("Сохранение состояния перед выходом...")
            save_virtual_positions()
            save_sent_signals()
            try:
                await telegram_notifier.close()
            except Exception as e:
                logger.error(f"Ошибка при закрытии Telegram: {e}")

    logger.info("Бот завершил работу.")

def run_research_train():
    print("[train] Старт режима обучения (логи ниже появятся по мере этапов)", flush=True)
    logger.info("Загрузка модуля ml.trainer (может занять время при первом импорте torch)...")
    from ml.trainer import run_full_training

    logger.info("Запуск run_full_training()...")
    stats = run_full_training()
    logger.info(
        "Обучение завершено: final_equity=%.2f, pnl_pct=%.2f%%",
        stats["final_equity"],
        stats["pnl_pct"],
    )


def run_research_predict():
    from config_ml import DATA
    from ml.data_processor import load_ohlcv
    from ml.predictor import HybridPredictor

    predictor = HybridPredictor()
    df = load_ohlcv(DATA.csv_path)
    result = predictor.predict_action(df=df, cash=100000.0, position=0.0)
    logger.info(f"Inference: action={result['action_name']}, confidence={result['confidence']:.3f}, price={result['price']:.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Trading AI project")
    parser.add_argument(
        "--mode",
        choices=["live", "train", "predict"],
        default="live",
        help="live=стриминг T-Invest, train=обучение transformer+RL, predict=одиночный инференс",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"[trading_ai] Режим: {args.mode}", flush=True)
    if args.mode == "live":
        asyncio.run(live_main())
    elif args.mode == "train":
        run_research_train()
    else:
        run_research_predict()