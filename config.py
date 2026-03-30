# Конфигурационный файл

from dotenv import load_dotenv
import os

load_dotenv()

# Данные по API T-invest
API_TOKEN = os.getenv("TINKOFF_TOKEN")
ACCOUNT_ID = os.getenv("ACCOUNT_ID")

NAME = os.getenv("NAME", "Автоматизированный торговый бот через API Т-Инвестиции")

# Данные по API Телеграм
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Идентификаторы инструментов (Фиги)
FIGI_TO_TICKER = {
    "TCS80A107UL4": "T",
    "TCS00A107T19": "YDEX",
    "BBG000R607Y3": "PLZL",
    "BBG0063FKTD9": "LENT",
    "BBG004730N88": "SBER",
    "BBG004730ZJ9": "VTBR",
    "BBG004730JJ5": "MOEX",
    "BBG000R04X57": "FLOT",
    "TCS00A0JPP37": "UGLD",
    "BBG002458LF8": "SELG",
    "TCS00A108GD8": "IVAT",
    "BBG004S68B31": "ALRS",
}

TIMEFRAME = "15m"
LIMIT = int(os.getenv("LIMIT", 100))

# Параметры индикаторов
BOLLINGER_WINDOW = int(os.getenv("BOLLINGER_WINDOW", 12))
BOLLINGER_WINDOW_DEV = int(os.getenv("BOLLINGER_WINDOW_DEV", 2))
RSI_WINDOW = int(os.getenv("RSI_WINDOW", 9))
ADX_WINDOW = int(os.getenv("ADX_WINDOW", 12))
MOMENTUM_WINDOW = int(os.getenv("MOMENTUM_WINDOW", 10))
CHAIKIN_VOLATILITY_WINDOW = int(os.getenv("CHAIKIN_VOLATILITY_WINDOW", 10))
CHAIKIN_VOLATILITY_ROC_PERIOD = int(os.getenv("CHAIKIN_VOLATILITY_ROC_PERIOD", 10))  

# Индивидуальные пороги ADX
ADX_THRESHOLDS = {
    "TCS80A107UL4": {"buy": 27, "sell": 27}, # T
    "TCS00A107T19": {"buy": 23, "sell": 23}, # YDEX
    "BBG000R607Y3": {"buy": 37, "sell": 37}, # Полюс
    "BBG0063FKTD9": {"buy": 36, "sell": 36}, # Лента
    "BBG004730N88": {"buy": 30, "sell": 30}, # Сбер
    "BBG004730ZJ9": {"buy": 33, "sell": 33}, # ВТБ
    "BBG004730JJ5": {"buy": 28, "sell": 28}, # МосБиржа
    "BBG000R04X57": {"buy": 27, "sell": 27}, # СовкомФлот
    "TCS00A0JPP37": {"buy": 27, "sell": 27}, # ЮжУралЗолото
    "BBG002458LF8": {"buy": 25, "sell": 25}, # Селигдар
    "TCS00A108GD8": {"buy": 30, "sell": 30}, # ИВА Технология
    "BBG004S68B31": {"buy": 23, "sell": 23}, # Алроса
}
ADX_DEFAULT_THRESHOLD = 27

# Формат: "buy": (momentum <= X, chaikin >= Y), "sell": (momentum >= A, chaikin <= B)
MOMENTUM_CHAIKIN_THRESHOLDS = {
    "TCS80A107UL4": {"buy": (0, 100), "sell": (0, -25)},        # T
    "TCS00A107T19": {"buy": (0, 100), "sell": (0, -25)},        # YDEX
    "BBG000R607Y3": {"buy": (0, 60), "sell": (0, -25)},         # Полюс
    "BBG0063FKTD9": {"buy": (0, 100), "sell": (0, -25)},        # Лента
    "BBG004730N88": {"buy": (0, 100), "sell": (0, -25)},        # Сбер
    "BBG004730ZJ9": {"buy": (0, 100), "sell": (0, -25)},        # ВТБ
    "BBG004730JJ5": {"buy": (0, 70), "sell": (0, -25)},         # МосБиржа
    "BBG000R04X57": {"buy": (0, 60), "sell": (0, -25)},         # СовкомФлот
    "TCS00A0JPP37": {"buy": (0, 60), "sell": (0, -25)},         # ЮжУралЗолото
    "BBG002458LF8": {"buy": (0, 100), "sell": (0, -25)},        # Селигдар
    "TCS00A108GD8": {"buy": (0, 60), "sell": (0, -25)},         # ИВА Технология
    "BBG004S68B31": {"buy": (0, 60), "sell": (0, -25)},         # Алроса
}

# Значения по умолчанию (если инструмент не указан)
MOMENTUM_CHAIKIN_DEFAULT = {
    "buy": (0, 60),   # momentum <= 0 и chaikin >= 60
    "sell": (0, -20)   # momentum >= 0 и chaikin <= 0
}

STOP_LOSS_LONG_PERCENT = float(os.getenv("STOP_LOSS_LONG_PERCENT", 1))
STOP_LOSS_SHORT_PERCENT = float(os.getenv("STOP_LOSS_SHORT_PERCENT", 1))