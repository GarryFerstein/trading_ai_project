# Trading_ai_project

Гибридный проект для российского рынка (T-Invest):
- **Transformer** извлекает латентные признаки из OHLCV-последовательностей;
- **RL (PPO)** принимает торговые решения (`hold/buy/sell`);
- правила из `signals.py` остаются основой и могут фильтроваться ML-моделью.

## Структура

- `config.py` - конфиг live-бота и индикаторов
- `config_ml.py` - конфиг офлайн-обучения
- `data_fetcher.py` - загрузка свечей из T-Invest
- `indicators.py` - Bollinger/RSI/ADX/Momentum/Chaikin
- `signals.py` - сигнальная логика (дополнена ML-фильтром)
- `telegram_notifier.py` - уведомления
- `main.py` - единая точка входа (`live/train/predict`)
- `models/transformer.py` - Transformer extractor
- `models/rl_agent.py` - PPO агент
- `environment/trading_env.py` - торговая среда
- `ml/data_processor.py` - подготовка датасета
- `ml/trainer.py` - обучение Transformer + PPO
- `ml/predictor.py` - инференс гибридной модели
- `data/historical_data.csv` - датасет
- `saved_models/` - сохраненные веса и артефакты

## Формат датасета

`historical_data.csv` должен содержать колонки:
- `timestamp`
- `open`
- `high`
- `low`
- `close`
- `volume`

## Как начать

1. Установить зависимости:
   - `pip install -r requirements.txt`
2. Положить исторические данные в `data/historical_data.csv`.
3. Обучить модель:
   - `python main.py --mode train`
4. Проверить единичный прогноз:
   - `python main.py --mode predict`
5. Запустить live-бота:
   - `python main.py --mode live`

Если в `saved_models/` есть обученные веса, `main.py --mode live` автоматически подключит ML+RL фильтр к правилам сигналов.
