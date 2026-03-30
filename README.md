# Trading_ai_project

Гибридный проект для российского рынка (T-Invest), объединяющий:
- **Transformer** (извлечение признаков из последовательностей OHLCV);
- **RL (PPO)** (выбор дискретного действия `hold/buy/sell`);
- вашу текущую **сигнальную стратегию** из `signals.py` (как основу для входов/выходов).

Ключевая идея: в live-режиме ML/RL используется как **дополнительный фильтр** перед тем, как выставлять `buy_signal`/`sell_signal` в `signals.py`, а не как полностью автономный торговый исполнитель.

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

## Offline: обучение ( `python main.py --mode train` )

Обучение выполняется из `ml/trainer.py` и включает 3 шага.

1. **Подготовка данных** (`ml/data_processor.py`)
   - загрузка CSV и сортировка по времени;
   - добавление производных фич (`log_return`, `hl_spread`, `oc_change`, `vol_z`);
   - формирование окон длиной `sequence_length`;
   - целевая переменная `target_return` (зависит от `target_horizon`).

2. **Transformer** (`models/transformer.py`)
   - получает окно `[sequence_length, num_features]`;
   - учится предсказывать следующую доходность (регрессия, MSE);
   - возвращает вектор скрытых признаков размерности `d_model`, который затем используется для RL.

3. **RL агент (PPO-подобная логика)** (`environment/trading_env.py`, `models/rl_agent.py`)
   - среда задаёт дискретные действия `hold/buy/sell`;
   - состояние включает `cash`, `position`, текущую цену и признаки из Transformer;
   - награда — изменение equity между шагами (с учётом комиссии).

После завершения обучения сохраняются веса и артефакты в `saved_models/`.

## Online: режим реального времени (`python main.py --mode live`)

Live-часть запускает стрим T-Invest, обновляет свечи и считает индикаторы (`indicators.py`).

Дальше на каждом шаге:
1. `process_signals()` формирует DataFrame с индикаторами;
2. если в `saved_models/` есть обученные веса, подключается `HybridPredictor` (`ml/predictor.py`);
3. `HybridPredictor` формирует последнее окно из `OHLCV+фич`, прогоняет его через Transformer и RL-политику, получая `ml_action` и `ml_confidence`;
4. `signals.py` применяет вашу rule-based логику, но входы может **блокировать** или **разрешать** в зависимости от ML/RL.

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

Перед запуском live убедитесь, что заданы переменные окружения:
- `TINKOFF_TOKEN` — токен T-Invest API
- `TELEGRAM_TOKEN` и `TELEGRAM_CHAT_ID` — для уведомлений

Если в `saved_models/` есть обученные веса, `main.py --mode live` автоматически подключит ML+RL фильтр к правилам сигналов.

### Артефакты обучения в `saved_models/`

- `transformer.pth` — веса Transformer
- `rl_agent.pth` — веса политики RL
- `scaler.json` — mean/std и порядок `feature_cols` (критично для корректного инференса)
- `training_stats.json` — итог roll-out RL на тестовом отрезке (например, `final_equity` и `pnl_pct`)
- `config_snapshot.json` — снимок гиперпараметров обучения
