# Telegram Ads Reach Forecast

## Описание проекта

Проект представляет собой ML-сервис для прогнозирования охвата (VIEWS) рекламных объявлений в Telegram на основе:

* CPM (стоимость за 1000 показов)
* канала размещения (CHANNEL_NAME)
* даты размещения (DATE)

Решение предназначено для data-driven медиапланирования и позволяет заранее оценивать эффективность рекламных кампаний.

---

## Структура репозитория

```
telegram_ads_forecast/
├── data/
│   └── data_train.csv        # Исторические данные (train dataset)
├── models/                   # НЕ хранится в Git (скачивается при старте)
├── src/
│   ├── api.py                # FastAPI приложение (API + web)
│   ├── features.py           # Feature engineering для inference
│   └── train.py              # Обучение модели
├── templates/
│   └── index.html            # Веб-страница (форма + CSV upload)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Данные

### Train dataset

Файл `data/data_train.csv` хранится в репозитории **намеренно**:

* для воспроизводимости обучения
* для демонстрации EDA и feature engineering

Используемые колонки:

* `CPM` — стоимость за 1000 показов
* `CHANNEL_NAME` — Telegram-канал размещения
* `DATE` — дата размещения
* `VIEWS` — целевая переменная (охват)
* `CLICKS`, `ACTIONS` — вспомогательные признаки

---

## Модель

Используется `CatBoostRegressor`:

* поддержка категориальных признаков (`CHANNEL_NAME`)
* устойчивая работа с нелинейными зависимостями CPM → VIEWS
* обучение в лог-пространстве (`log1p(VIEWS)`)

### Feature engineering

Основные признаки:

* `CPM`, `cpm_log`, `cpm_sq`
* временные: `dayofweek`, `week`, `month`, `is_weekend`
* статистики канала:

  * `channel_mean_views`
  * `channel_median_views`
  * `channel_history_count`
* `cpm_vs_channel_mean`
* `CHANNEL_NAME` (категориальный)

Логика генерации признаков вынесена в `src/features.py` и полностью совпадает между train и inference.

---

## API

### Single prediction (JSON)

**POST** `/predict`

```json
{
  "CPM": 120.5,
  "CHANNEL_NAME": "crypto_news",
  "DATE": "2024-12-01"
}
```

Ответ:

```json
{
  "VIEWS": 18450
}
```

---

### Single prediction (HTML form)

**POST** `/predict_form`

Используется веб-форма на главной странице.

---

### Batch prediction (CSV upload)

**POST** `/predict_csv`

#### Входной CSV (пример):

```csv
CPM,CHANNEL_NAME,DATE
120.5,crypto_news,2024-12-01
95.0,marketing_tips,2024-12-03
```

#### Результат:

* автоматически скачивается файл `prediction_result.csv`
* добавляется колонка `VIEWS`

```csv
CPM,CHANNEL_NAME,DATE,VIEWS
120.5,crypto_news,2024-12-01,18450
95.0,marketing_tips,2024-12-03,13210
```

---

## Модели и Google Drive

Файлы моделей **не хранятся в GitHub** из-за ограничений по размеру.

При старте приложения автоматически:

* скачивается `catboost_views_model.cbm`
* скачивается `channel_stats.pkl`

Источник — Google Drive (через `gdown`).

---

## Запуск проекта локально

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Открыть в браузере:

```
http://localhost:8000
```

---

## Деплой

Проект разворачивается на **Render** как Web Service:

* стартовая команда: `uvicorn src.api:app --host 0.0.0.0 --port 10000`
* Python environment
* публичный доступ к API

---

## Бизнес-ценность

Решение позволяет:

* прогнозировать охват до запуска рекламы
* сравнивать каналы при одинаковом CPM
* оптимизировать рекламный бюджет
* использовать модель как сервис в медиапланировании

Проект реализован как полноценный ML-сервис, готовый к использованию в production-сценариях.
