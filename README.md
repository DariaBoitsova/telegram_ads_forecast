```md
# Telegram Ads Reach Forecast

## Описание проекта

Telegram Ads Reach Forecast — это ML-сервис для прогнозирования охвата (`VIEWS`) рекламных объявлений в Telegram на основе исторических данных.

Сервис предназначен для data-driven медиапланирования и позволяет заранее оценить потенциальный охват рекламы **до запуска кампании**.

Проект реализован как end-to-end ML-решение: от EDA и обучения модели до production-ready API и batch-прогнозирования через CSV.

---

## Входные параметры прогноза

Во всех сценариях (API, HTML-форма, CSV) используются одинаковые названия полей:

- `CPM` — стоимость за 1000 показов  
- `CHANNEL_NAME` — Telegram-канал размещения  
- `DATE` — дата размещения (YYYY-MM-DD)

Целевая переменная:

- `VIEWS` — прогнозируемый охват

---

## Структура репозитория

```

telegram_ads_forecast/
├── data/
│   └── data_train.csv
├── notebooks/
│   └── eda.ipynb
├── models/                   # скачиваются при старте
│   ├── catboost_views_model.cbm
│   └── channel_stats.pkl
├── src/
│   ├── api.py
│   ├── features.py
│   └── train.py
├── templates/
│   ├── index.html
│   └── csv_result.html
├── requirements.txt
├── README.md
└── .gitignore

```

---

## Данные

Файл `data/data_train.csv` хранится в репозитории и используется для:

- обучения модели
- EDA
- проверки feature engineering

Основные колонки:

- `CPM`
- `CHANNEL_NAME`
- `DATE`
- `VIEWS`
- `CLICKS`, `ACTIONS` (не используются в модели)

---

## Exploratory Data Analysis (EDA)

EDA выполнен в ноутбуке `notebooks/eda.ipynb`.

### Цели EDA

- изучить распределения `CPM` и `VIEWS`
- выявить зависимость CPM → VIEWS
- проанализировать различия между каналами
- исследовать временные паттерны

### Основные выводы

- распределение `VIEWS` сильно скошено вправо
- зависимость CPM → VIEWS нелинейная
- каналы существенно различаются по среднему охвату
- присутствуют сезонные эффекты

EDA обосновывает использование лог-трансформации и CatBoost.

---

## Модель

Используется `CatBoostRegressor`:

- поддержка категориальных признаков
- устойчивость к нелинейностям
- обучение в лог-пространстве:

```

target = log1p(VIEWS)

````

---

## Feature engineering

Feature engineering реализован в `src/features.py` и совпадает между train и inference.

Используемые признаки:

- `CPM`, `cpm_log`, `cpm_sq`
- `dayofweek`, `week`, `month`, `is_weekend`
- `channel_mean_views`
- `channel_median_views`
- `channel_history_count`
- `cpm_vs_channel_mean`
- `CHANNEL_NAME`

---

## API и Web-интерфейс

### Одиночный прогноз (JSON)

`POST /predict`

```json
{
  "CPM": 120.5,
  "CHANNEL_NAME": "crypto_news",
  "DATE": "2024-12-01"
}
````

Ответ:

```json
{
  "VIEWS": 18450
}
```

---

### Batch-прогнозирование (CSV)

`POST /predict_csv`

* регистронезависимые заголовки
* preview результата
* скачивание CSV с колонкой `VIEWS`

---

## Улучшение качества без переобучения

Применены inference-time методы:

* линейная калибровка предсказаний
* caps на основе статистики канала

Позволяет повысить стабильность прогнозов без retraining.

---

## Запуск локально

```bash
pip install -r requirements.txt
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

---

## Деплой

Render / Python Web Service:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 10000
```

---

## Ограничения и допущения

* не используется текст объявления
* не учитываются внешние инфоповоды
* для новых каналов применяется глобальная статистика
* модель оценивает ожидаемый масштаб охвата

---

## Команда проекта

**MISIS Future Tech**

* Бойцова Дарья — Data Scientist, UX/UI Designer

---

## Бизнес-ценность

Сервис позволяет:

* прогнозировать охват до запуска рекламы
* сравнивать каналы при одинаковом CPM
* оптимизировать рекламный бюджет
* использовать решение как готовый API

```
