# Telegram Ads Reach Forecast

## Описание проекта

**Telegram Ads Reach Forecast** — ML-сервис для прогнозирования охвата (`VIEWS`) рекламных объявлений в Telegram на основе исторических данных.

Сервис предназначен для data-driven медиапланирования и позволяет заранее оценить потенциальный охват рекламы **до запуска кампании**.

Проект реализован как end-to-end ML-решение: от Exploratory Data Analysis (EDA) и обучения модели до production-ready API и batch-прогнозирования через CSV.

---

## Входные параметры прогноза

Во всех сценариях (JSON API, HTML-форма, CSV) используются одинаковые названия полей, совпадающие с train-dataset:

* `CPM` — стоимость за 1000 показов
* `CHANNEL_NAME` — Telegram-канал размещения
* `DATE` — дата размещения объявления (`YYYY-MM-DD`)

Целевая переменная:

* `VIEWS` — прогнозируемый охват

---

## Структура репозитория

```
telegram_ads_forecast/
├── data/
│   └── data_train.csv          # Исторические данные для обучения и EDA
├── notebooks/
│   └── eda.ipynb               # Exploratory Data Analysis
├── models/                     # НЕ хранится в Git (скачивается при старте)
│   ├── catboost_views_model.cbm
│   └── channel_stats.pkl
├── src/
│   ├── api.py                  # FastAPI: API + Web UI
│   ├── features.py             # Feature engineering (train == inference)
│   └── train.py                # Обучение модели
├── templates/
│   ├── index.html              # Главная страница (форма + CSV upload)
│   └── csv_result.html         # Preview результатов CSV
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Данные

Файл `data/data_train.csv` хранится в репозитории и используется для:

* обучения модели
* воспроизводимости экспериментов
* проведения EDA

Основные колонки датасета:

* `CPM`
* `CHANNEL_NAME`
* `DATE`
* `VIEWS`
* `CLICKS`, `ACTIONS` (не используются в модели)

---

## Exploratory Data Analysis (EDA)

EDA выполнен в ноутбуке:

```
notebooks/eda.ipynb
```

### Цели EDA

* анализ распределений `CPM` и `VIEWS`
* исследование зависимости CPM → VIEWS
* анализ различий между каналами
* выявление временных и сезонных паттернов

### Ключевые выводы

* распределение `VIEWS` сильно скошено вправо
* зависимость CPM → VIEWS нелинейная
* каналы существенно различаются по среднему охвату
* присутствуют сезонные эффекты по дням недели и месяцам

Результаты EDA обосновывают использование лог-трансформации целевой переменной и модели CatBoost.

---

## Модель

Используется `CatBoostRegressor`:

* поддержка категориальных признаков (`CHANNEL_NAME`)
* устойчивость к нелинейным зависимостям
* корректная работа с табличными данными

Обучение производится в лог-пространстве:

```
target = log1p(VIEWS)
```

---

## Feature engineering

Feature engineering реализован в `src/features.py` и полностью совпадает между train и inference.

Используемые признаки:

* `CPM`, `cpm_log`, `cpm_sq`
* временные: `dayofweek`, `week`, `month`, `is_weekend`
* статистики канала:

  * `channel_mean_views`
  * `channel_median_views`
  * `channel_history_count`
* `cpm_vs_channel_mean`
* `CHANNEL_NAME` (категориальный)

---

## API и Web-интерфейс

### Главная страница

`GET /`

Предоставляет:

* HTML-форму для одиночного прогноза
* загрузку CSV-файла для batch-прогнозирования

---

### Одиночный прогноз (JSON API)

`POST /predict`

Пример запроса:

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

### Одиночный прогноз (HTML форма)

`POST /predict_form`

Возвращает HTML-страницу с результатом:

```html
<h2>Prediction result</h2>
<p><b>VIEWS:</b> 18450</p>
<a href="/">Back</a>
```

---

### Batch-прогнозирование (CSV)

`POST /predict_csv`

* регистронезависимые заголовки CSV (`CPM`, `CHANNEL_NAME`, `DATE`)
* preview первых 10 строк
* автоматическое добавление колонки `VIEWS`
* возможность скачать итоговый CSV

---

## Улучшение качества без переобучения

* **Линейная калибровка предсказаний** (`CALIBRATION_ALPHA`)
* **Caps по среднему канала** (`MAX_VIEWS_MULTIPLIER`)

Позволяет повысить стабильность прогнозов без retraining.

---

## Запуск проекта локально

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

После запуска приложение доступно по адресу:

```
http://localhost:8000
```

---

## Деплой

Render / Python Web Service:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 10000
```

* Тип сервиса: Python Web Service
* Публичный доступ к API и Web UI

---

## Ограничения и допущения

* не учитывается текст объявления
* не учитываются внешние инфоповоды
* для новых каналов применяется глобальная статистика (`__global__`)
* прогноз отражает ожидаемый охват, без гарантий точности

---

## Команда проекта

**MISIS Future Tech**:

* Бойцова Дарья — Data Scientist, UX/UI Designer, team lead
* Холев Никита - fullstack developer

---

## Бизнес-ценность

Сервис позволяет:

* прогнозировать охват до запуска кампании
* сравнивать каналы при одинаковом CPM
* оптимизировать распределение рекламного бюджета
* использовать решение как готовый API для медиапланирования
