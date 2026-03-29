# README.md

# MLflow + FastAPI service

Небольшой сервис на FastAPI для инференса модели из MLflow.

Что делает сервис:

* при старте загружает модель из MLflow;
* принимает входные признаки в `POST /predict` и возвращает класс и вероятность;
* позволяет переключить модель через `POST /updateModel`;
* отдаёт метрики в `GET /metrics` для Prometheus;
* периодически формирует отчёты Evidently для контроля drift.

## Переменные окружения

* `MLFLOW_TRACKING_URI` — адрес MLflow Tracking Server, например `http://158.160.2.37:5000/`
* `DEFAULT_RUN_ID` — `run_id` модели, которая загружается при старте
* `EVIDENTLY_URL` — адрес сервиса Evidently
* `EVIDENTLY_PROJECT_ID` — идентификатор проекта в Evidently
* `DRIFT_BATCH_SIZE` — размер батча для drift-monitoring
* `DRIFT_INTERVAL_SEC` — интервал между попытками отправки отчёта

## Запуск

```bash
export MLFLOW_TRACKING_URI=http://158.160.2.37:5000/
export DEFAULT_RUN_ID=71a3cc8476b24c56ae36fffbbc2e23e3
docker compose up --build
```

После запуска сервис доступен по адресу:

* `http://158.160.83.144:8890/docs`
* `http://158.160.83.144:8890/metrics`
* `http://158.160.83.144:8890/health`

## Тесты

```bash
pytest -q
```

Если тесты запускаются внутри контейнера:

```bash
docker compose exec mlflow_example pytest -q
```