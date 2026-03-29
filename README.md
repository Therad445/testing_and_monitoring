# MLflow + FastAPI service

Сервис на FastAPI, который:

* при старте приложения загружает ML-модель из MLflow;
* имеет хэндлер `POST /predict` — принимает на вход признаки, делает предсказание и возвращает вероятность;
* имеет хэндлер `POST /updateModel`, который принимает `run_id` и подменяет текущую модель;
* имеет хэндлер `GET /metrics` для Prometheus;
* периодически строит отчёты Evidently для контроля drift.

## Переменные окружения

* `MLFLOW_TRACKING_URI` — адрес MLflow Tracking Server, например `http://158.160.2.37:5000/`
* `DEFAULT_RUN_ID` — run_id модели, которая загружается на старте
* `EVIDENTLY_URL` — адрес Evidently
* `EVIDENTLY_PROJECT_ID` — id проекта в Evidently
* `DRIFT_BATCH_SIZE` — размер батча для drift-monitoring
* `DRIFT_INTERVAL_SEC` — период проверки накопленных данных

## Запуск

```bash
export MLFLOW_TRACKING_URI=http://158.160.2.37:5000/
export DEFAULT_RUN_ID=<your_run_id> 71a3cc8476b24c56ae36fffbbc2e23e3
docker compose up --build
```
Например: DEFAULT_RUN_ID=71a3cc8476b24c56ae36fffbbc2e23e3

Сервис будет доступен на:

* `http://158.160.83.144:8890/docs`
* `http://158.160.83.144:8890/metrics`
* `http://158.160.83.144:8890/health`

## Тесты

```bash
pytest -q
```
