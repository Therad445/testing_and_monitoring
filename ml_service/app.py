import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ml_service import config
from ml_service.drift import drift_monitor
from ml_service.features import NUMERIC_COLUMNS, to_dataframe
from ml_service.metrics import (
    FEATURE_CATEGORICAL_VALUE_TOTAL,
    FEATURE_NUMERIC_VALUE,
    HTTP_5XX_TOTAL,
    HTTP_REQUEST_DURATION_SECONDS,
    HTTP_REQUESTS_TOTAL,
    MODEL_INFERENCE_DURATION_SECONDS,
    MODEL_LOADED,
    MODEL_PREDICTION_PROBABILITY,
    MODEL_PREDICTION_TOTAL,
    MODEL_UPDATES_TOTAL,
    PREPROCESS_DURATION_SECONDS,
    set_current_model_metrics,
)
from ml_service.mlflow_utils import configure_mlflow
from ml_service.model import Model
from ml_service.schemas import (
    PredictRequest,
    PredictResponse,
    UpdateModelRequest,
    UpdateModelResponse,
)

logger = logging.getLogger(__name__)
MODEL = Model()


def observe_feature_metrics(row: dict[str, Any]) -> None:
    for name, value in row.items():
        if value is None:
            continue

        if name in NUMERIC_COLUMNS:
            FEATURE_NUMERIC_VALUE.labels(name).observe(float(value))
        else:
            FEATURE_CATEGORICAL_VALUE_TOTAL.labels(name, str(value)).inc()


@asynccontextmanager
async def lifespan(app: FastAPI):
    drift_task = None

    try:
        configure_mlflow()
        state = MODEL.set(run_id=config.default_run_id())
        set_current_model_metrics(state.run_id, state.model_type, state.features)
    except Exception:
        MODEL_LOADED.set(0)
        logger.exception('Initial model load failed')

    drift_task = asyncio.create_task(drift_monitor.run_forever())

    try:
        yield
    finally:
        if drift_task is not None:
            drift_task.cancel()


def create_app() -> FastAPI:
    app = FastAPI(title='MLflow FastAPI service', version='1.0.0', lifespan=lifespan)

    @app.middleware('http')
    async def metrics_middleware(request: Request, call_next):
        method = request.method
        endpoint = request.url.path
        started = time.perf_counter()
        status = 500

        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            duration = time.perf_counter() - started
            HTTP_REQUESTS_TOTAL.labels(method, endpoint, str(status)).inc()
            HTTP_REQUEST_DURATION_SECONDS.labels(method, endpoint).observe(duration)
            if status >= 500:
                HTTP_5XX_TOTAL.labels(method, endpoint, str(status)).inc()

    @app.get('/health')
    def health() -> dict[str, Any]:
        state = MODEL.get()
        if state.model is None:
            raise HTTPException(status_code=503, detail='Model is not loaded')

        return {
            'status': 'ok',
            'run_id': state.run_id,
            'model_type': state.model_type,
            'features': state.features,
        }

    @app.get('/metrics')
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post('/predict', response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        state = MODEL.get()
        model = state.model

        if model is None:
            raise HTTPException(status_code=503, detail='Model is not loaded yet')

        try:
            preprocess_started = time.perf_counter()
            df = to_dataframe(request, needed_columns=state.features)
            PREPROCESS_DURATION_SECONDS.observe(time.perf_counter() - preprocess_started)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        try:
            observe_feature_metrics(df.iloc[0].to_dict())

            inference_started = time.perf_counter()
            probability = float(model.predict_proba(df)[0][1])
            MODEL_INFERENCE_DURATION_SECONDS.observe(time.perf_counter() - inference_started)
        except Exception as exc:
            logger.exception('Inference failed')
            raise HTTPException(status_code=500, detail='Inference failed') from exc

        prediction = int(probability >= 0.5)
        MODEL_PREDICTION_TOTAL.labels(str(prediction)).inc()
        MODEL_PREDICTION_PROBABILITY.observe(probability)

        drift_monitor.add_event(
            {
                **df.iloc[0].to_dict(),
                'prediction': prediction,
                'probability': probability,
            }
        )

        return PredictResponse(
            prediction=prediction,
            probability=probability,
            run_id=state.run_id or '',
            model_type=state.model_type or '',
        )

    @app.post('/updateModel', response_model=UpdateModelResponse)
    def update_model(req: UpdateModelRequest) -> UpdateModelResponse:
        try:
            state = MODEL.set(run_id=req.run_id)
            set_current_model_metrics(state.run_id, state.model_type, state.features)
            MODEL_UPDATES_TOTAL.labels('success').inc()

            return UpdateModelResponse(
                run_id=state.run_id or '',
                model_type=state.model_type or '',
                features=state.features,
            )
        except Exception as exc:
            MODEL_UPDATES_TOTAL.labels('failed').inc()
            logger.exception('Model update failed')
            raise HTTPException(
                status_code=400,
                detail=f'Failed to load model for run_id={req.run_id}',
            ) from exc

    return app


app = create_app()