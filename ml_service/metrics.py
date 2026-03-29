from prometheus_client import Counter, Gauge, Histogram

HTTP_REQUESTS_TOTAL = Counter(
    'http_requests_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status'],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint'],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)

HTTP_5XX_TOTAL = Counter(
    'http_5xx_total',
    'Total number of 5xx responses',
    ['method', 'endpoint', 'status'],
)

PREPROCESS_DURATION_SECONDS = Histogram(
    'preprocess_duration_seconds',
    'Preprocessing duration',
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
)

MODEL_INFERENCE_DURATION_SECONDS = Histogram(
    'model_inference_duration_seconds',
    'Model inference duration',
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1),
)

MODEL_PREDICTION_TOTAL = Counter(
    'model_prediction_total',
    'Predicted labels count',
    ['prediction'],
)

MODEL_PREDICTION_PROBABILITY = Histogram(
    'model_prediction_probability',
    'Distribution of output probabilities',
    buckets=(0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)

MODEL_UPDATES_TOTAL = Counter(
    'model_updates_total',
    'Number of model updates',
    ['status'],
)

MODEL_LOADED = Gauge(
    'model_loaded',
    'Whether model is loaded',
)

CURRENT_MODEL_INFO = Gauge(
    'current_model_info',
    'Current model metadata',
    ['run_id', 'model_type'],
)

CURRENT_MODEL_FEATURE = Gauge(
    'current_model_feature',
    'Features required by current model',
    ['run_id', 'feature'],
)

FEATURE_NUMERIC_VALUE = Histogram(
    'feature_numeric_value',
    'Numeric feature values',
    ['feature'],
    buckets=(0, 1, 5, 10, 20, 30, 40, 50, 70, 100, 1000, 10000, 100000, 1000000),
)

FEATURE_CATEGORICAL_VALUE_TOTAL = Counter(
    'feature_categorical_value_total',
    'Categorical feature values',
    ['feature', 'value'],
)

_CURRENT_MODEL_LABELS: tuple[str, str] | None = None
_CURRENT_FEATURE_LABELS: list[tuple[str, str]] = []


def set_current_model_metrics(run_id: str, model_type: str, features: list[str]) -> None:
    global _CURRENT_MODEL_LABELS, _CURRENT_FEATURE_LABELS

    if _CURRENT_MODEL_LABELS is not None:
        CURRENT_MODEL_INFO.remove(*_CURRENT_MODEL_LABELS)

    for labels in _CURRENT_FEATURE_LABELS:
        CURRENT_MODEL_FEATURE.remove(*labels)

    CURRENT_MODEL_INFO.labels(run_id, model_type).set(1)
    _CURRENT_MODEL_LABELS = (run_id, model_type)

    _CURRENT_FEATURE_LABELS = []
    for feature in features:
        CURRENT_MODEL_FEATURE.labels(run_id, feature).set(1)
        _CURRENT_FEATURE_LABELS.append((run_id, feature))

    MODEL_LOADED.set(1)