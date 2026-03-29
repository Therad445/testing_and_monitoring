import os

MODEL_ARTIFACT_PATH = 'model'


def getenv_default(name: str, default: str) -> str:
    return os.getenv(name, default)


def tracking_uri() -> str:
    value = os.getenv('MLFLOW_TRACKING_URI')
    if not value:
        raise RuntimeError('Please set MLFLOW_TRACKING_URI')
    return value


def default_run_id() -> str:
    value = os.getenv('DEFAULT_RUN_ID')
    if not value:
        raise RuntimeError('Set DEFAULT_RUN_ID to load model on startup')
    return value