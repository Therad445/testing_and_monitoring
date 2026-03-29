import threading
from dataclasses import dataclass, field
from typing import Any

from ml_service.mlflow_utils import load_model


@dataclass(frozen=True)
class ModelData:
    model: Any | None = None
    run_id: str | None = None
    features: list[str] = field(default_factory=list)
    model_type: str | None = None


class Model:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.data = ModelData()

    def get(self) -> ModelData:
        with self.lock:
            return self.data

    def set(self, run_id: str) -> ModelData:
        model = load_model(run_id=run_id)

        features = list(getattr(model, 'feature_names_in_', []))
        if not features:
            raise ValueError('Loaded model does not expose feature_names_in_')

        if hasattr(model, 'steps') and model.steps:
            model_type = type(model.steps[-1][1]).__name__
        else:
            model_type = type(model).__name__

        new_data = ModelData(
            model=model,
            run_id=run_id,
            features=features,
            model_type=model_type,
        )

        with self.lock:
            self.data = new_data

        return new_data

    @property
    def features(self) -> list[str]:
        return self.get().features