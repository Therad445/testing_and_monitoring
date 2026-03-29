import numpy as np
from fastapi.testclient import TestClient

from ml_service.app import app, MODEL
from ml_service.model import ModelData


class DummyModel:
    feature_names_in_ = np.array(['age', 'hours.per.week'])

    def predict_proba(self, df):
        return np.array([[0.2, 0.8]])


def test_predict_success():
    MODEL.data = ModelData(
        model=DummyModel(),
        run_id='run-1',
        features=['age', 'hours.per.week'],
        model_type='LogisticRegression',
    )

    client = TestClient(app)
    response = client.post('/predict', json={'age': 30, 'hours.per.week': 40})

    assert response.status_code == 200
    body = response.json()
    assert body['prediction'] == 1
    assert abs(body['probability'] - 0.8) < 1e-9
    assert body['run_id'] == 'run-1'
    assert body['model_type'] == 'LogisticRegression'


def test_predict_missing_feature():
    MODEL.data = ModelData(
        model=DummyModel(),
        run_id='run-1',
        features=['age', 'hours.per.week'],
        model_type='LogisticRegression',
    )

    client = TestClient(app)
    response = client.post('/predict', json={'age': 30})

    assert response.status_code == 422
    assert 'Missing required feature' in response.text


def test_predict_returns_503_if_model_not_loaded():
    MODEL.data = ModelData()

    client = TestClient(app)
    response = client.post('/predict', json={'age': 30, 'hours.per.week': 40})

    assert response.status_code == 503


def test_metrics_endpoint_available():
    MODEL.data = ModelData()

    client = TestClient(app)
    response = client.get('/metrics')

    assert response.status_code == 200
    assert 'http_requests_total' in response.text