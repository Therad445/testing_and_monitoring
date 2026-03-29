from fastapi.testclient import TestClient

from ml_service.app import app, MODEL


def test_update_model_invalid_run_id(monkeypatch):
    def broken_set(run_id: str):
        raise RuntimeError('run not found')

    monkeypatch.setattr(MODEL, 'set', broken_set)

    client = TestClient(app)
    response = client.post('/updateModel', json={'run_id': 'bad-run-id'})

    assert response.status_code == 400
    assert 'Failed to load model' in response.text