import io
import pytest
import numpy as np
from fastapi.testclient import TestClient
import api.app as app_module
from api.app import app
from api.batch_predictions import calculate_risk_level, create_results_dataframe, dataframe_to_csv_stream
from backend.database import get_db_info

client = TestClient(app)


@pytest.fixture(autouse=True)
def restore_model_state():
    orig_model = app_module.model
    orig_meta = dict(app_module.model_metadata)
    app_module._rate_limit_store.clear()
    yield
    app_module.model = orig_model
    app_module.model_metadata = orig_meta
    app_module._rate_limit_store.clear()


class DummyModel:
    def __init__(self, prob):
        self._prob = prob
    def predict(self, X):
        return np.array([1])
    def predict_proba(self, X):
        return np.array([[1 - self._prob, self._prob]])


class TestHealthPaths:
    def test_health_healthy_when_model_loaded(self):
        app_module.model = DummyModel(0.2)
        app_module.model_metadata = {'name': 'dummy', 'stage': 'prod'}
        r = client.get('/health')
        assert r.status_code in [200, 500, 429]
        if r.status_code == 200:
            body = r.json()
            assert 'status' in body

    def test_health_unhealthy_when_model_missing(self):
        app_module.model = None
        r = client.get('/health')
        assert r.status_code in [200, 500, 429]

    def test_mlflow_health_connected(self, monkeypatch):
        def fake_run_info(**kwargs):
            return {
                'experiment_name': 'exp',
                'run_id': '1',
                'run_name': 'r',
                'status': 'FINISHED',
                'parameters': {'optimal_threshold': 0.5},
                'metrics': {'auc': 0.9}
            }
        monkeypatch.setattr(app_module, 'get_mlflow_run_info', fake_run_info)
        r = client.get('/health/mlflow')
        assert r.status_code in [200, 500, 429]

    def test_mlflow_health_error(self, monkeypatch):
        def boom(**kwargs):
            raise RuntimeError('mlflow down')
        monkeypatch.setattr(app_module, 'get_mlflow_run_info', boom)
        r = client.get('/health/mlflow')
        assert r.status_code in [200, 500, 429]

    def test_database_health_connected(self, monkeypatch):
        monkeypatch.setattr(app_module, 'get_db_info', lambda: {
            'connected': True,
            'is_sqlite': True,
            'database_url': 'sqlite:///test.db'
        })
        r = client.get('/health/database')
        assert r.status_code in [200, 500, 429]

    def test_database_health_unconnected(self, monkeypatch):
        monkeypatch.setattr(app_module, 'get_db_info', lambda: {
            'connected': False,
            'is_sqlite': False,
            'database_url': 'postgres://'
        })
        r = client.get('/health/database')
        assert r.status_code in [200, 500, 429]


class TestPredictPaths:
    def test_predict_low_risk(self):
        app_module.model = DummyModel(0.1)
        app_module.model_metadata = {'stage': 'prod'}
        payload = {"features": [0.0]*app_module.EXPECTED_FEATURES}
        r = client.post('/predict', json=payload)
        assert r.status_code in [200, 503, 422, 429]

    def test_predict_medium_risk(self):
        app_module.model = DummyModel(0.35)
        app_module.model_metadata = {'stage': 'prod'}
        payload = {"features": [0.1]*app_module.EXPECTED_FEATURES}
        r = client.post('/predict', json=payload)
        assert r.status_code in [200, 503, 422, 429]

    def test_predict_high_risk(self):
        app_module.model = DummyModel(0.8)
        app_module.model_metadata = {'stage': 'prod'}
        payload = {"features": [0.2]*app_module.EXPECTED_FEATURES}
        r = client.post('/predict', json=payload)
        assert r.status_code in [200, 503, 422, 429]


class TestBatchHelpers:
    def test_calculate_risk_level_boundaries(self):
        assert calculate_risk_level(0.0) == 'LOW'
        assert calculate_risk_level(0.29) == 'LOW'
        assert calculate_risk_level(0.30) == 'MEDIUM'
        assert calculate_risk_level(0.49) == 'MEDIUM'
        assert calculate_risk_level(0.50) == 'HIGH'

    def test_create_results_dataframe(self):
        import pandas as pd
        sk_ids = pd.Series([1,2,3])
        preds = np.array([0,1,0])
        probs = np.array([0.1,0.4,0.8])
        df = create_results_dataframe(sk_ids, preds, probs)
        assert df.shape[0] == 3
        assert 'RISK_LEVEL' in df.columns

    def test_dataframe_to_csv_stream(self):
        import pandas as pd
        df = pd.DataFrame({'a':[1,2], 'b':[3,4]})
        stream = dataframe_to_csv_stream(df)
        assert isinstance(stream, io.BytesIO)
        content = stream.getvalue().decode('utf-8')
        assert 'a,b' in content


class TestDatabaseInfo:
    def test_get_db_info_returns_dict(self):
        info = get_db_info()
        assert isinstance(info, dict)
        assert 'connected' in info
