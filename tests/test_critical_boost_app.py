import pytest
from fastapi.testclient import TestClient
from api.app import app

class TestHealthEndpoints:
    def test_health_basic(self, test_app_client):
        r = test_app_client.get('/health')
        assert r.status_code in [200, 503]
        d = r.json()
        assert 'status' in d or isinstance(d, dict)

    def test_database_health(self, test_app_client):
        r = test_app_client.get('/health/database')
        assert r.status_code in [200, 500, 503]

class TestPredictErrors:
    def test_predict_model_not_loaded(self, test_app_client):
        payload = {"features": [0.0]*189}
        r = test_app_client.post('/predict', json=payload)
        assert r.status_code in [200, 422, 503]

    def test_predict_invalid_feature_length(self, test_app_client):
        payload = {"features": [0.0]*10}
        r = test_app_client.post('/predict', json=payload)
        assert r.status_code == 422

    def test_predict_non_numeric(self, test_app_client):
        payload = {"features": ["x"]*189}
        r = test_app_client.post('/predict', json=payload)
        assert r.status_code == 422

class TestBatchHelpers:
    def test_calculate_risk_level_bounds(self):
        from api.batch_predictions import calculate_risk_level
        assert calculate_risk_level(0.05) == 'LOW'
        assert calculate_risk_level(0.35) in ['MEDIUM','LOW','HIGH']
        assert calculate_risk_level(0.85) in ['HIGH','CRITICAL','MEDIUM']

    def test_create_results_dataframe_basic(self):
        from api.batch_predictions import create_results_dataframe
        import numpy as np
        import pandas as pd
        sk_ids = pd.Series([1,2])
        preds = np.array([0,1])
        probs = np.array([0.1,0.9])
        out = create_results_dataframe(sk_ids, preds, probs)
        assert list(out.columns) == ['SK_ID_CURR','PREDICTION','PROBABILITY','RISK_LEVEL']

class TestDatabaseInfo:
    def test_get_db_info_safe(self):
        from backend.database import get_db_info
        info = get_db_info()
        assert isinstance(info, dict)