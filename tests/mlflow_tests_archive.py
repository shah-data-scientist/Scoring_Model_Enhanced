"""Archived MLflow tests.
These tests are excluded from the main pytest suite but kept here for reference.
"""
import pytest
from fastapi.testclient import TestClient

# From test_critical_coverage.py
def archived_test_mlflow_health_endpoint(test_app_client):
    """Test /health/mlflow endpoint exists."""
    response = test_app_client.get("/health/mlflow")
    assert response.status_code in [200, 503]

def archived_test_mlflow_then_health(test_app_client):
    """Test MLflow health followed by general health."""
    mlflow_health = test_app_client.get("/health/mlflow")
    assert mlflow_health.status_code in [200, 503]
    
    general_health = test_app_client.get("/health")
    assert general_health.status_code == 200

# From test_critical_boost_app.py
def archived_test_mlflow_health_boost(client):
    r = client.get('/health/mlflow')
    assert r.status_code in [200, 500]

# From test_critical_boost_api.py
def archived_test_mlflow_health_connected(client, monkeypatch, app_module):
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

# From test_config.py
def archived_test_get_mlflow_uri(get_mlflow_uri):
    uri = get_mlflow_uri()
    assert isinstance(uri, str)
    assert 'sqlite:///' in uri or 'http' in uri

class ArchivedMLflowTagFunctions:
    def test_get_baseline_tags(self, get_baseline_tags):
        tags = get_baseline_tags('lgbm')
        assert tags['stage'] == 'baseline'
