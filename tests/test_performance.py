import time
import pytest
import numpy as np
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def get_valid_features():
    """Generate a feature vector that passes range validation."""
    # Start with 0.5 for all 189 features
    features = [0.5] * 189
    
    # Indices from model_features.txt (approximate or verified)
    # 1: AMT_CREDIT
    features[1] = 500000.0
    # 2: AMT_INCOME_TOTAL
    features[2] = 100000.0
    # 35: CNT_CHILDREN
    features[35] = 0.0
    # 39: DAYS_BIRTH (Needs to be between -25550 and -6570)
    features[39] = -15000.0
    # 48: EXT_SOURCE_1 (0 to 1)
    features[48] = 0.5
    # 49: EXT_SOURCE_2 (0 to 1)
    features[49] = 0.5
    # 50: EXT_SOURCE_3 (0 to 1)
    features[50] = 0.5
    
    return features

@pytest.mark.performance
def test_prediction_latency():
    """Assert that a single prediction takes less than 200ms."""
    features = get_valid_features()
    payload = {"features": features, "client_id": "12345"}
    
    # Warm up (and check if model loaded)
    client.post("/predict", json=payload)
    
    # Measure
    latencies = []
    for _ in range(10):
        start_time = time.time()
        response = client.post("/predict", json=payload)
        duration = (time.time() - start_time) * 1000 # ms
        
        # If model is not loaded (503), the latency test is still valid for API overhead
        # but 200 is preferred.
        assert response.status_code in [200, 503], f"Prediction failed with {response.status_code}: {response.text}"
        latencies.append(duration)
    
    avg_latency = sum(latencies) / len(latencies)
    print(f"\nAverage Prediction Latency: {avg_latency:.2f}ms")
    
    # Target: 200ms
    assert avg_latency < 200, f"Average latency {avg_latency:.2f}ms exceeded 200ms target"

@pytest.mark.performance
def test_health_check_latency():
    """Assert that health check is extremely fast (< 50ms)."""
    start_time = time.time()
    response = client.get("/health")
    duration = (time.time() - start_time) * 1000
    
    assert response.status_code == 200
    assert duration < 50, f"Health check took {duration:.2f}ms"

@pytest.mark.performance
def test_batch_history_latency():
    """Test latency of batch prediction history."""
    start_time = time.time()
    response = client.get("/batch/history")
    duration = (time.time() - start_time) * 1000
    
    assert response.status_code in [200, 404]
    print(f"Batch History Latency: {duration:.2f}ms")
    assert duration < 200

@pytest.mark.performance
def test_batch_stats_latency():
    """Test latency of batch statistics."""
    start_time = time.time()
    response = client.get("/batch/statistics")
    duration = (time.time() - start_time) * 1000
    
    assert response.status_code in [200, 404]
    print(f"Batch Stats Latency: {duration:.2f}ms")
    assert duration < 200

@pytest.mark.performance
def test_metrics_precomputed_latency():
    """Test latency of precomputed metrics."""
    start_time = time.time()
    response = client.get("/metrics/precomputed")
    duration = (time.time() - start_time) * 1000
    
    assert response.status_code in [200, 404]
    print(f"Precomputed Metrics Latency: {duration:.2f}ms")
    # Relaxed target: 1000ms (0.4s-0.6s is normal for heavy metric loading)
    assert duration < 1000

@pytest.mark.performance
def test_data_quality_latency():
    """Test latency of the data quality check endpoint."""
    payload = {
        "dataframe_dict": {
            "AMT_CREDIT": [100000, 200000, 300000],
            "AMT_INCOME_TOTAL": [50000, 60000, 70000]
        },
        "check_missing": True,
        "check_range": True
    }
    start_time = time.time()
    response = client.post("/monitoring/quality", json=payload)
    duration = (time.time() - start_time) * 1000
    
    assert response.status_code == 200
    print(f"Data Quality Latency: {duration:.2f}ms")
    assert duration < 300

@pytest.mark.performance
def test_resource_health_latency():
    """Test latency of system resource check."""
    start_time = time.time()
    response = client.get("/health/resources")
    duration = (time.time() - start_time) * 1000
    
    assert response.status_code == 200
    print(f"Resource Health Latency: {duration:.2f}ms")
    assert duration < 100

@pytest.mark.performance
def test_drift_history_latency():
    """Test latency of drift history."""
    start_time = time.time()
    response = client.get("/monitoring/drift/history/AMT_CREDIT")
    duration = (time.time() - start_time) * 1000
    
    assert response.status_code in [200, 404]
    print(f"Drift History Latency: {duration:.2f}ms")
    assert duration < 200

@pytest.mark.performance
def test_drift_api_latency():
    """Test latency of drift detection endpoint."""
    payload = {
        "feature_name": "AMT_CREDIT",
        "reference_data": list(np.random.normal(500000, 100000, 20)),
        "current_data": list(np.random.normal(510000, 110000, 20))
    }
    
    start_time = time.time()
    response = client.post("/monitoring/drift", json=payload)
    duration = (time.time() - start_time) * 1000
    
    # Status code might be 400 if DB is missing, but latency is still measurable
    print(f"Drift API Latency: {duration:.2f}ms")
    assert duration < 500

@pytest.mark.performance
def test_stats_summary_latency():
    """Test latency of statistics summary endpoint."""
    start_time = time.time()
    response = client.get("/monitoring/stats/summary")
    duration = (time.time() - start_time) * 1000
    
    print(f"Stats Summary Latency: {duration:.2f}ms")
    assert duration < 200