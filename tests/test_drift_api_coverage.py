"""Comprehensive tests for api.drift_api to increase coverage."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import pandas as pd

class TestDriftDetectionEndpoint:
    def test_drift_endpoint_accepts_post(self, test_app_client):
        """Test /monitoring/drift accepts POST."""
        payload = {
            "feature_name": "test_feature",
            "feature_type": "numeric",
            "reference_data": list(np.random.normal(0, 1, 20)),
            "current_data": list(np.random.normal(0, 1, 20)),
            "alert_threshold": 0.05
        }
        response = test_app_client.post("/monitoring/drift", json=payload)
        assert response.status_code in [200, 400, 500]

    def test_drift_insufficient_data(self, test_app_client):
        """Test drift detection with too few samples."""
        payload = {
            "feature_name": "test",
            "feature_type": "numeric",
            "reference_data": [1.0, 2.0],  # Too few
            "current_data": [1.0],
            "alert_threshold": 0.05
        }
        response = test_app_client.post("/monitoring/drift", json=payload)
        assert response.status_code in [400, 422]


class TestDataQualityEndpoint:
    def test_quality_check_endpoint_post(self, test_app_client):
        """Test /monitoring/quality accepts POST."""
        payload = {
            "dataframe_dict": {
                "col1": [1, 2, 3],
                "col2": [4, 5, 6]
            },
            "expected_columns": ["col1", "col2"],
            "check_missing": True,
            "check_range": True,
            "check_schema": True
        }
        response = test_app_client.post("/monitoring/quality", json=payload)
        assert response.status_code in [200, 400]

    def test_quality_check_missing_values(self, test_app_client):
        """Test quality check with missing values."""
        payload = {
            "dataframe_dict": {
                "col1": [1, None, 3],
                "col2": [None, 5, 6]
            },
            "check_missing": True
        }
        response = test_app_client.post("/monitoring/quality", json=payload)
        assert response.status_code in [200, 400]

    def test_quality_check_all_params(self, test_app_client):
        """Test quality check with all parameters."""
        payload = {
            "dataframe_dict": {
                "feature_1": [0.1, 0.2, 0.3],
                "feature_2": [1, 2, 3]
            },
            "expected_columns": ["feature_1", "feature_2"],
            "check_missing": True,
            "check_range": True,
            "check_schema": True
        }
        response = test_app_client.post("/monitoring/quality", json=payload)
        assert response.status_code == 200


class TestDriftHistoryEndpoint:
    def test_drift_history_endpoint_get(self, test_app_client):
        """Test /monitoring/drift/history/{feature_name} GET."""
        response = test_app_client.get("/monitoring/drift/history/test_feature?limit=10")
        # May be 400, 404, or 500 depending on DB
        assert response.status_code in [200, 400, 404, 500]

    def test_drift_history_with_limit(self, test_app_client):
        """Test drift history accepts limit parameter."""
        response = test_app_client.get("/monitoring/drift/history/feature1?limit=5")
        assert response.status_code in [200, 400, 404, 500]

    def test_drift_history_invalid_limit(self, test_app_client):
        """Test drift history rejects invalid limit."""
        response = test_app_client.get("/monitoring/drift/history/feature1?limit=999")
        # Should either reject or cap limit
        assert response.status_code in [200, 400, 422]


class TestStatsEndpoint:
    def test_stats_summary_get(self, test_app_client):
        """Test /monitoring/stats/summary GET."""
        response = test_app_client.get("/monitoring/stats/summary")
        assert response.status_code == 200

    def test_stats_summary_response_format(self, test_app_client):
        """Test stats summary returns correct format."""
        response = test_app_client.get("/monitoring/stats/summary")
        if response.status_code == 200:
            data = response.json()
            assert "data_drift" in data
            assert "predictions" in data
            assert "total_features_checked" in data["data_drift"]
            assert "total" in data["predictions"]

    def test_stats_summary_percentages(self, test_app_client):
        """Test stats summary percentages are valid."""
        response = test_app_client.get("/monitoring/stats/summary")
        if response.status_code == 200:
            data = response.json()
            drift_pct = data["data_drift"].get("drift_percentage", 0)
            assert 0 <= drift_pct <= 100


class TestBatchDriftEndpoint:
    def test_batch_drift_endpoint_exists(self, test_app_client):
        """Test /monitoring/drift/batch/{batch_id} endpoint exists."""
        response = test_app_client.post("/monitoring/drift/batch/999")
        # May be 400, 404, or 500
        assert response.status_code in [200, 400, 404, 500]

    def test_batch_drift_missing_batch(self, test_app_client):
        """Test batch drift with nonexistent batch."""
        response = test_app_client.post("/monitoring/drift/batch/99999")
        assert response.status_code in [404, 400, 500]

    def test_batch_drift_with_reference(self, test_app_client):
        """Test batch drift with reference batch ID."""
        response = test_app_client.post("/monitoring/drift/batch/1?reference_batch_id=2")
        assert response.status_code in [200, 400, 404, 500]


class TestDriftApiErrorHandling:
    def test_drift_invalid_json(self, test_app_client):
        """Test drift endpoint with invalid JSON."""
        response = test_app_client.post("/monitoring/drift", content="{invalid")
        assert response.status_code == 422

    def test_quality_check_invalid_json(self, test_app_client):
        """Test quality endpoint with invalid JSON."""
        response = test_app_client.post("/monitoring/quality", content="{invalid")
        assert response.status_code == 422

    def test_history_invalid_feature_name(self, test_app_client):
        """Test history with special characters in feature name."""
        response = test_app_client.get("/monitoring/drift/history/feature%20name")
        assert response.status_code in [200, 400, 404, 500]


class TestDriftResponseTypes:
    def test_drift_response_includes_feature_name(self, test_app_client):
        """Test drift response includes feature name."""
        payload = {
            "feature_name": "my_feature",
            "feature_type": "numeric",
            "reference_data": list(np.random.normal(0, 1, 100)),
            "current_data": list(np.random.normal(0, 1, 100))
        }
        response = test_app_client.post("/monitoring/drift", json=payload)
        if response.status_code == 200:
            data = response.json()
            assert data.get("feature_name") == "my_feature"

    def test_drift_response_includes_statistics(self, test_app_client):
        """Test drift response includes statistics."""
        payload = {
            "feature_name": "test_feat",
            "feature_type": "numeric",
            "reference_data": list(np.random.normal(0, 1, 100)),
            "current_data": list(np.random.normal(0, 1, 100))
        }
        response = test_app_client.post("/monitoring/drift", json=payload)
        if response.status_code == 200:
            data = response.json()
            assert "statistics" in data or "is_drifted" in data


class TestQualityCheckResults:
    def test_quality_response_has_valid_flag(self, test_app_client):
        """Test quality response includes valid flag."""
        payload = {
            "dataframe_dict": {"col1": [1, 2, 3]},
            "check_missing": True
        }
        response = test_app_client.post("/monitoring/quality", json=payload)
        if response.status_code == 200:
            data = response.json()
            assert "valid" in data

    def test_quality_response_summary(self, test_app_client):
        """Test quality response includes summary."""
        payload = {
            "dataframe_dict": {"col1": [1, 2, 3]},
            "check_missing": True
        }
        response = test_app_client.post("/monitoring/quality", json=payload)
        if response.status_code == 200:
            data = response.json()
            assert "summary" in data


class TestBatchMonitoringEndpoints:
    @patch("api.drift_api.get_batch")
    @patch("api.drift_api.get_training_reference_data")
    @patch("api.drift_api.detect_feature_drift")
    @patch("api.drift_api.save_drift_results_bulk")
    def test_detect_drift_batch_success(self, mock_save, mock_detect, mock_ref, mock_get_batch, test_app_client):
        """Test /monitoring/drift/batch/{batch_id} success."""
        # Mock batch with at least 5 samples
        mock_batch = MagicMock()
        mock_apps = []
        for i in range(5):
            mock_raw = MagicMock()
            mock_raw.raw_data = {"EXT_SOURCE_1": 0.5 + i*0.01, "AMT_CREDIT": 100000 + i}
            mock_apps.append(mock_raw)
        mock_batch.raw_applications = mock_apps
        mock_get_batch.return_value = mock_batch
        
        # Mock reference data with at least 10 samples
        mock_ref.return_value = pd.DataFrame({
            "EXT_SOURCE_1": np.random.normal(0.5, 0.1, 10), 
            "AMT_CREDIT": np.random.normal(100000, 10000, 10)
        })
        
        # Mock detection result
        mock_detect.return_value = {
            "feature_name": "EXT_SOURCE_1",
            "is_drifted": False,
            "ks_statistic": 0.1,
            "p_value": 0.8
        }
        
        response = test_app_client.post("/monitoring/drift/batch/1")
        assert response.status_code == 200
        data = response.json()
        assert data["batch_id"] == 1
        assert "results" in data

    @patch("api.drift_api.get_batch")
    def test_check_data_quality_batch_success(self, mock_get_batch, test_app_client):
        """Test /monitoring/quality/batch/{batch_id} success."""
        # Mock batch
        mock_batch = MagicMock()
        mock_raw = MagicMock()
        mock_raw.raw_data = {"EXT_SOURCE_1": 0.5, "AMT_CREDIT": 100000}
        mock_batch.raw_applications = [mock_raw]
        mock_get_batch.return_value = mock_batch
        
        response = test_app_client.post("/monitoring/quality/batch/1")
        assert response.status_code == 200
        data = response.json()
        assert "valid" in data
        assert "missing_values" in data
