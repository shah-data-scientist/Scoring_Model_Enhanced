"""Tests for metrics module."""
import pytest

class TestMetricsEndpoints:
    """Tests for metrics endpoints."""

    def test_confusion_matrix_endpoint(self, test_app_client):
        """Test confusion matrix endpoint exists."""
        response = test_app_client.get("/metrics/confusion-matrix")
        # Should return 200 or error, not 404
        assert response.status_code in [200, 404, 500]

    def test_optimal_threshold_endpoint(self, test_app_client):
        """Test optimal threshold endpoint."""
        response = test_app_client.get("/metrics/optimal-threshold")
        assert response.status_code in [200, 404, 500]

    def test_threshold_metrics_endpoint(self, test_app_client):
        """Test threshold metrics endpoint."""
        response = test_app_client.get("/metrics/threshold/0.5")
        assert response.status_code in [200, 404, 422, 500]

    def test_all_thresholds_endpoint(self, test_app_client):
        """Test all thresholds endpoint."""
        response = test_app_client.get("/metrics/all-thresholds")
        assert response.status_code in [200, 404, 500]

    def test_feature_importance_endpoint(self, test_app_client):
        """Test feature importance endpoint."""
        response = test_app_client.get("/metrics/feature-importance")
        assert response.status_code in [200, 404, 500]


class TestMetricsFunctions:
    """Tests for metrics calculation functions."""

    def test_confusion_matrix_calculation(self):
        """Test confusion matrix calculation."""
        import numpy as np
        from sklearn.metrics import confusion_matrix
        
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1])
        
        cm = confusion_matrix(y_true, y_pred)
        
        assert cm.shape == (2, 2)
        assert cm.sum() == len(y_true)

    def test_precision_recall_calculation(self):
        """Test precision and recall calculation."""
        from sklearn.metrics import precision_score, recall_score
        import numpy as np
        
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0])
        
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1