
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

class TestDriftApiEdgeCases:
    
    @patch("api.drift_api.get_batch")
    def test_detect_drift_batch_no_raw_apps(self, mock_get_batch, test_app_client):
        """Test batch drift detection when batch has no raw applications."""
        mock_batch = MagicMock()
        mock_batch.raw_applications = []
        mock_get_batch.return_value = mock_batch
        
        response = test_app_client.post("/monitoring/drift/batch/1")
        assert response.status_code == 400
        assert "No raw data in batch" in response.json()["detail"]

    @patch("api.drift_api.get_batch")
    def test_detect_drift_batch_no_valid_data(self, mock_get_batch, test_app_client):
        """Test batch drift detection when raw applications have no data."""
        mock_batch = MagicMock()
        mock_raw = MagicMock()
        mock_raw.raw_data = None  # Empty data
        mock_batch.raw_applications = [mock_raw]
        mock_get_batch.return_value = mock_batch
        
        response = test_app_client.post("/monitoring/drift/batch/1")
        assert response.status_code == 400
        assert "No valid data in batch" in response.json()["detail"]

    @patch("api.drift_api.get_batch")
    def test_detect_drift_batch_ref_not_found(self, mock_get_batch, test_app_client):
        """Test batch drift detection when reference batch is not found."""
        # First call returns current batch, second call (for ref) returns None
        mock_current = MagicMock()
        mock_current.raw_applications = [MagicMock(raw_data={"A": 1})]
        
        mock_get_batch.side_effect = [mock_current, None]
        
        response = test_app_client.post("/monitoring/drift/batch/1?reference_batch_id=999")
        assert response.status_code == 400
        assert "Reference batch not found" in response.json()["detail"]

    @patch("api.drift_api.get_batch")
    @patch("api.drift_api.get_training_reference_data")
    def test_detect_drift_batch_no_numeric_cols(self, mock_ref, mock_get_batch, test_app_client):
        """Test batch drift detection with no numeric columns."""
        # Current batch has only strings
        mock_batch = MagicMock()
        mock_batch.raw_applications = [MagicMock(raw_data={"Name": "A", "City": "B"})]
        mock_get_batch.return_value = mock_batch
        
        # Reference data
        mock_ref.return_value = pd.DataFrame({"Name": ["A"], "City": ["B"]})
        
        response = test_app_client.post("/monitoring/drift/batch/1")
        assert response.status_code == 400
        assert "No numeric features found" in response.json()["detail"]

    @patch("api.drift_api.get_batch")
    def test_check_quality_batch_not_found(self, mock_get_batch, test_app_client):
        """Test quality check when batch not found."""
        mock_get_batch.return_value = None
        
        response = test_app_client.post("/monitoring/quality/batch/999")
        assert response.status_code == 400
        assert "not found" in response.json()["detail"]

    @patch("api.drift_api.get_batch")
    def test_check_quality_batch_no_raw_apps(self, mock_get_batch, test_app_client):
        """Test quality check when batch has no raw apps."""
        mock_batch = MagicMock()
        mock_batch.raw_applications = []
        mock_get_batch.return_value = mock_batch
        
        response = test_app_client.post("/monitoring/quality/batch/1")
        assert response.status_code == 400
        assert "No raw data" in response.json()["detail"]

    @patch("api.drift_api.get_batch")
    def test_check_quality_batch_no_valid_data(self, mock_get_batch, test_app_client):
        """Test quality check when apps have no data."""
        mock_batch = MagicMock()
        mock_batch.raw_applications = [MagicMock(raw_data=None)]
        mock_get_batch.return_value = mock_batch
        
        response = test_app_client.post("/monitoring/quality/batch/1")
        assert response.status_code == 400
        assert "No valid data" in response.json()["detail"]
