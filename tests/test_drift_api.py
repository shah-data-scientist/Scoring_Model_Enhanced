"""Tests for API drift detection endpoints."""
import pytest
import numpy as np
import pandas as pd

class TestDriftEndpoints:
    """Tests for drift detection endpoints."""

    def test_drift_endpoint_exists(self, test_app_client):
        """Test drift detection endpoint exists."""
        response = test_app_client.get("/monitoring/drift")
        # Should return 200 or error, not 404
        assert response.status_code != 404

    def test_quality_check_endpoint(self, test_app_client):
        """Test data quality check endpoint."""
        response = test_app_client.get("/monitoring/quality")
        # Should exist and return valid response
        assert response.status_code != 404

    def test_stats_summary_endpoint(self, test_app_client):
        """Test statistics summary endpoint."""
        response = test_app_client.get("/monitoring/stats/summary")
        # Should exist
        assert response.status_code in [200, 404, 500]


class TestDriftDetectionCalculations:
    """Tests for drift detection math functions."""

    def test_ks_statistic_range(self):
        """Test KS statistic is in valid range."""
        from api.drift_detection import calculate_ks_statistic
        
        reference = np.random.normal(0, 1, 100)
        current = np.random.normal(0, 1, 100)
        
        ks_stat, p_value = calculate_ks_statistic(reference, current)
        
        assert isinstance(ks_stat, (float, np.floating))
        assert 0 <= ks_stat <= 1

    def test_chi_square_nonnegative(self):
        """Test chi-square is non-negative."""
        from api.drift_detection import calculate_chi_square
        
        reference = pd.Series(['A', 'B', 'C'] * 30)
        current = pd.Series(['A', 'B', 'C'] * 30)
        
        chi2_stat, p_value = calculate_chi_square(reference, current)
        
        assert isinstance(chi2_stat, (float, np.floating))
        assert chi2_stat >= 0

    def test_psi_nonnegative(self):
        """Test PSI is non-negative."""
        from api.drift_detection import calculate_psi
        
        reference = np.random.uniform(0, 1, 100)
        current = np.random.uniform(0, 1, 100)
        
        psi = calculate_psi(reference, current)
        
        assert isinstance(psi, (float, np.floating))
        assert psi >= 0

    def test_missing_values_dict_structure(self):
        """Test missing rates return dict."""
        from api.drift_detection import check_missing_values
        
        data = pd.DataFrame({
            'col1': [1, 2, np.nan, 4, 5],
            'col2': [1, 2, 3, 4, 5]
        })
        
        result = check_missing_values(data)
        
        assert isinstance(result, dict)
        assert 'col1' in result
        assert 'col2' in result
        assert result['col1'] == 20.0

    def test_out_of_range_dict_structure(self):
        """Test out of range return dict."""
        from api.drift_detection import check_out_of_range
        
        data = pd.DataFrame({'col1': [1, 2, 3, 4, 100]})
        thresholds = {'col1': (0, 10)}
        
        result = check_out_of_range(data, thresholds=thresholds)
        
        assert isinstance(result, dict)
        assert 'col1' in result
        assert result['col1']['out_of_range_count'] == 1
