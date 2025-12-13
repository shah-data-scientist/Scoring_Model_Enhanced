"""
Tests for risk calculation and batch processing.
"""
import pytest
from api.batch_predictions import calculate_risk_level, create_results_dataframe
import pandas as pd


class TestCalculateRiskLevel:
    """Test risk level determination logic."""

    def test_risk_level_low(self):
        """Test low risk classification."""
        assert calculate_risk_level(0.15) == "LOW"
        assert calculate_risk_level(0.25) == "LOW"

    def test_risk_level_medium(self):
        """Test medium risk classification."""
        assert calculate_risk_level(0.35) == "MEDIUM"
        assert calculate_risk_level(0.45) == "MEDIUM"

    def test_risk_level_high(self):
        """Test high risk classification."""
        assert calculate_risk_level(0.65) == "HIGH"
        assert calculate_risk_level(0.85) == "HIGH"

    def test_risk_level_extreme_values(self):
        """Test extreme probability values."""
        assert calculate_risk_level(0.0) == "LOW"
        assert calculate_risk_level(1.0) == "HIGH"

    def test_risk_level_boundary_values(self):
        """Test boundary values."""
        # Around 0.30
        low_boundary = calculate_risk_level(0.29)
        assert low_boundary in ["LOW", "MEDIUM"]

        # Around 0.50
        medium_boundary = calculate_risk_level(0.49)
        assert medium_boundary in ["MEDIUM", "HIGH"]


class TestCreateResultsDataframe:
    """Test results dataframe creation."""

    def test_create_results_dataframe(self):
        """Test creating results dataframe."""
        sk_id_curr = pd.Series([100001, 100002, 100003])
        probabilities = [0.2, 0.4, 0.7]
        predictions = [0, 0, 1]

        df = create_results_dataframe(sk_id_curr, predictions, probabilities)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert 'SK_ID_CURR' in df.columns
        assert 'PROBABILITY' in df.columns
        assert 'RISK_LEVEL' in df.columns

    def test_create_results_dataframe_single_row(self):
        """Test creating results dataframe with single row."""
        sk_id_curr = pd.Series([100001])
        probabilities = [0.3]
        predictions = [0]

        df = create_results_dataframe(sk_id_curr, predictions, probabilities)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_create_results_dataframe_empty(self):
        """Test creating results dataframe with empty lists."""
        sk_id_curr = pd.Series([])
        probabilities = []
        predictions = []

        df = create_results_dataframe(sk_id_curr, predictions, probabilities)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
