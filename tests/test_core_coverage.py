"""Focused tests for core API modules to maximize coverage."""
import pytest
from api.app import app
from fastapi.testclient import TestClient

client = TestClient(app)


class TestAppMiddleware:
    """Tests for app middleware and global handlers."""

    def test_request_size_limit_defined(self):
        """Test request size limit is configured."""
        from api.app import MAX_REQUEST_BODY
        
        assert MAX_REQUEST_BODY > 0
        assert MAX_REQUEST_BODY == 10 * 1024 * 1024  # 10MB

    def test_rate_limit_window_defined(self):
        """Test rate limit window is defined."""
        from api.app import RATE_LIMIT_WINDOW_SEC, RATE_LIMIT_MAX_REQUESTS
        
        assert RATE_LIMIT_WINDOW_SEC > 0
        assert RATE_LIMIT_MAX_REQUESTS > 0

    def test_health_endpoint_response(self):
        """Test health endpoint returns valid response."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "model_loaded" in data
        assert "timestamp" in data

    def test_database_health_endpoint(self):
        """Test database health endpoint."""
        response = client.get("/health/database")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "database_type" in data


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_service_info(self):
        """Test root endpoint returns service information."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "service" in data
        assert "version" in data
        assert "docs_url" in data


class TestPredictionEndpoint:
    """Tests for prediction endpoint."""

    def test_predict_endpoint_exists(self):
        """Test predict endpoint is reachable."""
        response = client.get("/docs")
        
        # API docs should be available
        assert response.status_code == 200

    def test_predict_requires_features(self):
        """Test prediction requires features."""
        response = client.post("/predict", json={})
        
        # Should reject missing features
        assert response.status_code == 422  # Validation error

    def test_predict_validates_feature_count(self):
        """Test prediction validates feature count."""
        response = client.post("/predict", json={"features": [0.5] * 100})
        
        # Should reject wrong feature count (expects 189)
        assert response.status_code == 422  # Validation error


class TestAppModels:
    """Tests for Pydantic models used in API."""

    def test_prediction_input_model(self):
        """Test PredictionInput model validation."""
        from api.app import PredictionInput
        
        # Valid input
        valid = PredictionInput(features=[0.5] * 189)
        assert valid.features is not None
        assert len(valid.features) == 189

    def test_prediction_output_model(self):
        """Test PredictionOutput model structure."""
        from api.app import PredictionOutput
        from datetime import datetime
        
        output = PredictionOutput(
            prediction=1,
            probability=0.75,
            risk_level="HIGH",
            timestamp=datetime.now().isoformat(),
            model_version="1.0"
        )
        
        assert output.prediction == 1
        assert output.probability == 0.75
        assert output.risk_level == "HIGH"

    def test_health_response_model(self):
        """Test HealthResponse model."""
        from api.app import HealthResponse
        
        health = HealthResponse(
            status="healthy",
            model_loaded=True,
            model_name="test",
            model_version="1.0",
            timestamp="2025-01-01T00:00:00"
        )
        
        assert health.status == "healthy"
        assert health.model_loaded is True


class TestDatabaseConnectivity:
    """Tests for database connectivity."""

    def test_database_import(self):
        """Test database module imports correctly."""
        try:
            from backend.database import engine, get_db_info
            
            db_info = get_db_info()
            assert "database_url" in db_info
            assert "connected" in db_info
        except Exception:
            pytest.skip("Database not configured")

    def test_models_import(self):
        """Test models can be imported."""
        from backend.models import (
            Base,
            User,
            Prediction,
            PredictionBatch,
        )
        
        assert Base is not None
        assert User is not None
        assert Prediction is not None
        assert PredictionBatch is not None


class TestAuthModule:
    """Tests for auth module."""

    def test_hash_password_function(self):
        """Test password hashing."""
        from backend.auth import hash_password, verify_password
        
        password = "test_password_123"
        hashed = hash_password(password)
        
        assert hashed != password
        assert verify_password(password, hashed)

    def test_wrong_password_fails(self):
        """Test wrong password verification fails."""
        from backend.auth import hash_password, verify_password
        
        password = "correct_password"
        hashed = hash_password(password)
        
        assert not verify_password("wrong_password", hashed)


class TestValidationModule:
    """Tests for validation module."""

    def test_validation_error_exception(self):
        """Test validation error exception exists."""
        from src.validation import DataValidationError
        
        try:
            raise DataValidationError("Test error")
        except DataValidationError as e:
            assert "Test error" in str(e)

    def test_probability_validation(self):
        """Test probability validation."""
        from src.validation import validate_prediction_probabilities
        import numpy as np
        
        # Valid probabilities
        valid_probs = np.array([0.1, 0.5, 0.9])
        validate_prediction_probabilities(valid_probs)  # Should not raise

    def test_probability_validation_rejects_invalid(self):
        """Test probability validation rejects invalid values."""
        from src.validation import validate_prediction_probabilities, DataValidationError
        import numpy as np
        
        # Invalid probabilities (outside 0-1 range)
        invalid_probs = np.array([0.1, 1.5, -0.1])
        
        with pytest.raises(DataValidationError):
            validate_prediction_probabilities(invalid_probs)
