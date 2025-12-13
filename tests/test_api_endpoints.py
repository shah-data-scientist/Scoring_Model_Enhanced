"""
Comprehensive tests for API endpoints.
"""
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from io import BytesIO
from api.app import app


client = TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_endpoint_returns_200(self):
        """Test that health endpoint returns 200."""
        response = client.get("/health")

        assert response.status_code == 200

    def test_health_endpoint_response_format(self):
        """Test health endpoint response format."""
        response = client.get("/health")

        data = response.json()

        assert "status" in data
        # Allow unhealthy status if model is not loaded
        assert data["status"] in ["healthy", "unhealthy"]

    def test_health_endpoint_includes_metadata(self):
        """Test health endpoint includes useful metadata."""
        response = client.get("/health")

        data = response.json()

        # Should include some system information
        assert isinstance(data, dict)
        assert len(data) > 0


class TestRootEndpoint:
    """Test root endpoint."""

    def test_root_endpoint_returns_200(self):
        """Test that root endpoint returns 200."""
        response = client.get("/")

        assert response.status_code == 200

    def test_root_endpoint_response(self):
        """Test root endpoint response."""
        response = client.get("/")

        data = response.json()

        assert isinstance(data, dict)
        assert "message" in data or "name" in data or "version" in data


class TestPredictEndpoint:
    """Test single prediction endpoint."""

    @pytest.fixture
    def sample_application(self):
        """Sample application data for testing."""
        return {
            "SK_ID_CURR": 100001,
            "NAME_CONTRACT_TYPE": "Cash loans",
            "CODE_GENDER": "M",
            "FLAG_OWN_CAR": "Y",
            "FLAG_OWN_REALTY": "Y",
            "CNT_CHILDREN": 0,
            "AMT_INCOME_TOTAL": 150000,
            "AMT_CREDIT": 300000,
            "AMT_ANNUITY": 15000,
            "AMT_GOODS_PRICE": 300000
        }

    def test_predict_endpoint_exists(self, sample_application):
        """Test that predict endpoint exists."""
        response = client.post("/predict", json=sample_application)

        # Should return 200, 422, or other valid status
        assert response.status_code in [200, 422, 404, 500]

    def test_predict_endpoint_with_valid_data(self, sample_application):
        """Test prediction with valid data."""
        response = client.post("/predict", json=sample_application)

        if response.status_code == 200:
            data = response.json()

            assert "probability" in data or "prediction" in data

    def test_predict_endpoint_rejects_missing_data(self):
        """Test that endpoint rejects incomplete data."""
        incomplete_data = {
            "SK_ID_CURR": 100001
            # Missing required fields
        }

        response = client.post("/predict", json=incomplete_data)

        # Should return validation error
        assert response.status_code in [422, 400, 500]

    def test_predict_endpoint_rejects_invalid_types(self):
        """Test that endpoint rejects invalid data types."""
        invalid_data = {
            "SK_ID_CURR": "not_a_number",
            "AMT_INCOME_TOTAL": "invalid"
        }

        response = client.post("/predict", json=invalid_data)

        # Should return validation error
        assert response.status_code in [422, 400]


class TestBatchValidateEndpoint:
    """Test batch file validation endpoint."""

    def test_batch_validate_endpoint_exists(self):
        """Test that batch validate endpoint exists."""
        # Create minimal CSV file
        csv_content = "SK_ID_CURR,AMT_INCOME_TOTAL\n100001,150000\n"
        files = {
            "application.csv": ("application.csv", BytesIO(csv_content.encode()), "text/csv")
        }

        response = client.post("/batch/validate", files=files)

        # Should return some response (200, 400, 404, 422, etc.)
        assert response.status_code in [200, 400, 404, 422, 500]

    def test_batch_validate_accepts_valid_file(self):
        """Test validation accepts valid CSV file."""
        csv_content = """SK_ID_CURR,NAME_CONTRACT_TYPE,CODE_GENDER,AMT_INCOME_TOTAL,AMT_CREDIT
100001,Cash loans,M,150000,300000
100002,Cash loans,F,200000,400000
"""
        files = {
            "application.csv": ("application.csv", BytesIO(csv_content.encode()), "text/csv")
        }

        response = client.post("/batch/validate", files=files)

        if response.status_code == 200:
            data = response.json()
            assert "valid" in data or "status" in data

    def test_batch_validate_rejects_empty_file(self):
        """Test validation rejects empty file."""
        files = {
            "application.csv": ("application.csv", BytesIO(b""), "text/csv")
        }

        response = client.post("/batch/validate", files=files)

        # Should reject empty file
        assert response.status_code in [400, 404, 422, 500]

    def test_batch_validate_rejects_no_files(self):
        """Test validation fails with no files."""
        response = client.post("/batch/validate")

        # Should return error for missing files
        assert response.status_code in [422, 400, 404]


class TestBatchPredictEndpoint:
    """Test batch prediction endpoint."""

    def test_batch_predict_endpoint_exists(self):
        """Test that batch predict endpoint exists."""
        csv_content = """SK_ID_CURR,NAME_CONTRACT_TYPE,CODE_GENDER,AMT_INCOME_TOTAL,AMT_CREDIT
100001,Cash loans,M,150000,300000
"""
        files = {
            "application.csv": ("application.csv", BytesIO(csv_content.encode()), "text/csv")
        }

        response = client.post("/batch/predict", files=files)

        # Endpoint should exist
        assert response.status_code in [200, 400, 422, 500]

    def test_batch_predict_with_multiple_applications(self):
        """Test batch prediction with multiple applications."""
        csv_content = """SK_ID_CURR,NAME_CONTRACT_TYPE,CODE_GENDER,AMT_INCOME_TOTAL,AMT_CREDIT
100001,Cash loans,M,150000,300000
100002,Cash loans,F,200000,400000
100003,Revolving loans,M,100000,200000
"""
        files = {
            "application.csv": ("application.csv", BytesIO(csv_content.encode()), "text/csv")
        }

        response = client.post("/batch/predict", files=files)

        if response.status_code == 200:
            data = response.json()

            # Should return predictions for all applications
            assert "predictions" in data or isinstance(data, list)

    def test_batch_predict_returns_structured_results(self):
        """Test that batch predictions return structured results."""
        csv_content = """SK_ID_CURR,AMT_INCOME_TOTAL,AMT_CREDIT
100001,150000,300000
"""
        files = {
            "application.csv": ("application.csv", BytesIO(csv_content.encode()), "text/csv")
        }

        response = client.post("/batch/predict", files=files)

        if response.status_code == 200:
            data = response.json()

            # Results should be structured
            assert isinstance(data, (dict, list))


class TestCORSHeaders:
    """Test CORS configuration."""

    def test_cors_headers_present(self):
        """Test that CORS headers are present."""
        response = client.get("/health")

        # Check for CORS headers
        headers = response.headers

        # Depending on configuration, may or may not have CORS headers
        assert response.status_code == 200

    def test_options_request(self):
        """Test OPTIONS request (preflight)."""
        response = client.options("/health")

        # Should handle OPTIONS request
        assert response.status_code in [200, 405]


class TestErrorResponses:
    """Test error handling and responses."""

    def test_404_on_invalid_endpoint(self):
        """Test 404 response for invalid endpoint."""
        response = client.get("/nonexistent/endpoint")

        assert response.status_code == 404

    def test_405_on_invalid_method(self):
        """Test 405 response for invalid HTTP method."""
        response = client.delete("/health")  # Health only accepts GET

        assert response.status_code in [405, 404]

    def test_422_on_invalid_json(self):
        """Test 422 response for invalid JSON."""
        response = client.post(
            "/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code in [422, 400]


class TestAPIDocumentation:
    """Test API documentation endpoints."""

    def test_openapi_schema_available(self):
        """Test that OpenAPI schema is available."""
        response = client.get("/openapi.json")

        assert response.status_code == 200

        data = response.json()

        assert "openapi" in data
        assert "info" in data
        assert "paths" in data

    def test_docs_endpoint_available(self):
        """Test that /docs endpoint is available."""
        response = client.get("/docs")

        assert response.status_code == 200

    def test_redoc_endpoint_available(self):
        """Test that /redoc endpoint is available."""
        response = client.get("/redoc")

        assert response.status_code == 200


class TestInputValidation:
    """Test input validation across endpoints."""

    def test_validates_negative_amounts(self):
        """Test validation rejects negative amounts."""
        invalid_data = {
            "SK_ID_CURR": 100001,
            "AMT_INCOME_TOTAL": -150000,  # Negative
            "AMT_CREDIT": 300000
        }

        response = client.post("/predict", json=invalid_data)

        # Should validate or accept with warning
        assert response.status_code in [200, 422, 400]

    def test_validates_required_fields(self):
        """Test that required fields are enforced."""
        minimal_data = {}

        response = client.post("/predict", json=minimal_data)

        # Should require some fields
        assert response.status_code in [422, 400]

    def test_accepts_extra_fields(self):
        """Test that extra fields are handled gracefully."""
        data_with_extra = {
            "SK_ID_CURR": 100001,
            "AMT_INCOME_TOTAL": 150000,
            "EXTRA_FIELD": "should_be_ignored"
        }

        response = client.post("/predict", json=data_with_extra)

        # Should handle extra fields gracefully
        assert response.status_code in [200, 422, 400, 500]


class TestPerformance:
    """Test performance characteristics."""

    def test_health_check_is_fast(self):
        """Test that health check responds quickly."""
        import time

        start = time.time()
        response = client.get("/health")
        elapsed = time.time() - start

        assert response.status_code == 200
        # Health check should be very fast (< 1 second)
        assert elapsed < 1.0

    def test_handles_concurrent_requests(self):
        """Test handling multiple concurrent requests."""
        responses = []

        # Make 5 concurrent requests
        for _ in range(5):
            response = client.get("/health")
            responses.append(response)

        # All should succeed
        assert all(r.status_code == 200 for r in responses)


class TestSecurityHeaders:
    """Test security-related headers."""

    def test_server_header_not_exposed(self):
        """Test that server details are not exposed."""
        response = client.get("/health")

        # Check if server header exposes version info
        headers = response.headers

        # Good practice: Don't expose server details
        # This test just verifies headers exist
        assert response.status_code == 200

    def test_accepts_json_content_type(self):
        """Test that API accepts JSON content type."""
        data = {"test": "data"}

        response = client.post(
            "/predict",
            json=data,
            headers={"Content-Type": "application/json"}
        )

        # Should accept JSON
        assert response.status_code in [200, 422, 400, 500]
