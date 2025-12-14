"""
Tests for model validation utilities.
These tests ensure model availability checks work correctly.
"""
import pytest
from fastapi import HTTPException
from api.model_validator import ModelValidator


class MockModel:
    """Mock model for testing."""
    def predict_proba(self, X):
        return [[0.3, 0.7]]


class TestModelValidator:
    """Test the ModelValidator class."""
    
    def test_check_model_loaded_with_valid_model(self):
        """Test that valid model passes check."""
        model = MockModel()
        # Should not raise exception
        ModelValidator.check_model_loaded(model, "test endpoint")
    
    def test_check_model_loaded_with_none(self):
        """Test that None model raises 503 error."""
        with pytest.raises(HTTPException) as exc_info:
            ModelValidator.check_model_loaded(None, "test endpoint")
        
        assert exc_info.value.status_code == 503
        assert "Model not loaded" in exc_info.value.detail
    
    def test_validate_model_attributes_success(self):
        """Test that model with required attributes passes."""
        model = MockModel()
        # Should not raise exception
        ModelValidator.validate_model_attributes(model, ['predict_proba'])
    
    def test_validate_model_attributes_missing(self):
        """Test that missing attributes raises 500 error."""
        model = MockModel()
        
        with pytest.raises(HTTPException) as exc_info:
            ModelValidator.validate_model_attributes(
                model, 
                ['predict_proba', 'nonexistent_method']
            )
        
        assert exc_info.value.status_code == 500
        assert "missing required attributes" in exc_info.value.detail
        assert "nonexistent_method" in exc_info.value.detail
    
    def test_validate_multiple_missing_attributes(self):
        """Test error message with multiple missing attributes."""
        model = MockModel()
        
        with pytest.raises(HTTPException) as exc_info:
            ModelValidator.validate_model_attributes(
                model,
                ['method1', 'method2', 'method3']
            )
        
        assert exc_info.value.status_code == 500
        detail = exc_info.value.detail
        assert "method1" in detail
        assert "method2" in detail
        assert "method3" in detail
