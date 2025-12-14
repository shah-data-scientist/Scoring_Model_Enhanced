"""
Model validation utilities to ensure model is available and functional.
"""
from typing import Optional
import logging
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)


class ModelValidator:
    """Validates model availability and functionality."""
    
    @staticmethod
    def check_model_loaded(model: Optional[object], endpoint_name: str = "this endpoint") -> None:
        """
        Check if model is loaded and raise appropriate error if not.
        
        Args:
            model: The model object to check
            endpoint_name: Name of the endpoint for error message
            
        Raises:
            HTTPException: 503 Service Unavailable if model is not loaded
        """
        if model is None:
            error_msg = f"Model not loaded. {endpoint_name} requires a trained model."
            logger.error(error_msg)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=error_msg
            )
        
        logger.debug(f"Model validation passed for {endpoint_name}")
    
    @staticmethod
    def validate_model_attributes(model: object, required_attrs: list[str]) -> None:
        """
        Validate that model has required attributes.
        
        Args:
            model: The model object to check
            required_attrs: List of attribute names that must exist
            
        Raises:
            HTTPException: 500 Internal Server Error if attributes missing
        """
        missing_attrs = [attr for attr in required_attrs if not hasattr(model, attr)]
        
        if missing_attrs:
            error_msg = f"Model missing required attributes: {', '.join(missing_attrs)}"
            logger.error(error_msg)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_msg
            )
        
        logger.debug(f"Model has all required attributes: {required_attrs}")
