"""
Credit Scoring API

FastAPI application for serving credit scoring predictions.

Run with:
    poetry run uvicorn api.app:app --reload --port 8000

Then visit:
    - API docs: http://localhost:8000/docs
    - Health check: http://localhost:8000/health
"""
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
import mlflow
import mlflow.sklearn
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import json # Added json import

from src.config import MLFLOW_TRACKING_URI, REGISTERED_MODELS
from src.validation import validate_prediction_probabilities, DataValidationError

# Import batch predictions router
from api.batch_predictions import router as batch_router
from api.file_validation import validate_input_data # Imported validate_input_data

# Import database components
from backend.database import engine, get_db_info
from backend.models import Base

# Load all raw features for validation
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
with open(CONFIG_DIR / "all_raw_features.json", 'r') as f:
    ALL_RAW_FEATURES = json.load(f)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Scoring API",
    description="Machine Learning API for credit default prediction with batch processing from raw CSV files",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include batch predictions router
app.include_router(batch_router)

# Global variables for model
model = None
model_metadata = {}
EXPECTED_FEATURES = 189  # Update based on your model


# ============================================================================
# Pydantic Models for Request/Response Validation
# ============================================================================

class PredictionInput(BaseModel):
    """Input schema for single prediction."""
    features: List[float] = Field(
        ...,
        description=f"List of {EXPECTED_FEATURES} feature values in correct order",
        min_items=EXPECTED_FEATURES,
        max_items=EXPECTED_FEATURES
    )
    feature_names: Optional[List[str]] = Field(
        None,
        description="Optional list of feature names for validation"
    )
    client_id: Optional[str] = Field(
        None,
        description="Optional client ID for tracking"
    )

    @validator('features')
    def validate_features_not_nan(cls, v):
        """Validate features don't contain NaN or Inf."""
        arr = np.array(v)
        if np.isnan(arr).any():
            raise ValueError("Features contain NaN values")
        if np.isinf(arr).any():
            raise ValueError("Features contain infinite values")
        return v


class BatchPredictionInput(BaseModel):
    """Input schema for batch predictions."""
    features: List[List[float]] = Field(
        ...,
        description="List of feature arrays for batch prediction"
    )
    client_ids: Optional[List[str]] = Field(
        None,
        description="Optional list of client IDs"
    )

    @validator('features')
    def validate_batch_shape(cls, v):
        """Validate all feature vectors have same length."""
        if not v:
            raise ValueError("Features list is empty")

        lengths = [len(features) for features in v]
        if len(set(lengths)) > 1:
            raise ValueError(f"Inconsistent feature vector lengths: {set(lengths)}")

        if lengths[0] != EXPECTED_FEATURES:
            raise ValueError(
                f"Expected {EXPECTED_FEATURES} features, got {lengths[0]}"
            )

        return v


class PredictionOutput(BaseModel):
    """Output schema for single prediction."""
    prediction: int = Field(..., description="Predicted class (0=no default, 1=default)")
    probability: float = Field(..., description="Probability of default [0-1]")
    risk_level: str = Field(..., description="Risk category (LOW/MEDIUM/HIGH)")
    client_id: Optional[str] = Field(None, description="Client ID if provided")
    timestamp: str = Field(..., description="Prediction timestamp")
    model_version: str = Field(..., description="Model version used")


class BatchPredictionOutput(BaseModel):
    """Output schema for batch predictions."""
    predictions: List[PredictionOutput]
    count: int = Field(..., description="Number of predictions")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_name: str
    model_version: Optional[str]
    timestamp: str


class ErrorResponse(BaseModel):
    """Error response schema."""
    detail: str
    timestamp: str


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def load_model():
    """Load ML model, preprocessing pipeline, and initialize database on startup."""
    global model, model_metadata

    # Initialize database tables
    print("Initializing database...")
    try:
        Base.metadata.create_all(bind=engine)
        db_info = get_db_info()
        print(f"Database ready: {db_info['database_url']} (SQLite: {db_info['is_sqlite']})")
    except Exception as e:
        print(f"Warning: Database initialization failed: {e}")
        print("API will continue without database persistence.")

    print("Loading credit scoring model...")

    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        # Try multiple loading methods
        model_loaded = False

        # Method 1: Try loading from file (most reliable)
        model_file = Path(__file__).parent.parent / "models" / "production_model.pkl"
        if model_file.exists():
            try:
                import pickle
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                model_metadata['source'] = 'file'
                model_metadata['path'] = str(model_file)
                model_loaded = True
                print(f"Loaded model from file: {model_file}")
            except Exception as e:
                print(f"Failed to load from file: {e}")

        # Method 2: Try MLflow run URI if file failed
        if not model_loaded:
            try:
                # Use the run ID from our registration
                run_id = "83e2e1ec9b254fc59b4d3bfa7ae75b1f"
                model_uri = f"runs:/{run_id}/model"
                model = mlflow.lightgbm.load_model(model_uri)
                model_metadata['source'] = 'mlflow_run'
                model_metadata['run_id'] = run_id
                model_loaded = True
                print(f"Loaded model from MLflow run: {run_id}")
            except Exception as e:
                print(f"Failed to load from MLflow run: {e}")

        # Method 3: Try registry as last resort
        if not model_loaded:
            try:
                model_name = REGISTERED_MODELS['production']
                model_uri = f"models:/{model_name}/Production"
                model = mlflow.sklearn.load_model(model_uri)
                model_metadata['source'] = 'mlflow_registry'
                model_metadata['name'] = model_name
                model_loaded = True
                print(f"Loaded model from registry: {model_name}")
            except Exception as e:
                print(f"Failed to load from registry: {e}")

        if model_loaded:
            model_metadata.update({
                'loaded_at': datetime.now().isoformat(),
                'type': type(model).__name__,
                'features': model.n_features_in_ if hasattr(model, 'n_features_in_') else EXPECTED_FEATURES
            })
            print(f"Model loaded successfully: {model_metadata}")
        else:
            raise Exception("All loading methods failed")

    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        print("API will start but predictions will fail until model is loaded.")
        model = None
        model_metadata = {'error': str(e)}

    # Initialize preprocessing pipeline (loads precomputed features)
    print("\nInitializing preprocessing pipeline...")
    try:
        from api.batch_predictions import get_preprocessing_pipeline
        pipeline = get_preprocessing_pipeline()
        print("Preprocessing pipeline initialized successfully")
    except Exception as e:
        print(f"Warning: Failed to initialize preprocessing pipeline: {e}")
        print("Batch predictions may be slower on first request.")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    print("Shutting down Credit Scoring API...")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Credit Scoring API",
        "version": "1.0.0",
        "status": "active",
        "docs_url": "/docs",
        "health_url": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_name=model_metadata.get('name', 'unknown'),
        model_version=model_metadata.get('stage', None),
        timestamp=datetime.now().isoformat()
    )


@app.get("/health/database", tags=["General"])
async def database_health():
    """Database health check endpoint."""
    db_info = get_db_info()
    return {
        "status": "healthy" if db_info['connected'] else "unhealthy",
        "database_type": "SQLite" if db_info['is_sqlite'] else "PostgreSQL",
        "database_url": db_info['database_url'],
        "connected": db_info['connected'],
        "timestamp": datetime.now().isoformat()
    }


@app.post(
    "/predict",
    response_model=PredictionOutput,
    tags=["Prediction"],
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Prediction failed"}
    }
)
async def predict(input_data: PredictionInput):
    """
    Single credit scoring prediction.

    Predicts probability of credit default for a single application.

    Args:
        input_data: Prediction input with features

    Returns:
        Prediction output with probability and risk level

    Raises:
        HTTPException: If model not loaded or prediction fails
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Check server logs."
        )

    try:
        # Prepare features
        features = np.array(input_data.features).reshape(1, -1)

        # Make predictions
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0, 1]

        # Validate probability
        try:
            validate_prediction_probabilities(np.array([probability]))
        except DataValidationError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Model output validation failed: {e}"
            )

        # Classify risk level (LOW < 15%, MEDIUM 15-30%, HIGH >= 30%)
        if probability < 0.30:
            risk_level = "LOW"
        elif probability < 0.50:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        return PredictionOutput(
            prediction=int(prediction),
            probability=float(probability),
            risk_level=risk_level,
            client_id=input_data.client_id,
            timestamp=datetime.now().isoformat(),
            model_version=model_metadata.get('stage', 'unknown')
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionOutput,
    tags=["Prediction"],
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Batch prediction failed"}
    }
)
async def predict_batch(input_data: BatchPredictionInput):
    """
    Batch credit scoring predictions.

    Predicts probabilities of credit default for multiple applications.

    Args:
        input_data: Batch prediction input

    Returns:
        Batch prediction output

    Raises:
        HTTPException: If model not loaded or prediction fails
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Check server logs."
        )

    try:
        # Convert input features to DataFrame for validation
        input_df = pd.DataFrame(input_data.features, columns=ALL_RAW_FEATURES)
        
        # Validate and clean the input DataFrame
        validated_df = validate_input_data(input_df)

        # Prepare features from the validated DataFrame
        features = validated_df.to_numpy()

        # Make predictions
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)[:, 1]

        # Validate probabilities
        try:
            validate_prediction_probabilities(probabilities)
        except DataValidationError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Model output validation failed: {e}"
            )

        # Build response
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            # Classify risk level
            if prob < 0.2:
                risk_level = "LOW"
            elif prob < 0.4:
                risk_level = "MEDIUM"
            elif prob < 0.6:
                risk_level = "HIGH"
            else:
                risk_level = "CRITICAL"

            client_id = None
            if input_data.client_ids and i < len(input_data.client_ids):
                client_id = input_data.client_ids[i]

            results.append(PredictionOutput(
                prediction=int(pred),
                probability=float(prob),
                risk_level=risk_level,
                client_id=client_id,
                timestamp=datetime.now().isoformat(),
                model_version=model_metadata.get('stage', 'unknown')
            ))

        return BatchPredictionOutput(
            predictions=results,
            count=len(results)
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/model/info", tags=["Model"])
async def model_info():
    """
    Get information about the loaded model.

    Returns:
        Model metadata and configuration
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    return {
        "model_metadata": model_metadata,
        "expected_features": EXPECTED_FEATURES,
        "model_type": type(model).__name__,
        "capabilities": {
            "single_prediction": True,
            "batch_prediction": True,
            "probability_scores": True
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
