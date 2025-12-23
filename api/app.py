"""Credit Scoring API

FastAPI application for serving credit scoring predictions.

Run with:
    poetry run uvicorn api.app:app --reload --port 8000

Then visit:
    - API docs: http://localhost:8000/docs
    - Health check: http://localhost:8000/health
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
import time
import os
import psutil
from pydantic import BaseModel, Field, validator

# Import routers
from api.batch_predictions import router as batch_router, production_logger
from api.drift_api import router as drift_router
from api.metrics import router as metrics_router, precompute_all_metrics
from api.file_validation import validate_input_data
from api.mlflow_loader import load_model_from_mlflow, get_mlflow_run_info
from api.utils.logging import log_prediction

# Import database components
from backend.database import engine, get_db_info
from backend.models import Base
from src.validation import DataValidationError, validate_prediction_probabilities

# Load all raw features for validation
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
with open(CONFIG_DIR / "all_raw_features.json") as f:
    RAW_FEATURES_CONFIG = json.load(f)

# Load the full list of 189 model features
with open(CONFIG_DIR / "model_features.txt") as f:
    ALL_MODEL_FEATURES = [line.strip() for line in f if line.strip()]

# Overwrite EXPECTED_FEATURES to be exactly the count from the model features
EXPECTED_FEATURES = len(ALL_MODEL_FEATURES)

# Load feature ranges for validation
FEATURE_RANGES = {}
if (CONFIG_DIR / "feature_ranges.json").exists():
    with open(CONFIG_DIR / "feature_ranges.json") as f:
        FEATURE_RANGES = json.load(f)

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

# Simple request size limit and rate limiting (in-memory, best-effort)
MAX_REQUEST_BODY = 10 * 1024 * 1024  # 10 MB
RATE_LIMIT_WINDOW_SEC = 60
RATE_LIMIT_MAX_REQUESTS = 600
_rate_limit_store: dict[str, list[float]] = {}

@app.middleware("http")
async def request_limits_middleware(request: Request, call_next):
    start_time = time.time()
    
    # Request body size guard (applies to uploads)
    try:
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_REQUEST_BODY:
            from starlette.responses import Response
            return Response(status_code=413, content="Payload Too Large")
    except Exception:
        pass

    # Very simple IP-based rate limiting
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SEC
    timestamps = _rate_limit_store.get(client_ip, [])
    timestamps = [t for t in timestamps if t >= window_start]
    if len(timestamps) >= RATE_LIMIT_MAX_REQUESTS:
        from starlette.responses import Response
        import logging
        logging.getLogger("uvicorn").warning(f"Rate limit exceeded for IP {client_ip}")
        return Response(status_code=429, content="Too Many Requests")
    timestamps.append(now)
    _rate_limit_store[client_ip] = timestamps

    response = await call_next(request)
    
    # Calculate duration and log if it was a prediction endpoint
    duration = (time.time() - start_time) * 1000
    response.headers["X-Process-Time"] = str(duration)
    
    return response

# Include routers
app.include_router(batch_router)
app.include_router(drift_router)
app.include_router(metrics_router)

# Global variables for model
model = None
model_metadata = {}
EXPECTED_FEATURES = 189  # Update based on your model


# ============================================================================
# Pydantic Models for Request/Response Validation
# ============================================================================

class PredictionInput(BaseModel):
    """Input schema for single prediction."""

    features: list[float] = Field(
        ...,
        description=f"List of {EXPECTED_FEATURES} feature values in correct order",
        min_items=EXPECTED_FEATURES,
        max_items=EXPECTED_FEATURES
    )
    feature_names: list[str] | None = Field(
        None,
        description="Optional list of feature names for validation"
    )
    client_id: str | None = Field(
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

    features: list[list[float]] = Field(
        ...,
        description="List of feature arrays for batch prediction"
    )
    client_ids: list[str] | None = Field(
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
    client_id: str | None = Field(None, description="Client ID if provided")
    timestamp: str = Field(..., description="Prediction timestamp")
    model_version: str = Field(..., description="Model version used")


class BatchPredictionOutput(BaseModel):
    """Output schema for batch predictions."""

    predictions: list[PredictionOutput]
    count: int = Field(..., description="Number of predictions")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_name: str
    model_version: str | None
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

    from api.onnx_wrapper import ONNXModelWrapper # Add this import

    print("Loading credit scoring model...")

    try:
        # Check for ONNX model first
        onnx_path = Path(__file__).parent.parent / "models" / "production_model.onnx"
        if onnx_path.exists():
            print(f"  Loading ONNX model from: {onnx_path}")
            model = ONNXModelWrapper(str(onnx_path))
            model_metadata.update({
                'source': 'onnx_runtime',
                'path': str(onnx_path),
                'loaded_at': datetime.now().isoformat(),
                'type': 'ONNX LGBMClassifier', # Specific to this model, adjust if model type can vary
                'features': model.n_features_in_
            })
            print(f"[OK] ONNX Model loaded")
        else:
            # Fallback to MLflow artifact or local pickle
            import pickle
            artifact_path = Path(__file__).parent.parent / "mlruns" / "7c" / "7ce7c8f6371e43af9ced637e5a4da7f0" / "artifacts" / "production_model.pkl"
            
            print(f"  Loading from: {artifact_path}")
            
            if artifact_path.exists():
                with open(artifact_path, 'rb') as f:
                    model = pickle.load(f)
                
                model_metadata.update({
                    'source': 'mlflow_artifacts',
                    'run_id': '7ce7c8f6371e43af9ced637e5a4da7f0',
                    'path': str(artifact_path),
                    'loaded_at': datetime.now().isoformat(),
                    'type': type(model).__name__,
                    'features': model.n_features_in_ if hasattr(model, 'n_features_in_') else EXPECTED_FEATURES
                })
                print(f"[OK] Model loaded from MLflow artifacts")
                print(f"  Type: {model_metadata['type']}, Features: {model_metadata['features']}")
            else:
                # Final fallback
                fallback_file = Path(__file__).parent.parent / "models" / "production_model.pkl"
                print(f"  Artifact not found, trying fallback: {fallback_file}")
                
                if not fallback_file.exists():
                    raise FileNotFoundError(f"Model file not found: {fallback_file}")
                
                with open(fallback_file, 'rb') as f:
                    model = pickle.load(f)
                
                model_metadata.update({
                    'source': 'file',
                    'path': str(fallback_file),
                    'loaded_at': datetime.now().isoformat(),
                    'type': type(model).__name__,
                    'features': model.n_features_in_ if hasattr(model, 'n_features_in_') else EXPECTED_FEATURES
                })
                print(f"[OK] Model loaded from fallback file")
                print(f"  Type: {model_metadata['type']}, Features: {model_metadata['features']}")

    except Exception as e: # This is the missing except block
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

    # Precompute model performance metrics (for /metrics endpoints)
    print("\nPrecomputing model performance metrics...")
    try:
        metrics_cache = precompute_all_metrics()
        print(f"Metrics precomputed successfully: {len(metrics_cache['all_metrics'])} thresholds")
    except Exception as e:
        print(f"Warning: Failed to precompute metrics: {e}")

    # Pre-load drift reference data
    print("\nPre-loading drift reference data (Training + Validation)...")
    try:
        from api.drift_detection import get_training_reference_data
        ref_df = get_training_reference_data()
        if not ref_df.empty:
            print(f"Drift reference data loaded: {len(ref_df)} rows")
        else:
            print("Warning: Drift reference data files not found.")
    except Exception as e:
        print(f"Warning: Failed to load drift reference data: {e}")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    print("Shutting down Credit Scoring API...")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint - API information."""
    return {
        "service": "Credit Scoring API",
        "version": "2.0.0",
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
        model_version=model_metadata.get('stage'),
        timestamp=datetime.now().isoformat()
    )


@app.get("/health/mlflow", tags=["General", "MLflow"])
async def mlflow_health():
    """MLflow connection and model information."""
    try:
        mlflow_info = get_mlflow_run_info(
            experiment_name="credit_scoring_final_delivery"
        )
        
        if 'error' in mlflow_info:
            return {
                'status': 'mlflow_unavailable',
                'message': mlflow_info['error'],
                'fallback': 'Using local model file'
            }
        
        return {
            'status': 'connected',
            'experiment': mlflow_info['experiment_name'],
            'run_id': mlflow_info['run_id'],
            'run_name': mlflow_info['run_name'],
            'model_status': mlflow_info['status'],
            'optimal_threshold': mlflow_info['parameters'].get('optimal_threshold'),
            'metrics': mlflow_info['metrics'],
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e),
            'fallback': 'Using local model file'
        }


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


@app.get("/health/resources", tags=["General"])
async def resources_health():
    """System and Process resource usage check."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    # System-wide memory
    sys_mem = psutil.virtual_memory()
    
    return {
        "process": {
            "memory_rss_mb": round(mem_info.rss / (1024 * 1024), 2),
            "memory_vms_mb": round(mem_info.vms / (1024 * 1024), 2),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "threads": process.num_threads(),
            "uptime_seconds": round(time.time() - process.create_time(), 2)
        },
        "system": {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": round(sys_mem.total / (1024**3), 2),
            "memory_used_percent": sys_mem.percent,
            "memory_available_gb": round(sys_mem.available / (1024**3), 2)
        },
        "timestamp": datetime.now().isoformat()
    }


def validate_input_feature_ranges_single_prediction(input_data: PredictionInput):
    """
    Validates input features against predefined ranges for a single prediction.
    Raises HTTPException if any feature is out of range.
    """
    if not FEATURE_RANGES:
        return # No ranges defined, skip validation

    # Map feature values to their corresponding names using ALL_MODEL_FEATURES order
    # Assuming input_data.features are ordered according to ALL_MODEL_FEATURES
    for i, value in enumerate(input_data.features):
        feature_name = ALL_MODEL_FEATURES[i] # Get feature name by index
        if feature_name in FEATURE_RANGES:
            rules = FEATURE_RANGES[feature_name]
            min_val = rules.get("min")
            max_val = rules.get("max")

            if (min_val is not None and value < min_val) or \
               (max_val is not None and value > max_val):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Feature '{feature_name}' value {value} is out of expected range "
                           f"[{min_val if min_val is not None else '-inf'}, {max_val if max_val is not None else '+inf'}]."
                )

@app.post(
    "/predict",
    response_model=PredictionOutput,
    tags=["Prediction"],
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        422: {"model": ErrorResponse, "description": "Validation error"}, # Added 422
        500: {"model": ErrorResponse, "description": "Prediction failed"}
    }
)
async def predict(input_data: PredictionInput):
    """Single credit scoring prediction.

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
        start_time = time.time()

        # Perform range validation
        validate_input_feature_ranges_single_prediction(input_data)

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

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Log prediction to predictions.jsonl
        try:
            log_prediction(
                logger=production_logger,
                sk_id_curr=int(input_data.client_id) if input_data.client_id else 0,
                probability=float(probability),
                prediction=int(prediction),
                risk_level=risk_level,
                processing_time_ms=processing_time_ms,
                source="api"
            )
        except Exception:
            pass  # Don't fail the request if logging fails

        return PredictionOutput(
            prediction=int(prediction),
            probability=float(probability),
            risk_level=risk_level,
            client_id=input_data.client_id,
            timestamp=datetime.now().isoformat(),
            model_version=model_metadata.get('stage', 'unknown')
        )
    except HTTPException: # Allow HTTPExceptions to pass through
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


# UNUSED: Replaced by /batch/predict router endpoint
# @app.post(
#     "/predict/batch",
#     response_model=BatchPredictionOutput,
#     tags=["Prediction"],
#     responses={
#         400: {"model": ErrorResponse, "description": "Invalid input"},
#         500: {"model": ErrorResponse, "description": "Batch prediction failed"}
#     }
# )
# async def predict_batch(input_data: BatchPredictionInput):
#     """Batch credit scoring predictions.
#
#     Predicts probabilities of credit default for multiple applications.
#
#     Args:
#         input_data: Batch prediction input
#
#     Returns:
#         Batch prediction output
#
#     Raises:
#         HTTPException: If model not loaded or prediction fails
#
#     """
#     if model is None:
#         raise HTTPException(
#             status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
#             detail="Model not loaded. Check server logs."
#         )
#
#     try:
#         # Convert input features to DataFrame for validation
#         input_df = pd.DataFrame(input_data.features, columns=ALL_RAW_FEATURES)
#
#         # Validate and clean the input DataFrame
#         validated_df = validate_input_data(input_df)
#
#         # Prepare features from the validated DataFrame
#         features = validated_df.to_numpy()
#
#         # Make predictions
#         predictions = model.predict(features)
#         probabilities = model.predict_proba(features)[:, 1]
#
#         # Validate probabilities
#         try:
#             validate_prediction_probabilities(probabilities)
#         except DataValidationError as e:
#             raise HTTPException(
#                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#                 detail=f"Model output validation failed: {e}"
#             )
#
#         # Build response
#         results = []
#         for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
#             # Classify risk level
#             if prob < 0.2:
#                 risk_level = "LOW"
#             elif prob < 0.4:
#                 risk_level = "MEDIUM"
#             elif prob < 0.6:
#                 risk_level = "HIGH"
#             else:
#                 risk_level = "CRITICAL"
#
#             client_id = None
#             if input_data.client_ids and i < len(input_data.client_ids):
#                 client_id = input_data.client_ids[i]
#
#             results.append(PredictionOutput(
#                 prediction=int(pred),
#                 probability=float(prob),
#                 risk_level=risk_level,
#                 client_id=client_id,
#                 timestamp=datetime.now().isoformat(),
#                 model_version=model_metadata.get('stage', 'unknown')
#             ))
#
#         return BatchPredictionOutput(
#             predictions=results,
#             count=len(results)
#         )
#
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Batch prediction failed: {str(e)}"
#         )


# UNUSED: Model info endpoint not used by Streamlit
# @app.get("/model/info", tags=["Model"])
# async def model_info():
#     """Get information about the loaded model.
#
#     Returns:
#         Model metadata and configuration
#
#     """
#     if model is None:
#         raise HTTPException(
#             status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
#             detail="Model not loaded"
#         )
#
#     return {
#         "model_metadata": model_metadata,
#         "expected_features": EXPECTED_FEATURES,
#         "model_type": type(model).__name__,
#         "capabilities": {
#             "single_prediction": True,
#             "batch_prediction": True,
#             "probability_scores": True
#         }
#     }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
