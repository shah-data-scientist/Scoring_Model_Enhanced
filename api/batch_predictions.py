"""Batch Predictions Endpoint for Raw CSV Files

Handles batch predictions from raw CSV uploads:
1. Accept 7 CSV files
2. Validate files and columns
3. Preprocess data
4. Make predictions
5. Store results in database
6. Return results with risk levels
"""

import io
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
# import api.app as main_app # Moved inside function to avoid circular import
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.file_validation import get_file_summaries, validate_all_files
from api.json_utils import dataframe_to_json_safe, sanitize_for_json
from api.model_validator import ModelValidator
from api.preprocessing_pipeline import PreprocessingPipeline
from api.utils.logging import setup_production_logger, log_batch_prediction, log_error
from backend import crud
from backend.database import get_db

# SHAP will be imported lazily inside get_shap_explainer

# Load the full list of 189 model features
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
with open(CONFIG_DIR / "model_features.txt") as f:
    ALL_MODEL_FEATURES = [line.strip() for line in f if line.strip()]

# Create router
router = APIRouter(prefix="/batch", tags=["Batch Predictions"])
production_logger = setup_production_logger()
logger = logging.getLogger(__name__)

# Upload size limit (per file)
MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB

# Global preprocessing pipeline
_preprocessing_pipeline = None

# Cached SHAP explainer for performance
_shap_explainer = None
_shap_explainer_model_id = None


def get_preprocessing_pipeline():
    """Get or create preprocessing pipeline instance (Singleton)."""
    global _preprocessing_pipeline
    if _preprocessing_pipeline is None:
        logger.info("Initializing PreprocessingPipeline (optimized mode with precomputed features)...")
        _preprocessing_pipeline = PreprocessingPipeline(use_precomputed=True)
    return _preprocessing_pipeline


def get_shap_explainer(model):
    """Get or create cached SHAP explainer. Loads pickle model and background data for probability output."""
    global _shap_explainer, _shap_explainer_model_id

    try:
        import shap
    except ImportError:
        return None

    # Handle ONNX models: Load the original pickle model for SHAP
    from api.onnx_wrapper import ONNXModelWrapper
    original_model = model
    if isinstance(model, ONNXModelWrapper):
        if _shap_explainer is not None:
             return _shap_explainer
             
        try:
            import pickle
            model_path = PROJECT_ROOT / "models" / "production_model.pkl"
            if not model_path.exists():
                return None
            with open(model_path, 'rb') as f:
                original_model = pickle.load(f)
        except Exception:
            return None

    model_id = id(original_model)
    if _shap_explainer is None or _shap_explainer_model_id != model_id:
        try:
            # Load small background sample for probability output support
            background_data = None
            try:
                train_path = PROJECT_ROOT / "data" / "processed" / "X_train.csv"
                if train_path.exists():
                    background_data = pd.read_csv(train_path, nrows=100)
                    # Filter to numeric only and match model features
                    background_data = background_data.select_dtypes(include=[np.number])
                    if hasattr(original_model, 'feature_name_'):
                        background_data = background_data[original_model.feature_name_]
            except Exception as e:
                logger.warning(f"Could not load background data for SHAP: {e}")

            # Initialize with background data if available
            if background_data is not None:
                _shap_explainer = shap.TreeExplainer(original_model, background_data, model_output='probability')
            else:
                _shap_explainer = shap.TreeExplainer(original_model, model_output='probability')
            
            _shap_explainer_model_id = model_id
            logger.info("SHAP TreeExplainer initialized (probability mode)")
        except Exception as e:
            try:
                _shap_explainer = shap.TreeExplainer(original_model)
                _shap_explainer_model_id = model_id
                logger.warning(f"SHAP probability mode failed, using log-odds: {e}")
            except Exception as e2:
                logger.error(f"Critical SHAP failure: {e2}")
                return None

    return _shap_explainer


# ============================================================================
# Pydantic Models
# ============================================================================

class BatchPredictionResult(BaseModel):
    """Single prediction result."""

    sk_id_curr: int = Field(..., description="Application ID")
    prediction: int = Field(..., description="Predicted class (0=no default, 1=default)")
    probability: float = Field(..., description="Probability of default [0-1]")
    risk_level: str = Field(..., description="Risk category (LOW/MEDIUM/HIGH)")

class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""

    success: bool
    timestamp: str
    n_applications: int
    n_predictions: int
    file_summaries: dict
    predictions: list[BatchPredictionResult]
    download_url: str | None = None
    model_version: str
    batch_id: int | None = None  # Database batch ID for retrieval


class ValidationResponse(BaseModel):
    """Response for file validation."""

    success: bool
    timestamp: str
    files_validated: list[str]
    file_summaries: dict
    critical_columns_check: dict
    message: str


# ============================================================================
# Helper Functions
# ============================================================================

def calculate_risk_level(probability: float) -> str:
    """Calculate risk level from probability.

    Args:
        probability: Probability of default [0-1]

    Returns:
        Risk level string (LOW, MEDIUM, or HIGH)

    """
    if probability < 0.30:
        return "LOW"
    if probability < 0.50:
        return "MEDIUM"
    return "HIGH"


def create_results_dataframe(
    sk_id_curr: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray
) -> pd.DataFrame:
    """Create results DataFrame with predictions and risk levels.

    Args:
        sk_id_curr: Series of application IDs
        predictions: Predicted classes
        probabilities: Predicted probabilities

    Returns:
        Results DataFrame

    """
    risk_levels = [calculate_risk_level(p) for p in probabilities]

    results_df = pd.DataFrame({
        'SK_ID_CURR': sk_id_curr,
        'PREDICTION': predictions,
        'PROBABILITY': probabilities,
        'RISK_LEVEL': risk_levels
    })

    return results_df


def dataframe_to_csv_stream(df: pd.DataFrame) -> io.BytesIO:
    """Convert DataFrame to CSV stream for download.

    Args:
        df: DataFrame to convert

    Returns:
        BytesIO stream

    """
    stream = io.BytesIO()
    df.to_csv(stream, index=False)
    stream.seek(0)
    return stream


# ============================================================================
# API Endpoints
# ============================================================================

# UNUSED: CSV validation not used by Streamlit
# @router.post("/validate", response_model=ValidationResponse)
# async def validate_files(
#     application: UploadFile = File(..., description="application.csv"),
#     bureau: UploadFile = File(..., description="bureau.csv"),
#     bureau_balance: UploadFile = File(..., description="bureau_balance.csv"),
#     previous_application: UploadFile = File(..., description="previous_application.csv"),
#     credit_card_balance: UploadFile = File(..., description="credit_card_balance.csv"),
#     installments_payments: UploadFile = File(..., description="installments_payments.csv"),
#     pos_cash_balance: UploadFile = File(..., description="POS_CASH_balance.csv")
# ):
#     """Validate uploaded CSV files without making predictions.
#
#     Checks:
#     - All required files present
#     - File structure valid
#     - Critical columns present in application.csv
#
#     Returns:
#         Validation results
#
#     """
#     # Organize uploaded files
#     uploaded_files = {
#         'application.csv': application,
#         'bureau.csv': bureau,
#         'bureau_balance.csv': bureau_balance,
#         'previous_application.csv': previous_application,
#         'credit_card_balance.csv': credit_card_balance,
#         'installments_payments.csv': installments_payments,
#         'POS_CASH_balance.csv': pos_cash_balance
#     }
#
#     # Validate files
#     dataframes = validate_all_files(uploaded_files)
#
#     # Get file summaries
#     summaries = get_file_summaries(dataframes)
#
#     # Check critical columns in application.csv
#     from api.file_validation import validate_application_columns
#     is_valid, missing_cols, coverage = validate_application_columns(
#         dataframes['application.csv']
#     )
#
#     return ValidationResponse(
#         success=True,
#         timestamp=datetime.now().isoformat(),
#         files_validated=list(uploaded_files.keys()),
#         file_summaries=summaries,
#         critical_columns_check={
#             "valid": is_valid,
#             "coverage": f"{coverage*100:.1f}%",
#             "missing_columns": missing_cols if missing_cols else []
#         },
#         message="All files validated successfully"
#     )


@router.post("/predict", response_model=BatchPredictionResponse)
async def predict_batch(
    application: UploadFile = File(..., description="application.csv"),
    bureau: UploadFile = File(..., description="bureau.csv"),
    bureau_balance: UploadFile = File(..., description="bureau_balance.csv"),
    previous_application: UploadFile = File(..., description="previous_application.csv"),
    credit_card_balance: UploadFile = File(..., description="credit_card_balance.csv"),
    installments_payments: UploadFile = File(..., description="installments_payments.csv"),
    pos_cash_balance: UploadFile = File(..., description="POS_CASH_balance.csv"),
    db: Session = Depends(get_db)
):
    """Batch credit scoring predictions from raw CSV files.

    Accepts 7 CSV files, preprocesses them, stores raw data, and returns predictions.

    Args:
        application: Main application data CSV
        bureau: Bureau credit history CSV
        bureau_balance: Bureau monthly balance CSV
        previous_application: Previous applications CSV
        credit_card_balance: Credit card balance CSV
        installments_payments: Payment installments CSV
        pos_cash_balance: POS/cash balance CSV
        db: Database session (injected)

    Returns:
        Batch prediction results with risk levels

    """
    batch = None
    start_time = time.time()
    
    # Get model from main app and validate
    import api.app as main_app
    model = main_app.model
    ModelValidator.check_model_loaded(model, "Batch prediction")
    ModelValidator.validate_model_attributes(model, ['predict_proba'])
    
    logger.info(f"Received batch prediction request")
    logger.info(f"Files received: application={application.filename}, bureau={bureau.filename}, "
                f"bureau_balance={bureau_balance.filename}, previous={previous_application.filename}, "
                f"credit_card={credit_card_balance.filename}, installments={installments_payments.filename}, "
                f"pos_cash={pos_cash_balance.filename}")

    try:
        # Step 1: Organize uploaded files (do not require model yet)
        uploaded_files = {
            'application.csv': application,
            'bureau.csv': bureau,
            'bureau_balance.csv': bureau_balance,
            'previous_application.csv': previous_application,
            'credit_card_balance.csv': credit_card_balance,
            'installments_payments.csv': installments_payments,
            'POS_CASH_balance.csv': pos_cash_balance
        }

        # Step 2: Validate files
        try:
            t_start = time.time()
            dataframes = validate_all_files(uploaded_files)
            summaries = get_file_summaries(dataframes)
            logger.info(f"TIMING: Validation and summarization took {time.time() - t_start:.2f}s")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File validation failed: {str(e)}"
            )

        # Step 3: Create batch record in database
        batch = crud.create_prediction_batch(
            db=db,
            user_id=None,  # Will be populated when auth is integrated
            batch_name=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            total_applications=len(dataframes['application.csv'])
        )
        crud.start_batch_processing(db, batch.id)

        # Step 4: Store raw application data in database
        try:
            t_start = time.time()
            app_records = dataframes['application.csv'].to_dict('records')
            crud.store_raw_applications_bulk(db, batch.id, app_records)
            logger.info(f"TIMING: Storing raw data took {time.time() - t_start:.2f}s")
        except Exception as e:
            logger.error(f"Error storing raw data: {e}", exc_info=True)
            # Continue anyway - predictions are more important

        # Step 5: Preprocess data
        try:
            t_start = time.time()
            pipeline = get_preprocessing_pipeline()
            features_df, sk_id_curr = pipeline.process(dataframes, keep_sk_id=False)

            # Ensure features are in correct format
            X = features_df.values
            feature_names = list(features_df.columns)
            logger.info(f"TIMING: Preprocessing took {time.time() - t_start:.2f}s")

        except Exception as e:
            crud.fail_batch(db, batch.id, f"Preprocessing failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Preprocessing error: {str(e)}"
            )

        # Step 6: Make predictions
        try:
            t_start = time.time()
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1]
            logger.info(f"TIMING: Model prediction took {time.time() - t_start:.2f}s")

        except Exception as e:
            crud.fail_batch(db, batch.id, f"Prediction failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {str(e)}"
            )

        # Step 6.5: Compute SHAP values (optional, for explainability)
        # Skip for very large batches to ensure performance
        shap_values_list = None
        
        # Lazy check for shap availability
        shap_available = False
        try:
            import shap
            shap_available = True
        except ImportError:
            shap_available = False

        if shap_available and len(X) <= 1000:
            try:
                t_start = time.time()
                # Use cached explainer for performance
                explainer = get_shap_explainer(model)
                if explainer is not None:
                    shap_values = explainer.shap_values(X)

                    # For binary classification, get values for positive class
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]  # Class 1 (default)

                    shap_values_list = shap_values
                    logger.info(f"TIMING: SHAP computation took {time.time() - t_start:.2f}s for {len(X)} predictions")

            except Exception as e:
                logger.warning(f"SHAP computation skipped: {e}")
                shap_values_list = None
        elif len(X) > 1000:
            logger.info(f"Skipping SHAP for batch size {len(X)} (limit: 1000) to ensure performance")

        # Step 7: Create results
        results_df = create_results_dataframe(sk_id_curr, predictions, probabilities)

        # Step 8: Store predictions in database with SHAP values (vectorized)
        try:
            t_start = time.time()

            # Vectorized data extraction - avoid iterrows()
            sk_ids = results_df['SK_ID_CURR'].values
            preds = results_df['PREDICTION'].values
            probs = results_df['PROBABILITY'].values
            risks = results_df['RISK_LEVEL'].values

            predictions_data = []
            for i in range(len(sk_ids)):
                pred_data = {
                    'sk_id_curr': int(sk_ids[i]),
                    'prediction': int(preds[i]),
                    'probability': float(probs[i]),
                    'risk_level': risks[i]
                }

                # Add SHAP values if available
                if shap_values_list is not None:
                    shap_row = shap_values_list[i]
                    # Vectorized NaN/inf check
                    valid_mask = ~(np.isnan(shap_row) | np.isinf(shap_row))
                    shap_dict = {feat_name: (float(shap_row[j]) if valid_mask[j] else None)
                                 for j, feat_name in enumerate(feature_names)}
                    pred_data['shap_values'] = shap_dict
                    
                    # Add base value (expected value) for waterfall plots
                    explainer = get_shap_explainer(model)
                    if explainer is not None:
                        try:
                            # For binary classification, expected_value might be a list [class0, class1]
                            val = explainer.expected_value
                            logger.info(f"SHAP expected_value type: {type(val)}, value: {val}")
                            
                            if isinstance(val, (list, np.ndarray)):
                                # If multiple outputs, take the second one (positive class)
                                if len(val) > 1:
                                    pred_data['expected_value'] = float(val[1])
                                else:
                                    pred_data['expected_value'] = float(val[0])
                            else:
                                pred_data['expected_value'] = float(val)
                        except Exception as e:
                            logger.error(f"Error extracting expected_value: {e}")
                            pred_data['expected_value'] = 0.0

                    # Get top 10 features by absolute SHAP value
                    abs_shap = np.abs(np.where(valid_mask, shap_row, 0))
                    top_indices = np.argsort(abs_shap)[-10:][::-1]
                    pred_data['top_features'] = [
                        {'feature': feature_names[j], 'shap_value': float(shap_row[j])}
                        for j in top_indices if valid_mask[j]
                    ]

                predictions_data.append(pred_data)

            # Sanitize predictions data for JSON storage
            predictions_data_safe = [sanitize_for_json(pred) for pred in predictions_data]
            crud.create_predictions_bulk(db, batch.id, predictions_data_safe)
            logger.info(f"TIMING: Result storage and post-processing took {time.time() - t_start:.2f}s")
        except Exception as e:
            logger.error(f"Error in result storage: {e}", exc_info=True)
            # Non-critical if we already have predictions in memory to return
            # but we need them in DB for history

        # Step 9: Calculate risk counts and complete batch
        risk_counts = {
            'LOW': sum(1 for p in predictions_data if p['risk_level'] == 'LOW'),
            'MEDIUM': sum(1 for p in predictions_data if p['risk_level'] == 'MEDIUM'),
            'HIGH': sum(1 for p in predictions_data if p['risk_level'] == 'HIGH'),
            'CRITICAL': sum(1 for p in predictions_data if p['risk_level'] == 'CRITICAL')
        }
        avg_prob = float(np.mean(probabilities))

        crud.complete_batch(
            db=db,
            batch_id=batch.id,
            avg_probability=avg_prob,
            risk_counts=risk_counts
        )

        # Convert to response format
        prediction_results = []
        for p in predictions_data:
            prediction_results.append(BatchPredictionResult(
                sk_id_curr=p['sk_id_curr'],
                prediction=p['prediction'],
                probability=p['probability'],
                risk_level=p['risk_level']
            ))

        # Log batch prediction for monitoring
        total_time_ms = (time.time() - start_time) * 1000
        log_batch_prediction(
            logger=production_logger,
            predictions=predictions_data,
            total_time_ms=total_time_ms,
            num_applications=len(dataframes['application.csv']),
            metadata={
                "batch_id": batch.id,
                "model_version": "Production",
                "file_sizes": {k: len(v) for k, v in dataframes.items()}
            }
        )

        return BatchPredictionResponse(
            success=True,
            timestamp=datetime.now().isoformat(),
            n_applications=len(dataframes['application.csv']),
            n_predictions=len(prediction_results),
            file_summaries=summaries,
            predictions=prediction_results,
            model_version="Production",
            batch_id=batch.id
        )

    except HTTPException:
        raise
    except Exception as e:
        # Log error
        log_error(
            logger=production_logger,
            error_type=type(e).__name__,
            error_message=str(e),
            endpoint="/batch/predict",
            metadata={"batch_id": batch.id if batch else None}
        )

        if batch:
            crud.fail_batch(db, batch.id, str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )





# ============================================================================
# Batch History & Retrieval Endpoints
# ============================================================================

@router.get("/history")
async def get_batch_history(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get recent batch prediction history with thread offloading."""
    def _fetch_history():
        batches = crud.get_recent_batches(db, skip=skip, limit=limit)
        return {
            "success": True,
            "count": len(batches),
            "batches": [
                {
                    "id": b.id,
                    "batch_name": b.batch_name,
                    "status": b.status.value,
                    "total_applications": b.total_applications,
                    "processed_applications": b.processed_applications,
                    "avg_probability": b.avg_probability,
                    "risk_distribution": {
                        "LOW": b.risk_low_count,
                        "MEDIUM": b.risk_medium_count,
                        "HIGH": b.risk_high_count,
                        "CRITICAL": b.risk_critical_count
                    },
                    "processing_time_seconds": b.processing_time_seconds,
                    "created_at": b.created_at.isoformat() if b.created_at else None,
                    "completed_at": b.completed_at.isoformat() if b.completed_at else None
                }
                for b in batches
            ]
        }

    import anyio
    return await anyio.to_thread.run_sync(_fetch_history)


@router.get("/history/{batch_id}/download")
async def download_batch_results(
    batch_id: int,
    format: str = "json",  # json or csv
    db: Session = Depends(get_db)
):
    """Download batch predictions with SHAP values with thread offloading."""
    def _prepare_download():
        batch = crud.get_batch(db, batch_id)
        if not batch:
            return {"error_status": 404, "error_detail": f"Batch {batch_id} not found"}

        predictions = crud.get_batch_predictions(db, batch_id)
        if not predictions:
            return {"error_status": 404, "error_detail": f"No predictions found for batch {batch_id}"}

        if format == "csv":
            df = pd.DataFrame([
                {
                    "SK_ID_CURR": p.sk_id_curr,
                    "PREDICTION": p.prediction,
                    "PROBABILITY": p.probability,
                    "RISK_LEVEL": p.risk_level.value
                }
                for p in predictions
            ])
            return {"csv_df": df}

        predictions_data = [
            {
                "SK_ID_CURR": p.sk_id_curr,
                "prediction": p.prediction,
                "probability": p.probability,
                "risk_level": p.risk_level.value,
                "shap_values": p.shap_values if p.shap_values else {},
                "top_features": p.top_features if p.top_features else []
            }
            for p in predictions
        ]
        return {
            "success": True,
            "batch_id": batch_id,
            "batch_name": batch.batch_name,
            "predictions": predictions_data
        }

    import anyio
    result = await anyio.to_thread.run_sync(_prepare_download)
    
    if "error_status" in result:
        raise HTTPException(status_code=result["error_status"], detail=result["error_detail"])
        
    if format == "csv" and "csv_df" in result:
        csv_stream = dataframe_to_csv_stream(result["csv_df"])
        return StreamingResponse(
            csv_stream,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=batch_{batch_id}_predictions.csv"}
        )

    return result


# Global cache for statistics
_STATS_CACHE = {
    'data': None,
    'last_updated': 0
}
CACHE_TTL_SEC = 60  # Cache stats for 1 minute

@router.get("/statistics")
async def get_batch_statistics(db: Session = Depends(get_db)):
    """Get overall batch prediction statistics with caching."""
    now = time.time()
    if _STATS_CACHE['data'] and (now - _STATS_CACHE['last_updated'] < CACHE_TTL_SEC):
        return _STATS_CACHE['data']

    # Offload heavy DB queries to thread pool
    def _fetch_stats():
        stats = crud.get_batch_statistics(db)
        avg_time = crud.get_average_processing_time(db)
        daily_counts = crud.get_daily_prediction_counts(db, days=30)
        return {
            "success": True,
            "statistics": {
                "total_batches": stats['total_batches'],
                "completed_batches": stats['completed_batches'],
                "total_predictions": stats['total_predictions'],
                "risk_distribution": stats['risk_distribution'],
                "average_processing_time_seconds": avg_time
            },
            "daily_predictions": daily_counts,
            "cached_at": datetime.now().isoformat()
        }

    import anyio
    data = await anyio.to_thread.run_sync(_fetch_stats)
    
    _STATS_CACHE['data'] = data
    _STATS_CACHE['last_updated'] = now
    
    return data
