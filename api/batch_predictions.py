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

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.file_validation import get_file_summaries, validate_all_files
from api.preprocessing_pipeline import PreprocessingPipeline
from api.utils.logging import setup_production_logger, log_batch_prediction, log_error
from backend import crud
from backend.database import get_db

# Create router
router = APIRouter(prefix="/batch", tags=["Batch Predictions"])

# Setup production logger
production_logger = setup_production_logger()
logger = logging.getLogger(__name__)

# Global preprocessing pipeline
preprocessing_pipeline = None


def get_preprocessing_pipeline():
    """Get or create preprocessing pipeline instance."""
    global preprocessing_pipeline
    if preprocessing_pipeline is None:
        # Disable precomputed cache to ensure consistent predictions
        preprocessing_pipeline = PreprocessingPipeline(use_precomputed=False)
    return preprocessing_pipeline


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
    db: Session = Depends(get_db),
    model = None  # Will be injected from main app
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
    
    logger.info(f"Received batch prediction request")
    logger.info(f"Files received: application={application.filename}, bureau={bureau.filename}, "
                f"bureau_balance={bureau_balance.filename}, previous={previous_application.filename}, "
                f"credit_card={credit_card_balance.filename}, installments={installments_payments.filename}, "
                f"pos_cash={pos_cash_balance.filename}")

    try:
        # Check if model is loaded (will be handled by dependency injection)
        # For now, load from global or parameter
        if model is None:
            # Load from pickle file (fast and reliable)
            import pickle
            from pathlib import Path
            model_path = Path(__file__).parent.parent / "models" / "production_model.pkl"
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Model loaded from {model_path}")

        # Step 1: Organize uploaded files
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
            dataframes = validate_all_files(uploaded_files)
            summaries = get_file_summaries(dataframes)
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
            app_records = dataframes['application.csv'].to_dict('records')
            crud.store_raw_applications_bulk(db, batch.id, app_records)
        except Exception as e:
            print(f"Warning: Failed to store raw data: {e}")
            # Continue anyway - predictions are more important

        # Step 5: Preprocess data
        try:
            pipeline = get_preprocessing_pipeline()
            features_df, sk_id_curr = pipeline.process(dataframes, keep_sk_id=False)

            # Ensure features are in correct format
            X = features_df.values
            feature_names = list(features_df.columns)

        except Exception as e:
            crud.fail_batch(db, batch.id, f"Preprocessing failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Preprocessing failed: {str(e)}"
            )

        # Step 6: Make predictions
        try:
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1]

        except Exception as e:
            crud.fail_batch(db, batch.id, f"Prediction failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {str(e)}"
            )

        # Step 6.5: Compute SHAP values (optional, for explainability)
        shap_values_list = None
        try:
            import shap
            # Use TreeExplainer for LightGBM/XGBoost
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

            # For binary classification, get values for positive class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Class 1 (default)

            shap_values_list = shap_values
            print(f"Computed SHAP values for {len(shap_values)} predictions")

        except Exception as e:
            print(f"Warning: Failed to compute SHAP values: {e}")
            shap_values_list = None

        # Step 7: Create results
        results_df = create_results_dataframe(sk_id_curr, predictions, probabilities)

        # Step 8: Store predictions in database with SHAP values
        predictions_data = []
        for i, (_, row) in enumerate(results_df.iterrows()):
            pred_data = {
                'sk_id_curr': int(row['SK_ID_CURR']),
                'prediction': int(row['PREDICTION']),
                'probability': float(row['PROBABILITY']),
                'risk_level': row['RISK_LEVEL']
            }

            # Add SHAP values if available
            if shap_values_list is not None:
                shap_dict = {}
                for j, feat_name in enumerate(feature_names):
                    shap_dict[feat_name] = float(shap_values_list[i, j])
                pred_data['shap_values'] = shap_dict

                # Get top 10 features by absolute SHAP value
                sorted_features = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
                pred_data['top_features'] = [{'feature': f, 'shap_value': v} for f, v in sorted_features]

            predictions_data.append(pred_data)

        crud.create_predictions_bulk(db, batch.id, predictions_data)

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


# UNUSED: Alternative download endpoint not used by Streamlit
# @router.post("/predict/download")
# async def predict_batch_download(
#     application: UploadFile = File(...),
#     bureau: UploadFile = File(...),
#     bureau_balance: UploadFile = File(...),
#     previous_application: UploadFile = File(...),
#     credit_card_balance: UploadFile = File(...),
#     installments_payments: UploadFile = File(...),
#     pos_cash_balance: UploadFile = File(...)
# ):
#     """Batch predictions with CSV download.
#
#     Same as /predict but returns CSV file for download.
#
#     Returns:
#         CSV file with predictions
#
#     """
#     # Reuse predict_batch logic
#     result = await predict_batch(
#         application=application,
#         bureau=bureau,
#         bureau_balance=bureau_balance,
#         previous_application=previous_application,
#         credit_card_balance=credit_card_balance,
#         installments_payments=installments_payments,
#         pos_cash_balance=pos_cash_balance
#     )
#
#     # Convert predictions to DataFrame
#     predictions_data = [p.dict() for p in result.predictions]
#     df = pd.DataFrame(predictions_data)
#
#     # Create CSV stream
#     csv_stream = dataframe_to_csv_stream(df)
#
#     # Return as downloadable file
#     return StreamingResponse(
#         csv_stream,
#         media_type="text/csv",
#         headers={
#             "Content-Disposition": f"attachment; filename=predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
#         }
#     )


# UNUSED: Batch info endpoint not used by Streamlit
# @router.get("/info")
# async def batch_info():
#     """Get information about batch prediction endpoint.
#
#     Returns:
#         Endpoint information and requirements
#
#     """
#     return {
#         "endpoint": "/batch/predict",
#         "method": "POST",
#         "description": "Batch credit scoring predictions from raw CSV files",
#         "required_files": [
#             "application.csv",
#             "bureau.csv",
#             "bureau_balance.csv",
#             "previous_application.csv",
#             "credit_card_balance.csv",
#             "installments_payments.csv",
#             "POS_CASH_balance.csv"
#         ],
#         "critical_columns": {
#             "application.csv": 46,
#             "threshold": "85%"
#         },
#         "output_format": {
#             "SK_ID_CURR": "int",
#             "PREDICTION": "int (0=no default, 1=default)",
#             "PROBABILITY": "float [0-1]",
#             "RISK_LEVEL": "str (LOW/MEDIUM/HIGH/CRITICAL)"
#         },
#         "risk_levels": {
#             "LOW": "probability < 0.2",
#             "MEDIUM": "0.2 <= probability < 0.4",
#             "HIGH": "0.4 <= probability < 0.6",
#             "CRITICAL": "probability >= 0.6"
#         }
#     }


# ============================================================================
# Batch History & Retrieval Endpoints
# ============================================================================

@router.get("/history")
async def get_batch_history(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get recent batch prediction history.

    Args:
        skip: Number of records to skip (pagination)
        limit: Maximum records to return

    Returns:
        List of recent batches with summary info

    """
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


# UNUSED: Single batch details endpoint not used by Streamlit
# @router.get("/history/{batch_id}")
# async def get_batch_details(
#     batch_id: int,
#     db: Session = Depends(get_db)
# ):
#     """Get detailed information about a specific batch.
#
#     Args:
#         batch_id: The batch ID
#
#     Returns:
#         Batch details with all predictions
#
#     """
#     batch = crud.get_batch(db, batch_id)
#
#     if not batch:
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND,
#             detail=f"Batch {batch_id} not found"
#         )
#
#     # Get predictions for this batch
#     predictions = crud.get_batch_predictions(db, batch_id)
#
#     return {
#         "success": True,
#         "batch": {
#             "id": batch.id,
#             "batch_name": batch.batch_name,
#             "status": batch.status.value,
#             "total_applications": batch.total_applications,
#             "processed_applications": batch.processed_applications,
#             "avg_probability": batch.avg_probability,
#             "risk_distribution": {
#                 "LOW": batch.risk_low_count,
#                 "MEDIUM": batch.risk_medium_count,
#                 "HIGH": batch.risk_high_count,
#                 "CRITICAL": batch.risk_critical_count
#             },
#             "processing_time_seconds": batch.processing_time_seconds,
#             "created_at": batch.created_at.isoformat() if batch.created_at else None,
#             "completed_at": batch.completed_at.isoformat() if batch.completed_at else None,
#             "error_message": batch.error_message
#         },
#         "predictions": [
#             {
#                 "sk_id_curr": p.sk_id_curr,
#                 "prediction": p.prediction,
#                 "probability": p.probability,
#                 "risk_level": p.risk_level.value
#             }
#             for p in predictions
#         ]
#     }


@router.get("/history/{batch_id}/download")
async def download_batch_results(
    batch_id: int,
    format: str = "json",  # json or csv
    db: Session = Depends(get_db)
):
    """Download batch predictions with SHAP values.

    Args:
        batch_id: The batch ID
        format: Output format - 'json' (default) or 'csv'

    Returns:
        JSON with predictions and SHAP values, or CSV file

    """
    batch = crud.get_batch(db, batch_id)

    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch {batch_id} not found"
        )

    # Get predictions
    predictions = crud.get_batch_predictions(db, batch_id)

    if not predictions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No predictions found for batch {batch_id}"
        )

    # Build response with SHAP values
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

    if format == "csv":
        # Create DataFrame for CSV export
        df = pd.DataFrame([
            {
                "SK_ID_CURR": p.sk_id_curr,
                "PREDICTION": p.prediction,
                "PROBABILITY": p.probability,
                "RISK_LEVEL": p.risk_level.value
            }
            for p in predictions
        ])

        # Create CSV stream
        csv_stream = dataframe_to_csv_stream(df)

        return StreamingResponse(
            csv_stream,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=batch_{batch_id}_predictions.csv"
            }
        )

    # Default: return JSON with SHAP values
    return {
        "success": True,
        "batch_id": batch_id,
        "batch_name": batch.batch_name,
        "predictions": predictions_data
    }


@router.get("/statistics")
async def get_batch_statistics(db: Session = Depends(get_db)):
    """Get overall batch prediction statistics.

    Returns:
        Summary statistics for all batches

    """
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
        "daily_predictions": daily_counts
    }
