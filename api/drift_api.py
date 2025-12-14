"""API Endpoints for Data Drift Detection and Quality Monitoring"""

from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from api.drift_detection import (
    check_missing_values,
    check_out_of_range,
    detect_feature_drift,
    get_drift_history,
    save_drift_results,
    validate_schema,
)
from backend.crud import get_batch
from backend.database import get_db
from sqlalchemy.orm import Session

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class DriftDetectionRequest(BaseModel):
    """Request for drift detection."""
    feature_name: str
    feature_type: str = "numeric"  # numeric or categorical
    reference_data: List[float]
    current_data: List[float]
    alert_threshold: float = 0.05


class DataQualityCheckRequest(BaseModel):
    """Request for data quality check."""
    dataframe_dict: dict  # DataFrame as dict (columns: {col: [values]})
    expected_columns: Optional[List[str]] = None
    check_missing: bool = True
    check_range: bool = True
    check_schema: bool = True


class DriftDetectionResponse(BaseModel):
    """Response from drift detection."""
    feature_name: str
    feature_type: str
    drift_test: str
    is_drifted: bool
    interpretation: str
    statistics: dict


class DataQualityResponse(BaseModel):
    """Response from data quality check."""
    valid: bool
    missing_values: Optional[dict] = None
    out_of_range: Optional[dict] = None
    schema_validation: Optional[dict] = None
    summary: str


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.post("/drift", response_model=DriftDetectionResponse)
async def detect_drift(
    request: DriftDetectionRequest,
    db: Session = Depends(get_db)
):
    """
    Detect drift in a single feature.
    
    Performs statistical tests (KS for numeric, Chi-square for categorical)
    and calculates Population Stability Index (PSI).
    """
    try:
        reference = np.array(request.reference_data, dtype=float)
        current = np.array(request.current_data, dtype=float)
        
        if len(reference) < 10 or len(current) < 10:
            raise ValueError("Need at least 10 samples for drift detection")
        
        # Perform drift detection
        results = detect_feature_drift(
            feature_name=request.feature_name,
            reference_data=reference,
            current_data=current,
            feature_type=request.feature_type,
            alert_threshold=request.alert_threshold
        )
        
        # Save to database
        save_drift_results(db, request.feature_name, results)
        
        # Build response
        statistics = {
            key: value for key, value in results.items()
            if key not in ['feature_name', 'feature_type', 'drift_test', 'is_drifted', 'interpretation']
        }
        
        return DriftDetectionResponse(
            feature_name=results['feature_name'],
            feature_type=results['feature_type'],
            drift_test=results.get('drift_test', 'Unknown'),
            is_drifted=results['is_drifted'],
            interpretation=results['interpretation'],
            statistics=statistics
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/drift/batch/{batch_id}")
async def detect_drift_batch(
    batch_id: int,
    reference_batch_id: Optional[int] = Query(None, description="Batch ID to use as reference"),
    db: Session = Depends(get_db)
):
    """
    Detect drift for all features in a batch.
    
    Compares current batch against reference batch or training data.
    """
    try:
        # Get current batch
        current_batch = get_batch(db, batch_id)
        if not current_batch:
            raise HTTPException(status_code=404, detail="Batch not found")
        
        if not current_batch.raw_applications:
            raise HTTPException(status_code=400, detail="No raw data in batch")
        
        # Extract current data
        current_data = []
        for raw_app in current_batch.raw_applications:
            if raw_app.raw_data:
                current_data.append(raw_app.raw_data)
        
        if not current_data:
            raise HTTPException(status_code=400, detail="No valid data in batch")
        
        current_df = pd.DataFrame(current_data)
        
        # Get reference batch
        if reference_batch_id:
            ref_batch = get_batch(db, reference_batch_id)
            if not ref_batch:
                raise HTTPException(status_code=404, detail="Reference batch not found")
            
            ref_data = []
            for raw_app in ref_batch.raw_applications:
                if raw_app.raw_data:
                    ref_data.append(raw_app.raw_data)
            
            reference_df = pd.DataFrame(ref_data)
        else:
            # Use training data statistics (from config or hardcoded)
            # This would typically come from the trained model
            return {
                "error": "Reference batch required",
                "hint": "Provide reference_batch_id for comparison"
            }
        
        # Detect drift for numeric features
        drift_results = {}
        numeric_cols = current_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in reference_df.columns:
                continue
            
            ref_vals = reference_df[col].dropna().values
            curr_vals = current_df[col].dropna().values
            
            if len(ref_vals) > 10 and len(curr_vals) > 10:
                results = detect_feature_drift(
                    feature_name=col,
                    reference_data=ref_vals,
                    current_data=curr_vals,
                    feature_type='numeric'
                )
                save_drift_results(db, col, results, batch_id=batch_id)
                drift_results[col] = results
        
        return {
            "batch_id": batch_id,
            "features_checked": len(drift_results),
            "features_drifted": sum(1 for r in drift_results.values() if r['is_drifted']),
            "results": drift_results
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/quality", response_model=DataQualityResponse)
async def check_data_quality(request: DataQualityCheckRequest):
    """
    Check data quality for missing values, out-of-range, and schema validation.
    """
    try:
        # Convert dict to DataFrame
        df = pd.DataFrame(request.dataframe_dict)
        
        results = {
            'valid': True,
            'missing_values': None,
            'out_of_range': None,
            'schema_validation': None,
            'summary': ""
        }
        
        issues = []
        
        # Check missing values
        if request.check_missing:
            missing = check_missing_values(df)
            results['missing_values'] = missing
            high_missing = {k: v for k, v in missing.items() if v > 20}
            if high_missing:
                issues.append(f"{len(high_missing)} features with high missing values (>20%)")
                results['valid'] = False
        
        # Check out-of-range values
        if request.check_range:
            out_of_range = check_out_of_range(df)
            if out_of_range:
                results['out_of_range'] = out_of_range
                warnings = {k: v for k, v in out_of_range.items() if v['status'] == 'WARNING'}
                if warnings:
                    issues.append(f"Out-of-range warnings: {list(warnings.keys())}")
        
        # Check schema
        if request.check_schema and request.expected_columns:
            schema = validate_schema(df, request.expected_columns)
            results['schema_validation'] = schema
            if not schema['valid']:
                issues.append(f"Schema mismatch: missing {len(schema['missing_columns'])} columns")
                results['valid'] = False
        
        # Build summary
        if issues:
            results['summary'] = "; ".join(issues)
        else:
            results['summary'] = "✅ All checks passed"
        
        return DataQualityResponse(**results)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/drift/history/{feature_name}")
async def get_drift_detection_history(
    feature_name: str,
    limit: int = Query(30, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """
    Get drift detection history for a feature.
    
    Returns the last N drift detection results for a feature.
    """
    try:
        history = get_drift_history(db, feature_name, limit=limit)
        return {
            "feature_name": feature_name,
            "records": history,
            "count": len(history)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/stats/summary")
async def get_data_stats_summary(db: Session = Depends(get_db)):
    """
    Get summary statistics about data quality and drift across all batches.
    """
    try:
        from backend.models import DataDrift, Prediction

        drifted_count = db.query(DataDrift).filter(DataDrift.is_drifted == True).count()
        total_records = db.query(DataDrift).count()
        total_predictions = db.query(Prediction).count()

        return {
            "data_drift": {
                "total_features_checked": total_records,
                "features_with_drift": drifted_count,
                "drift_percentage": round((drifted_count / max(total_records, 1)) * 100, 2)
            },
            "predictions": {
                "total": total_predictions
            }
        }
    except Exception:
        # Gracefully return zeros when DB is not available or models not initialized
        return {
            "data_drift": {
                "total_features_checked": 0,
                "features_with_drift": 0,
                "drift_percentage": 0.0
            },
            "predictions": {
                "total": 0
            }
        }
