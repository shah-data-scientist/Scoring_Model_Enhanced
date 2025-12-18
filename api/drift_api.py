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
    get_training_reference_data,
    save_drift_results,
    save_drift_results_bulk,
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


import anyio

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
    def _run_drift_analysis():
        try:
            # Get current batch
            current_batch = get_batch(db, batch_id)
            if not current_batch:
                return {"error": "Batch not found"}
            
            if not current_batch.raw_applications:
                return {"error": "No raw data in batch"}
            
            # Extract current data
            current_data = []
            for raw_app in current_batch.raw_applications:
                if raw_app.raw_data:
                    current_data.append(raw_app.raw_data)
            
            if not current_data:
                return {"error": "No valid data in batch"}
            
            current_df = pd.DataFrame(current_data)
            
            # Get reference batch
            if reference_batch_id:
                ref_batch = get_batch(db, reference_batch_id)
                if not ref_batch:
                    return {"error": "Reference batch not found"}
                
                ref_data = []
                for raw_app in ref_batch.raw_applications:
                    if raw_app.raw_data:
                        ref_data.append(raw_app.raw_data)
                
                reference_df = pd.DataFrame(ref_data)
            else:
                # Use training data plus validation data as reference
                reference_df = get_training_reference_data()
                if reference_df.empty:
                    return {"error": "Training reference data not found on server"}
            
            # Detect drift for numeric features - Optimize by only checking important features
            drift_results = {}
            numeric_cols = current_df.select_dtypes(include=[np.number]).columns
            
            # Limit to top 50 features if many exist to prevent timeout
            MAX_DRIFT_FEATURES = 50
            if len(numeric_cols) > MAX_DRIFT_FEATURES:
                try:
                    importance_path = PROJECT_ROOT / "config" / "model_feature_importance.csv"
                    if importance_path.exists():
                        imp_df = pd.read_csv(importance_path)
                        top_features = imp_df.head(MAX_DRIFT_FEATURES)['feature'].tolist()
                        numeric_cols = [c for c in numeric_cols if c in top_features]
                    else:
                        numeric_cols = numeric_cols[:MAX_DRIFT_FEATURES]
                except:
                    numeric_cols = numeric_cols[:MAX_DRIFT_FEATURES]

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
                    drift_results[col] = results
            
            # Save all results in bulk (much faster than individual commits)
            if drift_results:
                save_drift_results_bulk(db, drift_results, batch_id=batch_id)
            
            return {
                "batch_id": batch_id,
                "features_checked": len(drift_results),
                "features_drifted": sum(1 for r in drift_results.values() if r['is_drifted']),
                "results": drift_results
            }
        except Exception as e:
            return {"error": str(e)}

    # Offload CPU intensive work to a thread pool
    results = await anyio.to_thread.run_sync(_run_drift_analysis)
    
    if "error" in results:
        raise HTTPException(status_code=400, detail=results["error"])
        
    return results


@router.post("/quality", response_model=DataQualityResponse)
async def check_data_quality(request: DataQualityCheckRequest):
    """
    Check data quality for missing values, out-of-range, and schema validation.
    """
    def _run_quality_checks():
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
            
            return results
        except Exception as e:
            return {"error": str(e)}

    results = await anyio.to_thread.run_sync(_run_quality_checks)
    
    if "error" in results:
        raise HTTPException(status_code=400, detail=results["error"])
        
    return DataQualityResponse(**results)


@router.get("/drift/history/{feature_name}")
async def get_drift_detection_history(
    feature_name: str,
    limit: int = Query(30, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """
    Get drift detection history for a feature with thread offloading.
    """
    def _fetch_history():
        return get_drift_history(db, feature_name, limit=limit)

    try:
        history = await anyio.to_thread.run_sync(_fetch_history)
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
    Get summary statistics about data quality and drift with thread offloading.
    """
    def _fetch_summary():
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
        except:
            return {
                "data_drift": {"total_features_checked": 0, "features_with_drift": 0, "drift_percentage": 0.0},
                "predictions": {"total": 0}
            }

    return await anyio.to_thread.run_sync(_fetch_summary)
