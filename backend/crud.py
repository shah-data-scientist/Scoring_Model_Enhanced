"""Database CRUD Operations
========================
Create, Read, Update, Delete operations for all models.
"""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from backend.models import (
    APIRequestLog,
    BatchStatus,
    DataDrift,
    ModelMetrics,
    Prediction,
    PredictionBatch,
    RawApplication,
    RiskLevel,
)

# =============================================================================
# PREDICTION BATCH OPERATIONS
# =============================================================================

def create_prediction_batch(
    db: Session,
    user_id: int | None = None,
    batch_name: str | None = None,
    total_applications: int = 0
) -> PredictionBatch:
    """Create a new prediction batch."""
    batch = PredictionBatch(
        user_id=user_id,
        batch_name=batch_name,
        total_applications=total_applications,
        status=BatchStatus.PENDING
    )
    db.add(batch)
    db.commit()
    db.refresh(batch)
    return batch


def start_batch_processing(db: Session, batch_id: int) -> PredictionBatch:
    """Mark batch as processing."""
    batch = db.query(PredictionBatch).filter(PredictionBatch.id == batch_id).first()
    if batch:
        batch.status = BatchStatus.PROCESSING
        batch.started_at = datetime.utcnow()
        db.commit()
    return batch


def complete_batch(
    db: Session,
    batch_id: int,
    avg_probability: float = None,
    risk_counts: dict[str, int] = None
) -> PredictionBatch:
    """Mark batch as completed with statistics."""
    batch = db.query(PredictionBatch).filter(PredictionBatch.id == batch_id).first()
    if batch:
        batch.status = BatchStatus.COMPLETED
        batch.completed_at = datetime.utcnow()

        if batch.started_at:
            batch.processing_time_seconds = (
                batch.completed_at - batch.started_at
            ).total_seconds()

        if avg_probability is not None:
            batch.avg_probability = avg_probability

        if risk_counts:
            batch.risk_low_count = risk_counts.get('LOW', 0)
            batch.risk_medium_count = risk_counts.get('MEDIUM', 0)
            batch.risk_high_count = risk_counts.get('HIGH', 0)
            batch.risk_critical_count = risk_counts.get('CRITICAL', 0)

        # Update processed count
        batch.processed_applications = db.query(Prediction).filter(
            Prediction.batch_id == batch_id
        ).count()

        db.commit()
    return batch


def fail_batch(db: Session, batch_id: int, error_message: str) -> PredictionBatch:
    """Mark batch as failed with error message."""
    batch = db.query(PredictionBatch).filter(PredictionBatch.id == batch_id).first()
    if batch:
        batch.status = BatchStatus.FAILED
        batch.completed_at = datetime.utcnow()
        batch.error_message = error_message

        if batch.started_at:
            batch.processing_time_seconds = (
                batch.completed_at - batch.started_at
            ).total_seconds()

        db.commit()
    return batch


def get_batch(db: Session, batch_id: int) -> PredictionBatch | None:
    """Get batch by ID."""
    return db.query(PredictionBatch).filter(PredictionBatch.id == batch_id).first()


def get_user_batches(
    db: Session,
    user_id: int,
    skip: int = 0,
    limit: int = 50
) -> list[PredictionBatch]:
    """Get batches for a specific user."""
    return db.query(PredictionBatch).filter(
        PredictionBatch.user_id == user_id
    ).order_by(
        desc(PredictionBatch.created_at)
    ).offset(skip).limit(limit).all()


def get_recent_batches(
    db: Session,
    skip: int = 0,
    limit: int = 50
) -> list[PredictionBatch]:
    """Get recent batches (all users)."""
    return db.query(PredictionBatch).order_by(
        desc(PredictionBatch.created_at)
    ).offset(skip).limit(limit).all()


# =============================================================================
# PREDICTION OPERATIONS
# =============================================================================

def create_predictions_bulk(
    db: Session,
    batch_id: int,
    predictions_data: list[dict[str, Any]]
) -> int:
    """Bulk insert predictions for a batch.
    
    Args:
        db: Database session
        batch_id: Batch ID
        predictions_data: List of dicts with keys:
            - sk_id_curr: Application ID
            - prediction: 0 or 1
            - probability: Float 0-1
            - risk_level: str (LOW/MEDIUM/HIGH/CRITICAL)
            - shap_values: Optional dict
            - top_features: Optional list
    
    Returns:
        Number of predictions inserted

    """
    predictions = []
    for data in predictions_data:
        pred = Prediction(
            batch_id=batch_id,
            sk_id_curr=data['sk_id_curr'],
            prediction=data['prediction'],
            probability=data['probability'],
            risk_level=RiskLevel(data['risk_level']),
            shap_values=data.get('shap_values'),
            top_features=data.get('top_features')
        )
        predictions.append(pred)

    db.bulk_save_objects(predictions)
    db.commit()

    return len(predictions)


def get_batch_predictions(
    db: Session,
    batch_id: int,
    skip: int = 0,
    limit: int = 1000
) -> list[Prediction]:
    """Get predictions for a batch."""
    return db.query(Prediction).filter(
        Prediction.batch_id == batch_id
    ).offset(skip).limit(limit).all()


def get_prediction_by_sk_id(
    db: Session,
    batch_id: int,
    sk_id_curr: int
) -> Prediction | None:
    """Get specific prediction by SK_ID_CURR."""
    return db.query(Prediction).filter(
        Prediction.batch_id == batch_id,
        Prediction.sk_id_curr == sk_id_curr
    ).first()


def get_predictions_by_risk_level(
    db: Session,
    batch_id: int,
    risk_level: str
) -> list[Prediction]:
    """Get predictions filtered by risk level."""
    return db.query(Prediction).filter(
        Prediction.batch_id == batch_id,
        Prediction.risk_level == RiskLevel(risk_level)
    ).all()


# =============================================================================
# RAW APPLICATION DATA OPERATIONS
# =============================================================================

def store_raw_applications_bulk(
    db: Session,
    batch_id: int,
    applications_data: list[dict[str, Any]]
) -> int:
    """Bulk store raw application data.
    
    Args:
        db: Database session
        batch_id: Batch ID
        applications_data: List of dicts with raw application data
    
    Returns:
        Number of applications stored

    """
    raw_apps = []
    for data in applications_data:
        raw_app = RawApplication(
            batch_id=batch_id,
            sk_id_curr=data.get('SK_ID_CURR'),
            amt_credit=data.get('AMT_CREDIT'),
            amt_annuity=data.get('AMT_ANNUITY'),
            amt_income_total=data.get('AMT_INCOME_TOTAL'),
            amt_goods_price=data.get('AMT_GOODS_PRICE'),
            ext_source_1=data.get('EXT_SOURCE_1'),
            ext_source_2=data.get('EXT_SOURCE_2'),
            ext_source_3=data.get('EXT_SOURCE_3'),
            days_birth=data.get('DAYS_BIRTH'),
            days_employed=data.get('DAYS_EMPLOYED'),
            code_gender=data.get('CODE_GENDER'),
            raw_data=data  # Store complete raw data as JSON
        )
        raw_apps.append(raw_app)

    db.bulk_save_objects(raw_apps)
    db.commit()

    return len(raw_apps)


def get_batch_raw_applications(
    db: Session,
    batch_id: int,
    skip: int = 0,
    limit: int = 1000
) -> list[RawApplication]:
    """Get raw applications for a batch."""
    return db.query(RawApplication).filter(
        RawApplication.batch_id == batch_id
    ).offset(skip).limit(limit).all()


# =============================================================================
# MONITORING OPERATIONS
# =============================================================================

def log_model_metric(
    db: Session,
    model_name: str,
    metric_name: str,
    metric_value: float,
    model_version: str = None,
    dataset_name: str = None,
    n_samples: int = None,
    extra_info: dict = None
) -> ModelMetrics:
    """Log a model metric."""
    metric = ModelMetrics(
        model_name=model_name,
        model_version=model_version,
        metric_name=metric_name,
        metric_value=metric_value,
        dataset_name=dataset_name,
        n_samples=n_samples,
        extra_info=extra_info
    )
    db.add(metric)
    db.commit()
    db.refresh(metric)
    return metric


def log_data_drift(
    db: Session,
    feature_name: str,
    drift_score: float,
    drift_type: str,
    is_drifted: bool,
    reference_mean: float = None,
    current_mean: float = None,
    reference_std: float = None,
    current_std: float = None,
    batch_id: int = None,
    n_samples: int = None
) -> DataDrift:
    """Log data drift detection."""
    drift = DataDrift(
        feature_name=feature_name,
        drift_score=drift_score,
        drift_type=drift_type,
        is_drifted=is_drifted,
        reference_mean=reference_mean,
        current_mean=current_mean,
        reference_std=reference_std,
        current_std=current_std,
        batch_id=batch_id,
        n_samples=n_samples
    )
    db.add(drift)
    db.commit()
    db.refresh(drift)
    return drift


def log_api_request(
    db: Session,
    endpoint: str,
    method: str,
    response_status: int,
    user_id: int = None,
    request_size_bytes: int = None,
    response_time_ms: float = None,
    error_message: str = None,
    client_ip: str = None,
    user_agent: str = None
) -> APIRequestLog:
    """Log an API request."""
    log = APIRequestLog(
        endpoint=endpoint,
        method=method,
        user_id=user_id,
        response_status=response_status,
        request_size_bytes=request_size_bytes,
        response_time_ms=response_time_ms,
        error_message=error_message,
        client_ip=client_ip,
        user_agent=user_agent
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


# =============================================================================
# STATISTICS & ANALYTICS
# =============================================================================

def get_batch_statistics(db: Session) -> dict[str, Any]:
    """Get overall batch statistics."""
    total_batches = db.query(func.count(PredictionBatch.id)).scalar()
    completed_batches = db.query(func.count(PredictionBatch.id)).filter(
        PredictionBatch.status == BatchStatus.COMPLETED
    ).scalar()
    total_predictions = db.query(func.count(Prediction.id)).scalar()

    # Risk distribution
    risk_dist = db.query(
        Prediction.risk_level,
        func.count(Prediction.id)
    ).group_by(Prediction.risk_level).all()

    return {
        'total_batches': total_batches,
        'completed_batches': completed_batches,
        'total_predictions': total_predictions,
        'risk_distribution': {r.value: c for r, c in risk_dist}
    }


def get_daily_prediction_counts(db: Session, days: int = 30) -> list[dict]:
    """Get daily prediction counts for the last N days."""
    from datetime import datetime, timedelta

    # Get predictions from the last N days
    cutoff_date = datetime.now(UTC) - timedelta(days=days)

    predictions = db.query(Prediction).filter(
        Prediction.created_at >= cutoff_date.replace(tzinfo=None)
    ).all()

    # Group by date in Python (SQLite compatible)
    date_counts: dict[str, int] = {}
    for pred in predictions:
        date_str = pred.created_at.strftime('%Y-%m-%d')
        date_counts[date_str] = date_counts.get(date_str, 0) + 1

    # Sort by date descending
    sorted_dates = sorted(date_counts.keys(), reverse=True)

    return [{'date': d, 'count': date_counts[d]} for d in sorted_dates[:days]]


def get_average_processing_time(db: Session) -> float:
    """Get average batch processing time in seconds."""
    result = db.query(
        func.avg(PredictionBatch.processing_time_seconds)
    ).filter(
        PredictionBatch.status == BatchStatus.COMPLETED,
        PredictionBatch.processing_time_seconds.isnot(None)
    ).scalar()

    return float(result) if result else 0.0
