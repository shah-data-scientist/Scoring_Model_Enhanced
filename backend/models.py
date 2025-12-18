"""Database Models for Credit Scoring API
======================================
SQLAlchemy ORM models for:
- Users (authentication)
- Prediction batches and results
- Raw data storage
- Model monitoring
"""

import enum
from datetime import datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


# =============================================================================
# ENUMS
# =============================================================================

class UserRole(str, enum.Enum):
    """User roles for access control."""

    ANALYST = "ANALYST"
    ADMIN = "ADMIN"


class RiskLevel(str, enum.Enum):
    """Credit risk levels."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class BatchStatus(str, enum.Enum):
    """Batch prediction status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# USER & AUTHENTICATION
# =============================================================================

class User(Base):
    """User model for authentication and access control."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(SQLEnum(UserRole, native_enum=False), default=UserRole.ANALYST, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    # Relationships
    prediction_batches = relationship("PredictionBatch", back_populates="user")

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', role={self.role.value})>"


# =============================================================================
# PREDICTION BATCHES & RESULTS
# =============================================================================

class PredictionBatch(Base):
    """Batch prediction job tracking."""

    __tablename__ = "prediction_batches"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # nullable for anonymous
    batch_name = Column(String(200), nullable=True)
    status = Column(SQLEnum(BatchStatus, native_enum=False), default=BatchStatus.PENDING, nullable=False)

    # File info
    total_applications = Column(Integer, nullable=True)
    processed_applications = Column(Integer, default=0)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Processing info
    processing_time_seconds = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)

    # Statistics (summary of predictions)
    avg_probability = Column(Float, nullable=True)
    risk_low_count = Column(Integer, default=0)
    risk_medium_count = Column(Integer, default=0)
    risk_high_count = Column(Integer, default=0)
    risk_critical_count = Column(Integer, default=0)

    # Relationships
    user = relationship("User", back_populates="prediction_batches")
    predictions = relationship("Prediction", back_populates="batch", cascade="all, delete-orphan")
    raw_applications = relationship("RawApplication", back_populates="batch", cascade="all, delete-orphan")

    __table_args__ = (
        Index('ix_prediction_batches_user_created', 'user_id', 'created_at'),
    )

    def __repr__(self):
        return f"<PredictionBatch(id={self.id}, status={self.status.value}, apps={self.total_applications})>"


class Prediction(Base):
    """Individual prediction results."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    batch_id = Column(Integer, ForeignKey("prediction_batches.id"), nullable=False, index=True)
    sk_id_curr = Column(BigInteger, nullable=False, index=True)

    # Prediction results
    prediction = Column(Integer, nullable=False)  # 0 or 1
    probability = Column(Float, nullable=False)
    risk_level = Column(SQLEnum(RiskLevel), nullable=False, index=True)

    # SHAP values (optional, stored as JSON)
    shap_values = Column(JSON, nullable=True)
    top_features = Column(JSON, nullable=True)  # Top contributing features

    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    batch = relationship("PredictionBatch", back_populates="predictions")

    __table_args__ = (
        Index('ix_predictions_batch_sk_id', 'batch_id', 'sk_id_curr'),
        Index('ix_predictions_risk_created', 'risk_level', 'created_at'),
        UniqueConstraint('batch_id', 'sk_id_curr', name='uq_batch_application'),
    )

    def __repr__(self):
        return f"<Prediction(sk_id={self.sk_id_curr}, prob={self.probability:.4f}, risk={self.risk_level.value})>"


# =============================================================================
# RAW DATA STORAGE
# =============================================================================

class RawApplication(Base):
    """Store raw application data from uploaded CSV files.
    
    Stores all columns from application.csv as JSON for flexibility.
    Critical columns are also stored as separate fields for querying.
    """

    __tablename__ = "raw_applications"

    id = Column(Integer, primary_key=True, autoincrement=True)
    batch_id = Column(Integer, ForeignKey("prediction_batches.id"), nullable=False, index=True)
    sk_id_curr = Column(BigInteger, nullable=False, index=True)

    # Critical fields (for fast querying)
    amt_credit = Column(Float, nullable=True)
    amt_annuity = Column(Float, nullable=True)
    amt_income_total = Column(Float, nullable=True)
    amt_goods_price = Column(Float, nullable=True)
    ext_source_1 = Column(Float, nullable=True)
    ext_source_2 = Column(Float, nullable=True)
    ext_source_3 = Column(Float, nullable=True)
    days_birth = Column(Integer, nullable=True)
    days_employed = Column(Integer, nullable=True)
    code_gender = Column(String(10), nullable=True)

    # Full raw data as JSON (all columns from application.csv)
    raw_data = Column(JSON, nullable=False)

    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    batch = relationship("PredictionBatch", back_populates="raw_applications")

    __table_args__ = (
        Index('ix_raw_applications_batch_sk_id', 'batch_id', 'sk_id_curr'),
        UniqueConstraint('batch_id', 'sk_id_curr', name='uq_batch_raw_application'),
    )

    def __repr__(self):
        return f"<RawApplication(sk_id={self.sk_id_curr}, credit={self.amt_credit})>"


# =============================================================================
# MODEL MONITORING (for Phase 4+)
# =============================================================================

class ModelMetrics(Base):
    """Track model performance metrics over time."""

    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Model identification
    model_name = Column(String(100), nullable=False, index=True)
    model_version = Column(String(50), nullable=True)

    # Metrics
    metric_name = Column(String(50), nullable=False)  # e.g., 'auc', 'accuracy', 'f1'
    metric_value = Column(Float, nullable=False)

    # Context
    dataset_name = Column(String(100), nullable=True)  # e.g., 'validation', 'production'
    n_samples = Column(Integer, nullable=True)

    # Timestamp
    recorded_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Additional metadata (renamed from 'metadata' which is reserved)
    extra_info = Column(JSON, nullable=True)

    __table_args__ = (
        Index('ix_model_metrics_model_date', 'model_name', 'recorded_at'),
    )

    def __repr__(self):
        return f"<ModelMetrics(model={self.model_name}, {self.metric_name}={self.metric_value:.4f})>"


class DataDrift(Base):
    """Track data drift metrics."""

    __tablename__ = "data_drift"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Feature info
    feature_name = Column(String(100), nullable=False, index=True)

    # Drift metrics
    drift_score = Column(Float, nullable=False)  # e.g., PSI, KS statistic
    drift_type = Column(String(50), nullable=False)  # e.g., 'PSI', 'KS', 'chi2'
    is_drifted = Column(Boolean, default=False)

    # Reference vs current statistics
    reference_mean = Column(Float, nullable=True)
    current_mean = Column(Float, nullable=True)
    reference_std = Column(Float, nullable=True)
    current_std = Column(Float, nullable=True)

    # Context
    batch_id = Column(Integer, ForeignKey("prediction_batches.id"), nullable=True)
    n_samples = Column(Integer, nullable=True)

    # Timestamp
    recorded_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    __table_args__ = (
        Index('ix_data_drift_feature_date', 'feature_name', 'recorded_at'),
    )

    def __repr__(self):
        return f"<DataDrift(feature={self.feature_name}, score={self.drift_score:.4f}, drifted={self.is_drifted})>"


class APIRequestLog(Base):
    """Log API requests for monitoring and debugging."""

    __tablename__ = "api_request_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Request info
    endpoint = Column(String(200), nullable=False, index=True)
    method = Column(String(10), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    # Request details
    request_size_bytes = Column(Integer, nullable=True)
    response_status = Column(Integer, nullable=False)
    response_time_ms = Column(Float, nullable=True)

    # Error info (if any)
    error_message = Column(Text, nullable=True)

    # Timestamp
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Client info
    client_ip = Column(String(50), nullable=True)
    user_agent = Column(String(500), nullable=True)

    __table_args__ = (
        Index('ix_api_logs_endpoint_time', 'endpoint', 'timestamp'),
    )

    def __repr__(self):
        return f"<APIRequestLog(endpoint={self.endpoint}, status={self.response_status})>"
