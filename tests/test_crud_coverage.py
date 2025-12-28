
import pytest
import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.models import Base, BatchStatus, RiskLevel
from backend import crud

# Create in-memory database for testing
@pytest.fixture(scope="function")
def db_session():
    """Create a fresh in-memory database session for each test."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()

class TestCrudPredictionBatch:
    def test_create_prediction_batch(self, db_session):
        batch = crud.create_prediction_batch(db_session, user_id=1, batch_name="Test Batch", total_applications=10)
        assert batch.id is not None
        assert batch.status == BatchStatus.PENDING
        assert batch.user_id == 1
        assert batch.total_applications == 10

    def test_start_batch_processing(self, db_session):
        batch = crud.create_prediction_batch(db_session, user_id=1)
        updated_batch = crud.start_batch_processing(db_session, batch.id)
        assert updated_batch.status == BatchStatus.PROCESSING
        assert updated_batch.started_at is not None

    def test_complete_batch(self, db_session):
        batch = crud.create_prediction_batch(db_session, user_id=1)
        crud.start_batch_processing(db_session, batch.id)
        
        updated_batch = crud.complete_batch(
            db_session, 
            batch.id, 
            avg_probability=0.25, 
            risk_counts={"LOW": 8, "HIGH": 2}
        )
        assert updated_batch.status == BatchStatus.COMPLETED
        assert updated_batch.completed_at is not None
        assert updated_batch.avg_probability == 0.25
        assert updated_batch.risk_low_count == 8
        assert updated_batch.risk_high_count == 2
        assert updated_batch.processing_time_seconds is not None

    def test_fail_batch(self, db_session):
        batch = crud.create_prediction_batch(db_session, user_id=1)
        crud.start_batch_processing(db_session, batch.id)
        
        failed_batch = crud.fail_batch(db_session, batch.id, "Something went wrong")
        assert failed_batch.status == BatchStatus.FAILED
        assert failed_batch.error_message == "Something went wrong"
        assert failed_batch.completed_at is not None

    def test_get_batch_not_found(self, db_session):
        assert crud.get_batch(db_session, 999) is None

    def test_get_user_batches(self, db_session):
        crud.create_prediction_batch(db_session, user_id=1, batch_name="Batch 1")
        crud.create_prediction_batch(db_session, user_id=1, batch_name="Batch 2")
        crud.create_prediction_batch(db_session, user_id=2, batch_name="Other User Batch")
        
        batches = crud.get_user_batches(db_session, user_id=1)
        assert len(batches) == 2
        assert batches[0].batch_name == "Batch 2"  # Ordered by created_at desc

    def test_get_recent_batches(self, db_session):
        crud.create_prediction_batch(db_session, user_id=1)
        crud.create_prediction_batch(db_session, user_id=2)
        batches = crud.get_recent_batches(db_session)
        assert len(batches) == 2

class TestCrudPredictions:
    def test_create_predictions_bulk(self, db_session):
        batch = crud.create_prediction_batch(db_session, user_id=1)
        data = [
            {
                "sk_id_curr": 1001, 
                "prediction": 0, 
                "probability": 0.1, 
                "risk_level": "LOW",
                "shap_values": {"f1": 0.1},
                "top_features": ["f1"]
            },
            {
                "sk_id_curr": 1002, 
                "prediction": 1, 
                "probability": 0.8, 
                "risk_level": "CRITICAL"
            }
        ]
        count = crud.create_predictions_bulk(db_session, batch.id, data)
        assert count == 2
        
        preds = crud.get_batch_predictions(db_session, batch.id)
        assert len(preds) == 2
        assert preds[0].sk_id_curr == 1001

    def test_get_prediction_by_sk_id(self, db_session):
        batch = crud.create_prediction_batch(db_session, user_id=1)
        crud.create_predictions_bulk(db_session, batch.id, [{
            "sk_id_curr": 12345, "prediction": 0, "probability": 0.1, "risk_level": "LOW"
        }])
        
        pred = crud.get_prediction_by_sk_id(db_session, batch.id, 12345)
        assert pred is not None
        assert pred.sk_id_curr == 12345
        
        assert crud.get_prediction_by_sk_id(db_session, batch.id, 99999) is None

    def test_get_predictions_by_risk_level(self, db_session):
        batch = crud.create_prediction_batch(db_session, user_id=1)
        crud.create_predictions_bulk(db_session, batch.id, [
            {"sk_id_curr": 1, "prediction": 0, "probability": 0.1, "risk_level": "LOW"},
            {"sk_id_curr": 2, "prediction": 1, "probability": 0.9, "risk_level": "CRITICAL"},
            {"sk_id_curr": 3, "prediction": 0, "probability": 0.2, "risk_level": "LOW"}
        ])
        
        low_risk = crud.get_predictions_by_risk_level(db_session, batch.id, "LOW")
        assert len(low_risk) == 2

class TestCrudRawApplications:
    def test_store_raw_applications_bulk(self, db_session):
        batch = crud.create_prediction_batch(db_session, user_id=1)
        data = [
            {
                "SK_ID_CURR": 1001,
                "AMT_CREDIT": 50000.0,
                "CODE_GENDER": "M",
                "EXT_SOURCE_1": 0.5,
                "MISSING_VAL": float('nan') # Test NaN handling
            }
        ]
        count = crud.store_raw_applications_bulk(db_session, batch.id, data)
        assert count == 1
        
        apps = crud.get_batch_raw_applications(db_session, batch.id)
        assert len(apps) == 1
        assert apps[0].sk_id_curr == 1001
        assert apps[0].raw_data['MISSING_VAL'] is None  # Check NaN -> None conversion

class TestCrudMonitoring:
    def test_log_model_metric(self, db_session):
        metric = crud.log_model_metric(
            db_session, 
            model_name="test_model", 
            metric_name="accuracy", 
            metric_value=0.95
        )
        assert metric.id is not None
        assert metric.metric_value == 0.95

    def test_log_data_drift(self, db_session):
        drift = crud.log_data_drift(
            db_session,
            feature_name="income",
            drift_score=0.1,
            drift_type="ks",
            is_drifted=True
        )
        assert drift.id is not None
        assert drift.is_drifted is True

    def test_log_api_request(self, db_session):
        log = crud.log_api_request(
            db_session,
            endpoint="/predict",
            method="POST",
            response_status=200
        )
        assert log.id is not None

class TestCrudStatistics:
    def test_get_batch_statistics(self, db_session):
        # Create some data
        b1 = crud.create_prediction_batch(db_session, user_id=1)
        crud.start_batch_processing(db_session, b1.id)
        crud.complete_batch(db_session, b1.id)
        
        crud.create_predictions_bulk(db_session, b1.id, [
            {"sk_id_curr": 1, "prediction": 0, "probability": 0.1, "risk_level": "LOW"},
            {"sk_id_curr": 2, "prediction": 1, "probability": 0.8, "risk_level": "HIGH"}
        ])
        
        stats = crud.get_batch_statistics(db_session)
        assert stats['total_batches'] == 1
        assert stats['completed_batches'] == 1
        assert stats['total_predictions'] == 2
        assert stats['risk_distribution']['LOW'] == 1
        assert stats['risk_distribution']['HIGH'] == 1

    def test_get_average_processing_time(self, db_session):
        # Empty DB
        assert crud.get_average_processing_time(db_session) == 0.0
        
        # Add a batch with time
        b1 = crud.create_prediction_batch(db_session, user_id=1)
        crud.start_batch_processing(db_session, b1.id)
        crud.complete_batch(db_session, b1.id)
        
        # Manually set time to ensure non-zero for test
        b1.processing_time_seconds = 10.0
        db_session.commit()
        
        assert crud.get_average_processing_time(db_session) == 10.0
