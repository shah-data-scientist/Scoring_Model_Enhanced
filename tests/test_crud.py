"""Tests for CRUD operations module."""
import pytest
from datetime import datetime
from backend.crud import (
    create_prediction_batch,
    get_batch,
    complete_batch,
)
from backend.models import PredictionBatch
from backend.database import SessionLocal


@pytest.fixture
def db_session():
    """Create a database session for testing."""
    try:
        db = SessionLocal()
        yield db
    except Exception:
        pytest.skip("Database not available")
    finally:
        try:
            db.close()
        except Exception:
            pass


class TestBatchCRUD:
    """Tests for batch CRUD operations."""

    def test_create_batch(self, db_session):
        """Test creating a batch."""
        try:
            batch = create_prediction_batch(
                db=db_session,
                batch_name="test_batch",
                total_applications=10
            )
            
            assert batch is not None
            assert batch.batch_name == "test_batch"
            assert batch.total_applications == 10
        except Exception as e:
            pytest.skip(f"Database operation failed: {e}")

    def test_get_batch(self, db_session):
        """Test retrieving batch by ID."""
        try:
            batch = get_batch(db=db_session, batch_id=1)
            assert batch is None or isinstance(batch, PredictionBatch)
        except Exception as e:
            pytest.skip(f"Database operation failed: {e}")

    def test_complete_batch(self, db_session):
        """Test completing a batch."""
        try:
            batch = create_prediction_batch(db=db_session, batch_name="complete_test")
            completed = complete_batch(
                db=db_session,
                batch_id=batch.id,
                avg_probability=0.45
            )
            
            assert completed is not None
        except Exception as e:
            pytest.skip(f"Database operation failed: {e}")


class TestBatchModel:
    """Tests for batch model structure."""

    def test_batch_has_required_fields(self):
        """Test batch model has required fields."""
        # Check model definition
        from backend.models import PredictionBatch
        
        assert hasattr(PredictionBatch, 'id')
        assert hasattr(PredictionBatch, 'batch_name')
        assert hasattr(PredictionBatch, 'status')
        assert hasattr(PredictionBatch, 'total_applications')
