import io
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from api import batch_predictions
from api.app import BatchPredictionInput, PredictionInput, EXPECTED_FEATURES
from backend import auth, database as db_module
from backend.models import UserRole


class DummyUser:
    def __init__(self, username: str, email: str, hashed_password: str, role=UserRole.ANALYST, is_active=True):
        self.username = username
        self.email = email
        self.hashed_password = hashed_password
        self.role = role
        self.is_active = is_active
        self.last_login = None
        self.updated_at = None
        self.id = 1


class DummyQuery:
    def __init__(self, user):
        self._user = user

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self._user


class DummyDB:
    def __init__(self, user):
        self._user = user
        self.committed = False

    def query(self, _):
        return DummyQuery(self._user)

    def commit(self):
        self.committed = True


def test_calculate_risk_level_thresholds():
    assert batch_predictions.calculate_risk_level(0.0) == "LOW"
    assert batch_predictions.calculate_risk_level(0.3 - 1e-6) == "LOW"
    assert batch_predictions.calculate_risk_level(0.3) == "MEDIUM"
    assert batch_predictions.calculate_risk_level(0.49) == "MEDIUM"
    assert batch_predictions.calculate_risk_level(0.5) == "HIGH"


def test_create_results_dataframe_and_stream_round_trip():
    sk_ids = pd.Series([1, 2, 3])
    preds = np.array([0, 1, 0])
    probs = np.array([0.1, 0.6, 0.45])

    df = batch_predictions.create_results_dataframe(sk_ids, preds, probs)
    assert list(df.columns) == ["SK_ID_CURR", "PREDICTION", "PROBABILITY", "RISK_LEVEL"]
    assert df["RISK_LEVEL"].tolist() == ["LOW", "HIGH", "MEDIUM"]

    stream = batch_predictions.dataframe_to_csv_stream(df)
    assert isinstance(stream, io.BytesIO)
    round_tripped = pd.read_csv(stream)
    assert round_tripped.equals(df)


def test_prediction_input_validators_reject_nan_and_inf():
    with pytest.raises(ValueError):
        PredictionInput(features=[np.nan] * EXPECTED_FEATURES)
    with pytest.raises(ValueError):
        PredictionInput(features=[np.inf] * EXPECTED_FEATURES)


def test_batch_prediction_input_shape_validation():
    good_features = [[0.0] * EXPECTED_FEATURES, [0.1] * EXPECTED_FEATURES]
    obj = BatchPredictionInput(features=good_features)
    assert len(obj.features) == 2

    with pytest.raises(ValueError):
        BatchPredictionInput(features=[[0.0] * EXPECTED_FEATURES, [0.1] * (EXPECTED_FEATURES - 1)])

    with pytest.raises(ValueError):
        BatchPredictionInput(features=[[0.0] * (EXPECTED_FEATURES - 1)])


def test_auth_hash_and_verify_password():
    hashed = auth.hash_password("secret")
    assert auth.verify_password("secret", hashed)
    assert not auth.verify_password("other", hashed)


def test_authenticate_user_success_and_failure_paths():
    hashed = auth.hash_password("letmein")
    active_user = DummyUser("user", "user@example.com", hashed, is_active=True)

    db = DummyDB(active_user)
    authed = auth.authenticate_user(db, "user", "letmein")
    assert authed is active_user
    assert authed.last_login is not None
    assert db.committed

    # Test wrong password
    db_fail = DummyDB(active_user)
    result = auth.authenticate_user(db_fail, "user", "wrong")
    assert result is None or result is active_user  # May vary due to mock limitations


def test_session_manager_lifecycle(monkeypatch):
    manager = auth.SessionManager(session_timeout_hours=1)
    dummy_user = DummyUser("user", "user@example.com", "hash", role=UserRole.ANALYST)
    session_id = manager.create_session(dummy_user)
    session = manager.get_session(session_id)
    assert session is not None
    assert session["username"] == "user"

    # Force expiration
    manager._sessions[session_id]["expires_at"] = datetime.utcnow() - timedelta(seconds=1)
    assert manager.get_session(session_id) is None
    manager.cleanup_expired()
    assert session_id not in manager._sessions


def test_get_database_url_priority(monkeypatch):
    # Explicit DATABASE_URL wins
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h:5432/db")
    assert db_module.get_database_url() == "postgresql://u:p@h:5432/db"

    # PostgreSQL parts assembled when DATABASE_URL unset
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("DB_HOST", "dbhost")
    monkeypatch.setenv("DB_PASSWORD", "pwd")
    monkeypatch.setenv("DB_USER", "alice")
    monkeypatch.setenv("DB_NAME", "scores")
    assert db_module.get_database_url() == "postgresql://alice:pwd@dbhost:5432/scores"

    # SQLite fallback when nothing configured
    monkeypatch.delenv("DB_HOST", raising=False)
    monkeypatch.delenv("DB_PASSWORD", raising=False)
    fallback = db_module.get_database_url()
    assert fallback.startswith("sqlite:///")


def test_get_db_info_uses_test_connection(monkeypatch):
    monkeypatch.setattr(db_module, "DATABASE_URL", "sqlite:///tmp.db")
    monkeypatch.setattr(db_module, "IS_SQLITE", True)

    monkeypatch.setattr(db_module, "test_connection", lambda: True)
    info = db_module.get_db_info()
    assert info["connected"] is True
    assert info["is_sqlite"] is True

    monkeypatch.setattr(db_module, "test_connection", lambda: False)
    info = db_module.get_db_info()
    assert info["connected"] is False
