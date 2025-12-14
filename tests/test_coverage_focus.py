"""Focused coverage tests for critical runtime modules."""

import io
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient

from api.app import app
from api import file_validation
from backend.database import get_db
from backend import auth


class TestFileValidation:
    def test_validate_file_presence_missing(self):
        is_valid, missing = file_validation.validate_file_presence({"application.csv": None})
        assert is_valid is False
        # Should report at least one missing required file
        assert len(missing) >= 1

    def test_validate_application_columns_empty_df(self):
        df = pd.DataFrame()
        is_valid, missing, coverage = file_validation.validate_application_columns(df)
        assert is_valid is False
        assert coverage == 0.0
        assert len(missing) == len(file_validation.CRITICAL_APPLICATION_COLUMNS)

    def test_validate_csv_structure_success_and_reset(self):
        csv_bytes = b"SK_ID_CURR,col1\n1,2\n"
        upload = UploadFile(filename="application.csv", file=io.BytesIO(csv_bytes))
        df = file_validation.validate_csv_structure(upload, "application.csv")
        assert not df.empty
        # File pointer should be reset for potential re-read
        assert upload.file.tell() == 0

    def test_validate_csv_structure_missing_sk_raises(self):
        csv_bytes = b"col1\n1\n"
        upload = UploadFile(filename="application.csv", file=io.BytesIO(csv_bytes))
        with pytest.raises(file_validation.FileValidationError):
            file_validation.validate_csv_structure(upload, "application.csv")


class _FailingQuery:
    def __init__(self, result):
        self._result = result

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self._result


class _FakeSession:
    def __init__(self, user):
        self.user = user
        self.commits = 0

    def query(self, *_args, **_kwargs):
        return _FailingQuery(self.user)

    def commit(self):
        self.commits += 1


class TestAuth:
    def test_authenticate_user_success(self):
        password = "secret123"
        hashed = auth.hash_password(password)
        user = SimpleNamespace(
            username="u1",
            email="e@example.com",
            hashed_password=hashed,
            is_active=True,
            last_login=None,
        )
        session = _FakeSession(user)

        result = auth.authenticate_user(session, "u1", password)

        assert result is user
        assert session.commits == 1
        assert result.last_login is not None

    def test_authenticate_user_wrong_password(self):
        hashed = auth.hash_password("correct")
        user = SimpleNamespace(username="u1", email="e", hashed_password=hashed, is_active=True, last_login=None)
        session = _FakeSession(user)

        result = auth.authenticate_user(session, "u1", "wrong")

        assert result is None
        assert session.commits == 0


class _BrokenQuerySession:
    def query(self, *_args, **_kwargs):
        raise RuntimeError("db unavailable")


class TestDriftApiSummary:
    def test_stats_summary_handles_db_failure(self):
        # Override DB dependency to force exception and exercise graceful fallback
        app.dependency_overrides = {}
        app.dependency_overrides[get_db] = lambda: _BrokenQuerySession()
        client = TestClient(app)

        response = client.get("/monitoring/stats/summary")

        assert response.status_code == 200
        payload = response.json()
        assert payload["data_drift"]["total_features_checked"] == 0
        assert payload["predictions"]["total"] == 0


class TestMlflowLoaderFallback:
    def test_fallback_raises_when_missing_file(self, tmp_path):
        from api import mlflow_loader

        missing = tmp_path / "no_model.pkl"
        with pytest.raises(FileNotFoundError):
            mlflow_loader._load_fallback_model(missing)
