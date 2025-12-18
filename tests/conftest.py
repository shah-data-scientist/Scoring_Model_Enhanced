"""Pytest configuration file.

Handles test setup and fixtures.
"""
import sys
import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set test database URL before importing anything that uses it
os.environ["TEST_DATABASE_URL"] = "sqlite:///:memory:"

from api.app import app
from backend.database import engine
from backend.models import Base

@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    """Initialize test database schema."""
    Base.metadata.create_all(bind=engine)
    yield
    # No need to drop for in-memory DB

@pytest.fixture(scope="module")
def test_app_client():
    """Fixture to provide a TestClient instance for the app."""
    # Ensure app startup events are run to load the model
    with TestClient(app) as client:
        yield client
