"""Pytest configuration file.

Handles test setup and fixtures.
"""
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.app import app

@pytest.fixture(scope="module")
def test_app_client():
    """Fixture to provide a TestClient instance for the app."""
    # Ensure app startup events are run to load the model
    with TestClient(app) as client:
        yield client
