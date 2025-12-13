"""Tests for backend authentication module."""
import pytest
from backend.auth import (
    verify_password,
    hash_password,
)


class TestPasswordHashing:
    """Tests for password hashing functions."""

    def test_hash_password(self):
        """Test password hashing."""
        password = "testpassword123"
        hashed = hash_password(password)

        assert hashed != password
        assert len(hashed) > 0
        assert verify_password(password, hashed)

    def test_verify_correct_password(self):
        """Test verifying correct password."""
        password = "correct_password"
        hashed = hash_password(password)

        assert verify_password(password, hashed)

    def test_verify_incorrect_password(self):
        """Test verifying incorrect password."""
        password = "correct_password"
        wrong_password = "wrong_password"
        hashed = hash_password(password)

        assert not verify_password(wrong_password, hashed)
