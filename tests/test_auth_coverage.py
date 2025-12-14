"""Comprehensive tests for backend.auth to increase coverage."""

from datetime import datetime
from types import SimpleNamespace
import pytest
from sqlalchemy.orm import Session

from backend import auth
from backend.models import User, UserRole


class MockUser:
    def __init__(self, username="testuser", email="test@test.com", password="password123", role=UserRole.ANALYST, is_active=True):
        self.id = 1
        self.username = username
        self.email = email
        self.hashed_password = auth.hash_password(password)
        self.role = role
        self.is_active = is_active
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.last_login = None


class MockSession:
    def __init__(self, users=None):
        self.users = users or {}
        self.added = []
        self.committed = False

    def query(self, model):
        return MockQuery(self.users.get(model, []))

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.committed = True

    def refresh(self, obj):
        pass


class MockQuery:
    def __init__(self, users):
        self.users = users if isinstance(users, list) else [users] if users else []
        self.filters = []

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self.users[0] if self.users else None

    def __getitem__(self, idx):
        return self.users[idx] if idx < len(self.users) else None


class TestPasswordHashing:
    def test_hash_password_creates_hash(self):
        """Test password hashing creates non-empty hash."""
        hashed = auth.hash_password("password123")
        assert hashed is not None
        assert len(hashed) > 0
        assert hashed != "password123"

    def test_hash_different_passwords_different_hashes(self):
        """Test different passwords produce different hashes."""
        hash1 = auth.hash_password("password1")
        hash2 = auth.hash_password("password2")
        assert hash1 != hash2

    def test_hash_same_password_different_hashes(self):
        """Test same password hashed twice produces different hashes (due to salt)."""
        hash1 = auth.hash_password("password123")
        hash2 = auth.hash_password("password123")
        # Due to random salt, hashes should be different
        assert hash1 != hash2

    def test_hash_empty_password(self):
        """Test hashing empty password."""
        hashed = auth.hash_password("")
        assert hashed is not None
        assert len(hashed) > 0


class TestPasswordVerification:
    def test_verify_correct_password(self):
        """Test verifying correct password."""
        password = "secret123"
        hashed = auth.hash_password(password)
        assert auth.verify_password(password, hashed) is True

    def test_verify_incorrect_password(self):
        """Test verifying incorrect password."""
        password = "secret123"
        hashed = auth.hash_password(password)
        assert auth.verify_password("wrong_password", hashed) is False

    def test_verify_empty_password(self):
        """Test verifying empty password."""
        hashed = auth.hash_password("nonempty")
        assert auth.verify_password("", hashed) is False

    def test_verify_with_invalid_hash(self):
        """Test verify handles invalid hash gracefully."""
        assert auth.verify_password("password", "invalid_hash") is False


class TestUserAuthentication:
    def test_authenticate_user_success(self):
        """Test successful user authentication by username."""
        user = MockUser(username="alice", password="pass123")
        session = MockSession({User: [user]})
        
        # Mock the query to return our user
        class AuthQuery:
            def filter(self, *args, **kwargs):
                return self
            def first(self):
                return user
        
        original_query = session.query
        session.query = lambda model: AuthQuery()
        
        result = auth.authenticate_user(session, "alice", "pass123")
        assert result is user
        assert result.last_login is not None

    def test_authenticate_user_wrong_password(self):
        """Test authentication fails with wrong password."""
        user = MockUser(username="alice", password="correct")
        
        class BadQuery:
            def filter(self, *args, **kwargs):
                return self
            def first(self):
                return user
        
        session = MockSession()
        session.query = lambda model: BadQuery()
        
        result = auth.authenticate_user(session, "alice", "wrong")
        assert result is None

    def test_authenticate_user_not_found(self):
        """Test authentication fails for nonexistent user."""
        class NoUserQuery:
            def filter(self, *args, **kwargs):
                return self
            def first(self):
                return None
        
        session = MockSession()
        session.query = lambda model: NoUserQuery()
        
        result = auth.authenticate_user(session, "nonexistent", "password")
        assert result is None

    def test_authenticate_inactive_user(self):
        """Test inactive user cannot authenticate."""
        user = MockUser(username="bob", password="pass", is_active=False)
        
        class InactiveQuery:
            def filter(self, *args, **kwargs):
                return self
            def first(self):
                return None  # Simulate is_active filter
        
        session = MockSession()
        session.query = lambda model: InactiveQuery()
        
        result = auth.authenticate_user(session, "bob", "pass")
        assert result is None


class TestAuthByEmail:
    def test_authenticate_by_email_success(self):
        """Test successful authentication by email."""
        user = MockUser(email="alice@example.com", password="pass123")
        
        class EmailQuery:
            def filter(self, *args, **kwargs):
                return self
            def first(self):
                return user
        
        session = MockSession()
        session.query = lambda model: EmailQuery()
        
        result = auth.authenticate_by_email(session, "alice@example.com", "pass123")
        assert result is user

    def test_authenticate_by_email_wrong_password(self):
        """Test email auth fails with wrong password."""
        user = MockUser(email="user@test.com", password="correct")
        
        class EmailQuery:
            def filter(self, *args, **kwargs):
                return self
            def first(self):
                return user
        
        session = MockSession()
        session.query = lambda model: EmailQuery()
        
        result = auth.authenticate_by_email(session, "user@test.com", "wrong")
        assert result is None

    def test_authenticate_by_email_not_found(self):
        """Test email auth fails for nonexistent email."""
        class NoEmailQuery:
            def filter(self, *args, **kwargs):
                return self
            def first(self):
                return None
        
        session = MockSession()
        session.query = lambda model: NoEmailQuery()
        
        result = auth.authenticate_by_email(session, "notfound@test.com", "pass")
        assert result is None


class TestCreateUser:
    def test_create_user_success(self):
        """Test successful user creation."""
        class CreateQuery:
            def __init__(self):
                self.existing = False
            def filter(self, *args, **kwargs):
                return self
            def first(self):
                return None if not self.existing else SimpleNamespace(id=1)
        
        session = MockSession()
        session.query = lambda model: CreateQuery()
        
        new_user = auth.create_user(session, "newuser", "new@test.com", "password123")
        
        assert new_user is not None
        assert new_user.username == "newuser"
        assert new_user.email == "new@test.com"
        assert session.committed

    def test_create_user_duplicate_username(self):
        """Test creation fails for duplicate username."""
        existing_user = MockUser(username="taken")
        
        class DupeQuery:
            def __init__(self, user=None):
                self.user = user
            def filter(self, *args, **kwargs):
                return DupeQuery(self.user)
            def first(self):
                return self.user
        
        session = MockSession()
        session.query = lambda model: DupeQuery(existing_user if model == User else None)
        
        with pytest.raises(ValueError, match="already exists"):
            auth.create_user(session, "taken", "new@test.com", "pass")

    def test_create_user_default_role(self):
        """Test created user has default ANALYST role."""
        class RoleQuery:
            def filter(self, *args, **kwargs):
                return self
            def first(self):
                return None
        
        session = MockSession()
        session.query = lambda model: RoleQuery()
        
        user = auth.create_user(session, "user1", "user1@test.com", "pass")
        assert user.role == UserRole.ANALYST


class TestUpdatePassword:
    def test_update_password_success(self):
        """Test successful password update."""
        user = MockUser(username="testuser")
        session = MockSession()
        
        old_hash = user.hashed_password
        result = auth.update_password(session, user, "newpassword123")
        
        assert result is True
        assert user.hashed_password != old_hash
        assert auth.verify_password("newpassword123", user.hashed_password)
        assert session.committed

    def test_update_password_timestamp(self):
        """Test password update sets updated_at."""
        user = MockUser()
        old_time = user.updated_at
        session = MockSession()
        
        auth.update_password(session, user, "newpass")
        
        assert user.updated_at > old_time


class TestDeactivateUser:
    def test_deactivate_user_success(self):
        """Test successful user deactivation."""
        user = MockUser(is_active=True)
        session = MockSession()
        
        result = auth.deactivate_user(session, user)
        
        assert result is True
        assert user.is_active is False
        assert session.committed

    def test_deactivate_already_inactive(self):
        """Test deactivating already inactive user."""
        user = MockUser(is_active=False)
        session = MockSession()
        
        result = auth.deactivate_user(session, user)
        
        assert result is True
        assert user.is_active is False


class TestGetUserById:
    def test_get_user_by_id(self):
        """Test retrieving user by ID."""
        user = MockUser()
        
        class IdQuery:
            def filter(self, *args, **kwargs):
                return self
            def first(self):
                return user
        
        session = MockSession()
        session.query = lambda model: IdQuery()
        
        result = auth.get_user_by_id(session, 1)
        assert result is user


class TestGetUserByUsername:
    def test_get_user_by_username(self):
        """Test retrieving user by username."""
        user = MockUser(username="alice")
        
        class NameQuery:
            def filter(self, *args, **kwargs):
                return self
            def first(self):
                return user
        
        session = MockSession()
        session.query = lambda model: NameQuery()
        
        result = auth.get_user_by_username(session, "alice")
        assert result is user


class TestIsAdmin:
    def test_is_admin_true(self):
        """Test is_admin returns True for admin."""
        user = MockUser(role=UserRole.ADMIN)
        assert auth.is_admin(user) is True

    def test_is_admin_false(self):
        """Test is_admin returns False for non-admin."""
        user = MockUser(role=UserRole.ANALYST)
        assert auth.is_admin(user) is False
