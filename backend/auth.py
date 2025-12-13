"""Authentication Module
=====================
Password hashing, user authentication, and session management.
Uses bcrypt for secure password hashing as required.
"""

from datetime import datetime, timedelta

import bcrypt
from sqlalchemy.orm import Session

from backend.models import User, UserRole

# =============================================================================
# PASSWORD HASHING (using bcrypt)
# =============================================================================

def hash_password(password: str) -> str:
    """Hash a password using bcrypt.
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password string

    """
    # Generate salt and hash
    # 10 rounds for faster login (still secure, ~100ms vs ~300ms with 12)
    salt = bcrypt.gensalt(rounds=10)
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash.
    
    Args:
        password: Plain text password to verify
        hashed: Stored password hash
        
    Returns:
        True if password matches, False otherwise

    """
    try:
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    except Exception:
        return False


# =============================================================================
# USER AUTHENTICATION
# =============================================================================

def authenticate_user(db: Session, username: str, password: str) -> User | None:
    """Authenticate a user by username and password.
    
    Args:
        db: Database session
        username: Username to authenticate
        password: Password to verify
        
    Returns:
        User object if authenticated, None otherwise

    """
    # Find user by username
    user = db.query(User).filter(
        User.username == username,
        User.is_active == True
    ).first()

    if not user:
        return None

    # Verify password
    if not verify_password(password, user.hashed_password):
        return None

    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()

    return user


def authenticate_by_email(db: Session, email: str, password: str) -> User | None:
    """Authenticate a user by email and password.
    
    Args:
        db: Database session
        email: Email to authenticate
        password: Password to verify
        
    Returns:
        User object if authenticated, None otherwise

    """
    # Find user by email
    user = db.query(User).filter(
        User.email == email,
        User.is_active == True
    ).first()

    if not user:
        return None

    # Verify password
    if not verify_password(password, user.hashed_password):
        return None

    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()

    return user


# =============================================================================
# USER MANAGEMENT
# =============================================================================

def create_user(
    db: Session,
    username: str,
    email: str,
    password: str,
    role: UserRole = UserRole.ANALYST
) -> User:
    """Create a new user.
    
    Args:
        db: Database session
        username: Unique username
        email: Unique email
        password: Plain text password (will be hashed)
        role: User role (default: ANALYST)
        
    Returns:
        Created User object
        
    Raises:
        ValueError: If username or email already exists

    """
    # Check if username exists
    if db.query(User).filter(User.username == username).first():
        raise ValueError(f"Username '{username}' already exists")

    # Check if email exists
    if db.query(User).filter(User.email == email).first():
        raise ValueError(f"Email '{email}' already exists")

    # Create user
    user = User(
        username=username,
        email=email,
        hashed_password=hash_password(password),
        role=role,
        is_active=True
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return user


def update_password(db: Session, user: User, new_password: str) -> bool:
    """Update a user's password.
    
    Args:
        db: Database session
        user: User to update
        new_password: New plain text password
        
    Returns:
        True if successful

    """
    user.hashed_password = hash_password(new_password)
    user.updated_at = datetime.utcnow()
    db.commit()
    return True


def deactivate_user(db: Session, user: User) -> bool:
    """Deactivate a user account.
    
    Args:
        db: Database session
        user: User to deactivate
        
    Returns:
        True if successful

    """
    user.is_active = False
    user.updated_at = datetime.utcnow()
    db.commit()
    return True


def get_user_by_id(db: Session, user_id: int) -> User | None:
    """Get user by ID."""
    return db.query(User).filter(User.id == user_id).first()


def get_user_by_username(db: Session, username: str) -> User | None:
    """Get user by username."""
    return db.query(User).filter(User.username == username).first()


def get_user_by_email(db: Session, email: str) -> User | None:
    """Get user by email."""
    return db.query(User).filter(User.email == email).first()


def list_users(db: Session, skip: int = 0, limit: int = 100) -> list:
    """List all users with pagination."""
    return db.query(User).offset(skip).limit(limit).all()


def is_admin(user: User) -> bool:
    """Check if user has admin role."""
    return user.role == UserRole.ADMIN


def is_analyst(user: User) -> bool:
    """Check if user has analyst role."""
    return user.role == UserRole.ANALYST


# =============================================================================
# SIMPLE SESSION MANAGEMENT (for Streamlit)
# =============================================================================

class SessionManager:
    """Simple in-memory session manager for Streamlit.
    
    For production, consider using Redis or database-backed sessions.
    """

    def __init__(self, session_timeout_hours: int = 8):
        self._sessions = {}
        self._timeout = timedelta(hours=session_timeout_hours)

    def create_session(self, user: User) -> str:
        """Create a new session for user."""
        import secrets
        session_id = secrets.token_urlsafe(32)
        self._sessions[session_id] = {
            'user_id': user.id,
            'username': user.username,
            'role': user.role.value,
            'created_at': datetime.utcnow(),
            'expires_at': datetime.utcnow() + self._timeout
        }
        return session_id

    def get_session(self, session_id: str) -> dict | None:
        """Get session data if valid."""
        session = self._sessions.get(session_id)
        if not session:
            return None

        # Check expiration
        if datetime.utcnow() > session['expires_at']:
            self.invalidate_session(session_id)
            return None

        return session

    def invalidate_session(self, session_id: str):
        """Remove a session."""
        self._sessions.pop(session_id, None)

    def cleanup_expired(self):
        """Remove all expired sessions."""
        now = datetime.utcnow()
        expired = [
            sid for sid, data in self._sessions.items()
            if data['expires_at'] < now
        ]
        for sid in expired:
            del self._sessions[sid]


# Global session manager instance
session_manager = SessionManager()
