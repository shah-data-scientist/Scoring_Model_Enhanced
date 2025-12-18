"""Database Configuration
======================
Database connection settings and session management.
"""

import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

# Load environment variables
load_dotenv()

# Project root for SQLite fallback
PROJECT_ROOT = Path(__file__).parent.parent


def get_database_url() -> str:
    """Get database URL from environment or use default.
    
    Priority:
    1. TEST_DATABASE_URL environment variable (for unit tests)
    2. DATABASE_URL environment variable (Explicit override)
    3. Individual PostgreSQL environment variables
    4. SQLite fallback for local development
    """
    # 1. Check for test database (Highest priority for testing reliability)
    test_db_url = os.getenv("TEST_DATABASE_URL")
    if test_db_url:
        return test_db_url

    # 2. Check for complete DATABASE_URL
    database_url = os.getenv("DATABASE_URL")
    if database_url and "${" not in database_url:
        return database_url

    # Check for PostgreSQL environment variables
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "credit_scoring")
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD")

    if db_host and db_password:
        return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    # Fallback to SQLite for local development
    sqlite_path = PROJECT_ROOT / "data" / "credit_scoring.db"
    return f"sqlite:///{sqlite_path}"


# Database URL
DATABASE_URL = get_database_url()

# Determine if using SQLite
IS_SQLITE = DATABASE_URL.startswith("sqlite")

# Create engine with appropriate settings
if IS_SQLITE:
    # SQLite-specific settings
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False  # Set to True for SQL debugging
    )
else:
    # PostgreSQL settings
    engine = create_engine(
        DATABASE_URL,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800,
        echo=False
    )

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Dependency for FastAPI to get database session.
    
    Usage:
        @app.get("/endpoint")
        def endpoint(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """Context manager for database session.
    
    Usage:
        with get_db_context() as db:
            db.query(...)
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def test_connection() -> bool:
    """Test database connection."""
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False


# Export database info
def get_db_info() -> dict:
    """Get database configuration info."""
    return {
        "database_url": DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else DATABASE_URL,
        "is_sqlite": IS_SQLITE,
        "is_postgresql": not IS_SQLITE,
        "connected": test_connection()
    }
