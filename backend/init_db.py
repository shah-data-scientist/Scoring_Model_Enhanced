"""
Database Initialization Script
==============================
Creates all tables and default admin user.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.database import engine, get_db_context, get_db_info, DATABASE_URL
from backend.models import Base, User, UserRole
from backend.auth import hash_password


def create_tables():
    """Create all database tables."""
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("[OK] All tables created successfully")


def create_default_admin(username: str = "admin", password: str = "admin123", 
                         email: str = "admin@creditscoring.local"):
    """Create default admin user if not exists."""
    with get_db_context() as db:
        # Check if admin exists
        existing_admin = db.query(User).filter(User.username == username).first()
        
        if existing_admin:
            print(f"[SKIP] Admin user '{username}' already exists")
            return existing_admin
        
        # Create admin user
        admin = User(
            username=username,
            email=email,
            password_hash=hash_password(password),
            role=UserRole.ADMIN,
            is_active=True
        )
        db.add(admin)
        db.commit()
        db.refresh(admin)
        
        print(f"[OK] Admin user created:")
        print(f"     Username: {username}")
        print(f"     Password: {password}")
        print(f"     Email: {email}")
        print(f"     Role: ADMIN")
        
        return admin


def create_default_analyst(username: str = "analyst", password: str = "analyst123",
                           email: str = "analyst@creditscoring.local"):
    """Create default analyst user if not exists."""
    with get_db_context() as db:
        # Check if analyst exists
        existing = db.query(User).filter(User.username == username).first()
        
        if existing:
            print(f"[SKIP] Analyst user '{username}' already exists")
            return existing
        
        # Create analyst user
        analyst = User(
            username=username,
            email=email,
            password_hash=hash_password(password),
            role=UserRole.ANALYST,
            is_active=True
        )
        db.add(analyst)
        db.commit()
        db.refresh(analyst)
        
        print(f"[OK] Analyst user created:")
        print(f"     Username: {username}")
        print(f"     Password: {password}")
        print(f"     Email: {email}")
        print(f"     Role: ANALYST")
        
        return analyst


def list_tables():
    """List all tables in the database."""
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print(f"\nDatabase tables ({len(tables)}):")
    for table in sorted(tables):
        columns = inspector.get_columns(table)
        print(f"  - {table} ({len(columns)} columns)")


def init_database(create_users: bool = True):
    """
    Initialize the database with tables and default users.
    
    Args:
        create_users: Whether to create default users
    """
    print("=" * 60)
    print("DATABASE INITIALIZATION")
    print("=" * 60)
    
    # Show database info
    db_info = get_db_info()
    print(f"\nDatabase: {db_info['database_url']}")
    print(f"Type: {'SQLite' if db_info['is_sqlite'] else 'PostgreSQL'}")
    
    # Create tables
    print()
    create_tables()
    
    # List tables
    list_tables()
    
    # Create default users
    if create_users:
        print()
        print("Creating default users...")
        create_default_admin()
        create_default_analyst()
    
    print()
    print("=" * 60)
    print("[OK] DATABASE INITIALIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize Credit Scoring Database")
    parser.add_argument("--no-users", action="store_true", 
                        help="Skip creating default users")
    parser.add_argument("--tables-only", action="store_true",
                        help="Only create tables, no users")
    
    args = parser.parse_args()
    
    create_users = not (args.no_users or args.tables_only)
    init_database(create_users=create_users)
