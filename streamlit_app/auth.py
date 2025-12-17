"""Authentication Module for Streamlit Application.

This module provides user authentication functionality including:
- Login/logout
- Session management
- Role-based access control
"""

import sys
from pathlib import Path

import streamlit as st

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.auth import SessionManager, authenticate_user
from backend.database import get_db_context
from backend.models import UserRole

# Initialize session manager (in-memory for development)
session_manager = SessionManager(session_timeout_hours=8)


def init_auth_session():
    """Initialize authentication session state variables.
    
    TEMPORARY: Authentication is disabled, giving admin rights by default.
    """
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = True
    if 'username' not in st.session_state:
        st.session_state.username = "admin"
    if 'user_role' not in st.session_state:
        st.session_state.user_role = UserRole.ADMIN.value
    if 'user_email' not in st.session_state:
        st.session_state.user_email = "admin@example.com"
    if 'session_token' not in st.session_state:
        st.session_state.session_token = "dev_session_token"
    if 'showing_login' not in st.session_state:
        st.session_state.showing_login = False


def login_user(username: str, password: str) -> tuple[bool, str]:
    """Authenticate a user and create a session.
    
    Args:
        username: User's username
        password: User's password
        
    Returns:
        Tuple of (success, message)

    """
    try:
        with get_db_context() as db:
            user = authenticate_user(db, username, password)

            if user:
                # Create session (pass the User object)
                session_token = session_manager.create_session(user)

                # Update session state
                st.session_state.authenticated = True
                st.session_state.username = user.username
                st.session_state.user_role = user.role.value
                st.session_state.user_email = user.email
                st.session_state.session_token = session_token

                return True, f"Welcome, {user.username}!"
            return False, "Invalid username or password."

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return False, f"Authentication error: {str(e)}\n\nDetails: {error_details}"


def logout_user():
    """Log out the current user and clear session."""
    if st.session_state.session_token:
        session_manager.invalidate_session(st.session_state.session_token)

    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.user_role = None
    st.session_state.user_email = None
    st.session_state.session_token = None


def is_authenticated() -> bool:
    """Check if the current user is authenticated."""
    # Don't call init_auth_session here - it should be called once at app startup
    # This prevents recursive session state modifications
    
    if 'authenticated' not in st.session_state:
        return False
    
    return st.session_state.authenticated


def get_current_user() -> dict | None:
    """Get the current authenticated user's information."""
    if not is_authenticated():
        return None

    return {
        'username': st.session_state.username,
        'role': st.session_state.user_role,
        'email': st.session_state.user_email
    }


def require_auth(page_func):
    """Decorator to require authentication for a page.
    
    Usage:
        @require_auth
        def my_page():
            st.write("Protected content")
    """
    def wrapper():
        if not is_authenticated():
            login_page()
            st.stop()
        else:
            page_func()
    return wrapper


def require_admin(page_func):
    """Decorator to require admin role for a page.
    
    Usage:
        @require_admin
        def admin_page():
            st.write("Admin-only content")
    """
    def wrapper():
        if not is_authenticated():
            login_page()
            st.stop()
        elif st.session_state.user_role != UserRole.ADMIN.value:
            st.error("🔒 Access Denied: This page requires administrator privileges.")
            st.info("Please contact an administrator for access.")
            st.stop()
        else:
            page_func()
    return wrapper


def login_page():
    """Display the login page - full page dedicated to authentication."""
    st.session_state.showing_login = True
    
    # Header
    st.markdown("# 🏦 Credit Scoring Dashboard")
    st.markdown("---")

    # Login form (without columns for debugging)
    st.markdown("## 🔐 Login")
    st.markdown("Please log in to access the Credit Scoring Dashboard.")
    st.markdown("")

    with st.form("login_form", clear_on_submit=True):
        username = st.text_input("Username", placeholder="Enter your username", key="login_username")
        password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_password")

        submitted = st.form_submit_button("Login", use_container_width=True, type="primary")

        if submitted:
            if not username or not password:
                st.error("Please enter both username and password.")
            else:
                with st.spinner("🔐 Authenticating..."):
                    success, message = login_user(username, password)
                if success:
                    st.success(message)
                    st.info("🔄 Loading dashboard... Please wait.")
                    # Clear login flag
                    st.session_state.showing_login = False
                    st.rerun()
                else:
                    st.error(message)

    st.markdown("---")
    st.info("""
    **Default credentials for testing:**
    - Analyst: `analyst` / `analyst123`
    - Admin: `admin` / `admin123`
    """)



def render_user_sidebar():
    """Render user information and logout button in sidebar."""
    if is_authenticated():
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 👤 User Information")
        st.sidebar.write(f"**Username:** {st.session_state.username}")
        st.sidebar.write(f"**Role:** {st.session_state.user_role}")

        if st.sidebar.button("🚪 Logout", use_container_width=True):
            logout_user()
            st.rerun()
