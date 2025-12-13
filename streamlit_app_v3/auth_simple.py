
import streamlit as st
import sys
from pathlib import Path

# Add root to path to import backend
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.database import get_db_context
from backend.auth import authenticate_user

def init_session():
    """Initialize session state variables."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.user = None
        st.session_state.role = None

def login(username, password):
    """Authenticate user against the database."""
    try:
        with get_db_context() as db:
            user = authenticate_user(db, username, password)
            if user:
                st.session_state.authenticated = True
                st.session_state.user = user.username
                st.session_state.role = user.role.value
                return True, "Success"
            return False, "Invalid credentials"
    except Exception as e:
        return False, f"Error: {str(e)}"

def logout():
    """Clear session and logout."""
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.role = None
    st.rerun()
