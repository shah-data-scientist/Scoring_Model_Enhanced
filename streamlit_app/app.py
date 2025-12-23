"""Credit Scoring Dashboard - Main Application Entry Point

This is the main Streamlit application with tab-based navigation.
Includes authentication and role-based access control.
NO SIDEBAR - uses tabs for navigation.

Run with: streamlit run streamlit_app/app.py
"""

import sys
from pathlib import Path
import logging
from datetime import datetime

import streamlit as st

# Add project root to path before importing local packages
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_app.config import API_BASE_URL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'streamlit_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import authentication module
from backend.models import UserRole
from streamlit_app.auth import (
    get_current_user,
    init_auth_session,
    is_authenticated,
    login_page,
    logout_user,
)

# Page configuration - NO sidebar
st.set_page_config(
    page_title="Credit Scoring Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide sidebar completely with CSS and ensure clean page transitions
# st.markdown("""
# <style>
#     [data-testid="stSidebar"] {
#         display: none;
#     }
#     [data-testid="stSidebarNav"] {
#         display: none;
#     }
#     section[data-testid="stSidebar"] {
#         display: none;
#     }
#     button[kind="header"] {
#         display: none;
#     }
#     /* Hide hamburger menu */
#     #MainMenu {
#         visibility: hidden;
#     }
# </style>
# """, unsafe_allow_html=True)

# Initialize authentication (only once per session)
if 'auth_initialized' not in st.session_state:
    init_auth_session()
    st.session_state.auth_initialized = True


def render_header():
    """Render the header with user info and logout."""
    # Get user from session state
    username = st.session_state.get('username')
    user_role = st.session_state.get('user_role')

    if username and user_role:
        col1, col2, col3 = st.columns([6, 2, 1])
        with col1:
            st.markdown("### 🏦 Credit Scoring Dashboard")
        with col2:
            st.markdown(f"👤 **{username}** ({user_role})")
        with col3:
            if st.button("🚪 Logout"):
                logout_user()
                st.rerun()
        st.markdown("---")


def main():
    """Main application entry point."""
    # Check authentication - show ONLY login page if not authenticated
    if not is_authenticated():
        logger.info("User not authenticated - showing login page")
        login_page()
        # Return here to let Streamlit finish rendering the login page
        return
    
    # User is authenticated - ensure login page is NOT showing
    if st.session_state.get('showing_login', False):
        st.session_state.session_state.showing_login = False

    # Track if this is a fresh session (first load after login)
    if 'initial_load_complete' not in st.session_state:
        st.session_state.initial_load_complete = False

    # User is authenticated - render main dashboard
    # Use a container to ensure clean separation from login page
    main_container = st.container()

    with main_container:
        # Render header with user info
        render_header()

        # Get user info for role-based access
        is_admin = st.session_state.get('user_role') == UserRole.ADMIN.value

        # Show loading indicator on first load only
        if not st.session_state.initial_load_complete:
            with st.spinner("🔄 Loading dashboard... This may take a moment on first access."):
                # Pre-warm API connection
                try:
                    import requests
                    requests.get(f"{API_BASE_URL}/health", timeout=1)
                except Exception:
                    pass  # API might not be running - that's OK
            st.session_state.initial_load_complete = True

        # Create tab-based navigation
        if is_admin:
            tab_labels = ["📈 Model Performance", "📁 Batch Predictions", "📉 Monitoring"]

            # Use session state to track active tab by label
            if 'main_active_tab' not in st.session_state:
                st.session_state.main_active_tab = tab_labels[0]

            tabs = st.tabs(tab_labels)

            with tabs[0]:
                st.session_state.main_active_tab = tab_labels[0]
                logger.info("Rendering Model Performance tab")
                render_model_performance_page()

            with tabs[1]:
                st.session_state.main_active_tab = tab_labels[1]
                logger.info("Rendering Batch Predictions tab")
                render_batch_predictions_page()

            with tabs[2]:
                st.session_state.main_active_tab = tab_labels[2]
                logger.info("Rendering Monitoring tab")
                render_monitoring_page()
        else:
            tab_labels = ["📈 Model Performance", "📁 Batch Predictions"]

            if 'main_active_tab' not in st.session_state:
                st.session_state.main_active_tab = tab_labels[0]

            tabs = st.tabs(tab_labels)

            with tabs[0]:
                st.session_state.main_active_tab = tab_labels[0]
                logger.info("Rendering Model Performance tab")
                render_model_performance_page()

            with tabs[1]:
                st.session_state.main_active_tab = tab_labels[1]
                logger.info("Rendering Batch Predictions tab")
                render_batch_predictions_page()


def render_model_performance_page():
    """Render the model performance page."""
    try:
        from streamlit_app.pages.model_performance import render_model_performance
        render_model_performance()
    except ImportError as e:
        st.error(f"Page not available: {e}")


def render_batch_predictions_page():
    """Render the batch predictions page."""
    try:
        from streamlit_app.pages.batch_predictions import render_batch_predictions
        render_batch_predictions()
    except ImportError as e:
        st.error(f"Page not available: {e}")


def render_monitoring_page():
    """Render the monitoring page (admin only)."""
    try:
        from streamlit_app.pages.monitoring import render_monitoring
        render_monitoring()
    except ImportError as e:
        st.error(f"Page not available: {e}")


if __name__ == "__main__":
    main()
