"""
Credit Scoring Dashboard - Main Application Entry Point

This is the main Streamlit application with tab-based navigation.
Includes authentication and role-based access control.
NO SIDEBAR - uses tabs for navigation.

Run with: streamlit run streamlit_app/app.py
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import authentication module
from streamlit_app.auth import (
    init_auth_session, 
    is_authenticated, 
    login_page, 
    logout_user,
    get_current_user
)
from backend.models import UserRole

# Page configuration - NO sidebar
st.set_page_config(
    page_title="Credit Scoring Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide sidebar completely with CSS
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        display: none;
    }
    [data-testid="stSidebarNav"] {
        display: none;
    }
    section[data-testid="stSidebar"] {
        display: none;
    }
    button[kind="header"] {
        display: none;
    }
    .css-1d391kg {
        display: none;
    }
    /* Hide hamburger menu */
    #MainMenu {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# Initialize authentication
init_auth_session()


def render_header():
    """Render the header with user info and logout."""
    user = get_current_user()
    if user:
        col1, col2, col3 = st.columns([6, 2, 1])
        with col1:
            st.markdown("### 🏦 Credit Scoring Dashboard")
        with col2:
            st.markdown(f"👤 **{user['username']}** ({user['role']})")
        with col3:
            if st.button("🚪 Logout"):
                logout_user()
                st.rerun()
        st.markdown("---")


def main():
    """Main application entry point."""
    
    # Check authentication - show ONLY login page if not authenticated
    if not is_authenticated():
        login_page()
        return
    
    # Render header with user info
    render_header()
    
    # Get user info for role-based access
    user = get_current_user()
    is_admin = user and user['role'] == UserRole.ADMIN.value
    
    # Create tab-based navigation (NO sidebar)
    if is_admin:
        # Admin has 3 tabs
        tab1, tab2, tab3 = st.tabs(["📈 Model Performance", "📁 Batch Predictions", "📉 Monitoring"])
        
        with tab1:
            render_model_performance_page()
        
        with tab2:
            render_batch_predictions_page()
        
        with tab3:
            render_monitoring_page()
    else:
        # Analyst has 2 tabs only
        tab1, tab2 = st.tabs(["📈 Model Performance", "📁 Batch Predictions"])
        
        with tab1:
            render_model_performance_page()
        
        with tab2:
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
