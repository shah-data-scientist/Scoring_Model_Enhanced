
import streamlit as st
import sys
from pathlib import Path

# Add root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_app_v3.auth_simple import init_session, login, logout

# Page Config
st.set_page_config(page_title="Credit Scoring V3", layout="wide", initial_sidebar_state="collapsed")

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
    /* .css-1d391kg { display: none; } REMOVED BRITTLE SELECTOR */
    #MainMenu {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# Init Session
init_session()

def login_page():
    st.title("🏦 Credit Scoring Login")
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", type="primary", use_container_width=True)
            
            if submitted:
                if not username or not password:
                    st.error("Please enter credentials")
                else:
                    success, msg = login(username, password)
                    if success:
                        st.rerun()
                    else:
                        st.error(msg)
        
        st.info("Default: admin / admin123")

def dashboard():
    # Main Content
    st.title("🚀 Credit Scoring Dashboard")
    
    # Simple Tabs
    tabs_list = ["📊 Model Performance", "📁 Batch Predictions"]
    if st.session_state.role == 'admin':
        tabs_list.append("⚙️ Monitoring")
        
    # selected_tab = st.tabs(tabs_list) # Use st.tabs
    nav_selection = st.radio("Navigate", tabs_list, horizontal=True)

    if nav_selection == tabs_list[0]: # Model Performance
        st.header("Model Performance")
        from streamlit_app_v3.pages.model_performance import render_model_performance_page
        render_model_performance_page()
    
    elif nav_selection == tabs_list[1]: # Batch Predictions
        from streamlit_app_v3.pages.batch_predictions import render_batch_predictions_page
        render_batch_predictions_page()
        
    elif len(tabs_list) > 2 and nav_selection == tabs_list[2]: # Monitoring
        if st.session_state.role == 'admin':
            st.header("System Monitoring")
            st.info("Admin metrics will go here.")

# Main Router
if not st.session_state.authenticated:
    login_page()
else:
    dashboard()
