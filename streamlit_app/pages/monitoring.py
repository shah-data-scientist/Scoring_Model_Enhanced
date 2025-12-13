"""Monitoring Dashboard for Credit Scoring Application.

This page is admin-only and provides:
- API health monitoring
- Prediction statistics
- Data drift detection
- System metrics
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_app.config import API_BASE_URL


def render_monitoring():
    """Render the monitoring dashboard."""
    # Check admin access (should already be checked by app.py)
    from backend.models import UserRole
    from streamlit_app.auth import get_current_user

    user = get_current_user()
    if not user or user['role'] != UserRole.ADMIN.value:
        st.error("🔒 Access Denied: Admin privileges required")
        return

    # Create tabs (admin-only)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🧠 Model Monitoring",
        "⚡ Performance Monitoring",
        "🔍 Data Quality",
        "⚙️ System Health"
    ])

    with tab1:
        render_overview_tab()

    with tab2:
        render_model_monitoring_tab()

    with tab3:
        render_performance_monitoring_tab()

    with tab4:
        render_data_quality_tab()

    with tab5:
        render_system_health_tab()


def render_overview_tab():
    """Render the overview tab with key metrics."""
    st.markdown("### 📊 System Overview")

    # Fetch statistics
    try:
        # Keep timeouts short to avoid persistent Streamlit spinner if API is down
        stats_response = requests.get(f"{API_BASE_URL}/batch/statistics", timeout=2)
        health_response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        db_health_response = requests.get(f"{API_BASE_URL}/health/database", timeout=2)

        if stats_response.status_code == 200 and health_response.status_code == 200:
            stats = stats_response.json()
            health = health_response.json()

            # Get database health
            db_health = {}
            if db_health_response.status_code == 200:
                db_health = db_health_response.json()

            # Health indicators
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                status = health.get('status', 'unknown')
                color = "🟢" if status == 'healthy' else "🔴"
                st.metric("API Status", f"{color} {status.upper()}")

            with col2:
                model_status = "loaded" if health.get('model_loaded') else "not loaded"
                color = "🟢" if health.get('model_loaded') else "🔴"
                st.metric("Model Status", f"{color} {model_status.upper()}")

            with col3:
                db_connected = db_health.get('connected', False)
                db_status = "connected" if db_connected else "disconnected"
                color = "🟢" if db_connected else "🔴"
                st.metric("Database", f"{color} {db_status.upper()}")

            with col4:
                # Get model name from health or use default
                model_name = health.get('model_name')
                if not model_name or model_name in ['N/A', 'unknown', None]:
                    model_name = 'LightGBM'  # Default model type used in this project
                st.metric("Model Name", model_name)

            st.markdown("---")

            # Statistics cards
            if stats.get('success'):
                stats_data = stats.get('statistics', {})

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Total Batches",
                    stats_data.get('total_batches', 0),
                    help="Total number of batch predictions processed"
                )
                col2.metric(
                    "Completed",
                    stats_data.get('completed_batches', 0),
                    help="Successfully completed batches"
                )
                col3.metric(
                    "Total Predictions",
                    f"{stats_data.get('total_predictions', 0):,}",
                    help="Total individual predictions made"
                )
                col4.metric(
                    "Avg Processing Time",
                    f"{stats_data.get('average_processing_time_seconds', 0):.2f}s",
                    help="Average time to process a batch"
                )

                # Risk distribution chart
                st.markdown("### Risk Distribution")
                risk_dist = stats_data.get('risk_distribution', {})

                if risk_dist:
                    # Create pie chart
                    fig = px.pie(
                        values=list(risk_dist.values()),
                        names=list(risk_dist.keys()),
                        color=list(risk_dist.keys()),
                        color_discrete_map={
                            'LOW': 'green',
                            'MEDIUM': 'yellow',
                            'HIGH': 'orange',
                            'CRITICAL': 'red'
                        },
                        title="Overall Risk Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Daily predictions chart
                daily = stats.get('daily_predictions', [])
                if daily:
                    daily_df = pd.DataFrame(daily)
                    fig = px.bar(
                        daily_df,
                        x='date',
                        y='count',
                        title='Daily Prediction Volume',
                        labels={'date': 'Date', 'count': 'Predictions'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

        else:
            st.error("Failed to fetch system statistics")
            st.info("💡 The API returned an error. Check that the backend is running and responding correctly.")

    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API")
        st.info("""
        **The monitoring API is not reachable.** 
        
        Make sure the FastAPI backend is running:
        ```bash
        poetry run uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
        ```
        """)
    except requests.exceptions.Timeout:
        st.warning("⏱️ API request timed out")
        st.info("The backend is taking too long to respond. It may be under heavy load.")
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("An unexpected error occurred while fetching monitoring data.")


def render_model_monitoring_tab():
    """Render the model monitoring tab (predictions, drift signals, top features)."""
    st.markdown("### 🧠 Model Monitoring")

    try:
        response = requests.get(f"{API_BASE_URL}/batch/history?limit=100", timeout=2)

        if response.status_code == 200:
            data = response.json()
            batches = data.get('batches', [])

            if batches:
                df = pd.DataFrame(batches)

                col1, col2, col3 = st.columns(3)
                with col1:
                    total_apps = df['total_applications'].sum()
                    st.metric("Total Applications Processed", f"{total_apps:,}")
                with col2:
                    avg_prob = df['avg_probability'].mean()
                    st.metric("Average Default Probability", f"{avg_prob:.1%}")
                with col3:
                    drift_flag = df['avg_probability'].diff().abs().fillna(0).max() if 'avg_probability' in df else 0
                    st.metric("Max Prob Shift", f"{drift_flag:.2%}")

                st.markdown("---")

                st.subheader("Recent Batches")
                display_cols = ['id', 'batch_name', 'status', 'total_applications', 'avg_probability']
                display_cols = [c for c in display_cols if c in df.columns]
                st.dataframe(df[display_cols], use_container_width=True)

                if 'avg_probability' in df.columns:
                    fig = px.line(
                        df,
                        x='id',
                        y='avg_probability',
                        title='Average Default Probability by Batch',
                        markers=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No batch data available yet.")

        else:
            st.error(f"Failed to fetch data: {response.status_code}")
            st.info("The batch history endpoint returned an error.")

    except requests.exceptions.ConnectionError:
        st.warning("❌ Model monitoring data unavailable (API offline)")
        st.info("Start the API backend to see batch prediction history and model metrics.")
    except requests.exceptions.Timeout:
        st.warning("⏱️ Request timed out")
    except Exception as e:
        st.error(f"Error: {str(e)}")


def render_performance_monitoring_tab():
    """Render performance monitoring (latency, uptime)."""
    st.markdown("### ⚡ Performance Monitoring")

    try:
        # Fast health endpoint for latency
        health_resp = requests.get(f"{API_BASE_URL}/health", timeout=2)
        stats_resp = requests.get(f"{API_BASE_URL}/batch/statistics", timeout=2)

        if health_resp.status_code == 200:
            health = health_resp.json()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("API Status", health.get('status', 'unknown'))
            with col2:
                st.metric("Model Loaded", "Yes" if health.get('model_loaded') else "No")
            with col3:
                # Get model name from health or use default
                model_name = health.get('model_name')
                if not model_name or model_name in ['N/A', 'unknown', None]:
                    model_name = 'LightGBM'
                st.metric("Model Name", model_name)
        else:
            st.warning("Health endpoint did not return OK.")

        if stats_resp.status_code == 200:
            stats = stats_resp.json().get('statistics', {})
            latency = stats.get('average_processing_time_seconds', 0)
            st.metric("Avg Batch Processing Time", f"{latency:.2f}s")
        else:
            st.info("No statistics available yet.")

    except requests.exceptions.ConnectionError:
        st.warning("❌ Performance data unavailable (API offline)")
        st.info("Start the API backend to see performance metrics.")
    except requests.exceptions.Timeout:
        st.warning("⏱️ Request timed out")
    except Exception as e:
        st.error(f"Error: {str(e)}")


def render_data_quality_tab():
    """Render the data quality and drift detection tab."""
    st.markdown("### 🔍 Data Quality Monitoring")

    st.warning("""
    ⚠️ **Feature Under Development**
    
    Data drift detection and quality monitoring features are planned for a future release.
    This tab shows a preview of the intended functionality.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Planned Feature Drift Detection:**
        - Distribution changes over time
        - Statistical tests (KS, Chi-square)
        - Alert thresholds
        """)

    with col2:
        st.markdown("""
        **Planned Data Quality Checks:**
        - Missing value rates
        - Out-of-range values
        - Schema validation
        """)

    # Placeholder for drift metrics
    st.markdown("---")
    st.subheader("📊 Sample Feature Statistics Preview")
    st.caption("The table below shows example data to illustrate the planned functionality.")

    # Create sample data for demonstration
    sample_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'AMT_CREDIT', 'AMT_INCOME']
    sample_data = {
        'Feature': sample_features,
        'Training Mean': [0.502, 0.514, 0.511, 538396, 168798],
        'Current Mean': [0.498, 0.520, 0.505, 542100, 171200],
        'Drift Score': [0.02, 0.04, 0.03, 0.01, 0.02],
        'Status': ['✅ OK', '⚠️ Watch', '✅ OK', '✅ OK', '✅ OK']
    }

    drift_df = pd.DataFrame(sample_data)
    st.dataframe(drift_df, use_container_width=True)

    st.info("💡 **Implementation Note:** Real drift detection requires collecting prediction data over time and comparing against training data distributions.")


def render_system_health_tab():
    """Render the system health tab."""
    st.markdown("### ⚙️ System Health")

    try:
        # API Health
        health_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        db_health_response = requests.get(f"{API_BASE_URL}/health/database", timeout=5)

        if health_response.status_code == 200:
            health = health_response.json()
            db_health = {}
            if db_health_response.status_code == 200:
                db_health = db_health_response.json()

            st.subheader("API Health")

            # Display health info
            col1, col2 = st.columns(2)

            with col1:
                st.json(health)

            with col2:
                st.markdown("**Status Indicators:**")

                checks = [
                    ("API Responsive", health.get('status') == 'healthy'),
                    ("Model Loaded", health.get('model_loaded', False)),
                    ("Database Connected", db_health.get('connected', False)),
                ]

                for check_name, passed in checks:
                    icon = "✅" if passed else "❌"
                    st.write(f"{icon} {check_name}")

        # Database health
        st.markdown("---")
        st.subheader("Database Health")

        if db_health_response.status_code == 200:
            db_info = db_health_response.json()
            st.json(db_info)
        else:
            st.warning("Database health check not available")

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API server")
    except Exception as e:
        st.error(f"Error: {str(e)}")

    # System information
    st.markdown("---")
    st.subheader("System Information")

    import platform

    try:
        import psutil
        has_psutil = True
    except ImportError:
        has_psutil = False

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Python Version", platform.python_version())
        st.metric("Platform", platform.system())

    with col2:
        if has_psutil:
            try:
                memory = psutil.virtual_memory()
                st.metric("Memory Usage", f"{memory.percent}%")
                st.metric("Available Memory", f"{memory.available / (1024**3):.1f} GB")
            except:
                st.write("Memory info not available")
        else:
            st.info("Install psutil for memory metrics")

    with col3:
        if has_psutil:
            try:
                # Use interval=None for non-blocking call (returns instant value based on last call)
                cpu_percent = psutil.cpu_percent(interval=None)
                st.metric("CPU Usage", f"{cpu_percent}%")
                st.metric("CPU Cores", psutil.cpu_count())
            except:
                st.write("CPU info not available")
        else:
            st.info("Install psutil for CPU metrics")
