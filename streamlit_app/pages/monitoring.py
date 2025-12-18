"""Monitoring Dashboard for Credit Scoring Application.

This page is admin-only and provides:
- API health monitoring
- Prediction statistics
- Data drift detection
- System metrics
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# Import JSON utils from local streamlit_app directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from json_utils import sanitize_for_json

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

    # Create tabs (admin-only) with persistence using the new 'default' parameter
    monitoring_labels = [
        "📊 Overview",
        "🧠 Model Monitoring",
        "⚡ Performance Monitoring",
        "🔍 Data Quality",
        "⚙️ System Health"
    ]
    
    if 'monitoring_active_tab' not in st.session_state:
        st.session_state.monitoring_active_tab = monitoring_labels[0]
        
    tabs = st.tabs(monitoring_labels, default=st.session_state.monitoring_active_tab)

    with tabs[0]:
        st.session_state.monitoring_active_tab = monitoring_labels[0]
        render_overview_tab()

    with tabs[1]:
        st.session_state.monitoring_active_tab = monitoring_labels[1]
        render_model_monitoring_tab()

    with tabs[2]:
        st.session_state.monitoring_active_tab = monitoring_labels[2]
        render_performance_monitoring_tab()

    with tabs[3]:
        st.session_state.monitoring_active_tab = monitoring_labels[3]
        render_data_quality_tab()

    with tabs[4]:
        st.session_state.monitoring_active_tab = monitoring_labels[4]
        render_system_health_tab()


def render_overview_tab():
    """Render the overview tab with key metrics."""
    st.markdown("### 📊 System Overview")

    # Fetch statistics
    try:
        # Keep timeouts short but reasonable
        stats_response = requests.get(f"{API_BASE_URL}/batch/statistics", timeout=10)
        health_response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        db_health_response = requests.get(f"{API_BASE_URL}/health/database", timeout=10)
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
                    f"{(stats_data.get('total_predictions') or 0):,}",
                    help="Total individual predictions made"
                )
                col4.metric(
                    "Avg Processing Time",
                    f"{(stats_data.get('average_processing_time_seconds') or 0):.2f}s",
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
                    st.plotly_chart(fig, width="stretch")

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
                    st.plotly_chart(fig, width="stretch")

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
        response = requests.get(f"{API_BASE_URL}/batch/history?limit=100", timeout=10)

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
                st.dataframe(df[display_cols], width="stretch")

                if 'avg_probability' in df.columns:
                    fig = px.line(
                        df,
                        x='id',
                        y='avg_probability',
                        title='Average Default Probability by Batch',
                        markers=True
                    )
                    st.plotly_chart(fig, width="stretch")
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
        health_resp = requests.get(f"{API_BASE_URL}/health", timeout=10)
        stats_resp = requests.get(f"{API_BASE_URL}/batch/statistics", timeout=10)

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
    st.markdown("### 🔍 Data Quality & Drift Detection")

    # Tab within the tab for different views
    quality_labels = [
        "📊 Feature Drift",
        "✔️ Data Quality",
        "📈 Drift History"
    ]
    
    if 'quality_active_tab' not in st.session_state:
        st.session_state.quality_active_tab = quality_labels[0]
        
    tabs = st.tabs(quality_labels, default=st.session_state.quality_active_tab)

    with tabs[0]:
        st.session_state.quality_active_tab = quality_labels[0]
        render_drift_detection()

    with tabs[1]:
        st.session_state.quality_active_tab = quality_labels[1]
        render_data_quality_checks()

    with tabs[2]:
        st.session_state.quality_active_tab = quality_labels[2]
        render_drift_history()


def render_drift_detection():
    """Render feature drift detection with dropdown selection and tabular results."""
    st.subheader("📊 Feature Drift Detection")
    
    st.info("""
    This uses statistical tests to detect distribution changes between a batch and the training data:
    - **KS Test**: For numeric features (p-value < 0.05 = drift)
    - **PSI**: Population Stability Index (>0.25 = significant drift)
    """)

    # Fetch available batches for dropdown
    try:
        response = requests.get(f"{API_BASE_URL}/batch/history?limit=100", timeout=15)
        if response.status_code == 200:
            batches = response.json().get('batches', [])
            batch_options = {f"Batch {b['id']} - {b['batch_name']}": b['id'] for b in batches if b['status'] == 'completed'}
        else:
            batch_options = {}
    except:
        batch_options = {}

    if not batch_options:
        st.warning("No completed batches found for analysis.")
        return

    with st.form("drift_detection_form"):
        col1, col2 = st.columns(2)

        with col1:
            selected_batch_label = st.selectbox(
                "Select Batch to analyze",
                options=list(batch_options.keys()),
                help="Choose a completed batch to compare against training data."
            )
            
            st.text_input("Reference Data", value="Training + Validation Data", disabled=True, 
                         help="The reference is fixed to the baseline training/validation dataset.")

        with col2:
            alert_threshold = st.slider("Alert Threshold (p-value)", 0.01, 0.1, 0.05, 
                                      help="P-values below this threshold signal statistical drift.")
            
            submit_drift = st.form_submit_button("🔍 Analyze Drift", type="primary", use_container_width=True)

    if submit_drift:
        batch_id = batch_options[selected_batch_label]
        # Pass 0 or None for reference_batch_id to trigger Training Data logic in API
        analyze_batch_drift(batch_id, 0, alert_threshold)

    # Display recent drift results summary
    st.markdown("---")
    st.subheader("📋 System-wide Drift Summary")

    try:
        response = requests.get(f"{API_BASE_URL}/monitoring/stats/summary", timeout=5)
        if response.status_code == 200:
            stats = response.json()

            col1, col2, col3 = st.columns(3)
            col1.metric("Features Checked", stats['data_drift']['total_features_checked'])
            col2.metric("Features with Drift", stats['data_drift']['features_with_drift'])
            col3.metric("Drift Rate", f"{stats['data_drift']['drift_percentage']:.1f}%")
    except:
        st.info("Global drift summary unavailable.")


def analyze_batch_drift(batch_id: int, reference_batch_id: int, alert_threshold: float):
    """Analyze drift for a batch and display in a tabular format."""
    try:
        with st.spinner("🚀 Performing statistical drift analysis..."):
            # reference_batch_id=0 triggers training data reference in our updated API
            params = {"reference_batch_id": reference_batch_id} if reference_batch_id > 0 else {}
            
            response = requests.post(
                f"{API_BASE_URL}/monitoring/drift/batch/{batch_id}",
                params=params,
                timeout=60
            )

            if response.status_code == 200:
                results = response.json()
                
                # Check if we have results
                if not results.get('results'):
                    st.warning(f"⚠️ No drift results returned. {results.get('features_checked', 0)} features were checked, but none could be analyzed.")
                    return

                st.success(f"✅ Analysis Complete: {results['features_checked']} features checked.")
                
                # Tabular results preparation
                drift_data = []
                for feature, r in results['results'].items():
                    drift_data.append({
                        "Status": "🔴 DRIFT" if r['is_drifted'] else "✅ OK",
                        "Feature Name": feature,
                        "KS Statistic": r.get('ks_statistic', 0),
                        "P-Value": r.get('p_value', 0),
                        "PSI": r.get('psi', 0),
                        "Ref Mean": r.get('reference_mean', 0),
                        "Batch Mean": r.get('current_mean', 0),
                        "Interpretation": r.get('interpretation', "")
                    })
                
                df_results = pd.DataFrame(drift_data)
                
                # Summary metrics for THIS analysis
                c1, c2, c3 = st.columns(3)
                c1.metric("Features Checked", results['features_checked'])
                c2.metric("Drifted Features", results['features_drifted'])
                drift_rate = (results['features_drifted'] / max(results['features_checked'], 1))
                c3.metric("Batch Drift Rate", f"{drift_rate:.1%}")

                st.markdown("### 📊 Detailed Results Table")
                
                if not df_results.empty and 'Status' in df_results.columns:
                    # Add highlighting for drifted features
                    def highlight_drift(s):
                        return ['background-color: #ffcccc' if v == "🔴 DRIFT" else '' for v in s]
                    
                    st.dataframe(
                        df_results.style.apply(highlight_drift, subset=['Status']),
                        width="stretch",
                        height=500
                    )
                else:
                    st.dataframe(df_results, width="stretch")
                
                # Export option
                csv = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Export Results to CSV",
                    csv,
                    f"drift_analysis_batch_{batch_id}.csv",
                    "text/csv",
                    key='download-csv'
                )

            else:
                st.error(f"Error: {response.json().get('detail', 'Unknown error')}")

    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")


def render_data_quality_checks():
    """Render data quality checks with dropdown selection."""
    st.subheader("✔️ Data Quality Analysis")

    st.info("""
    Automated checks for data integrity:
    - **Missing Values**: Alert if >20% missing
    - **Out-of-Range**: Values outside training data bounds
    - **Schema Validation**: Ensures all expected columns present
    """)

    # Fetch available batches for dropdown
    try:
        response = requests.get(f"{API_BASE_URL}/batch/history?limit=100", timeout=15)
        if response.status_code == 200:
            batches = response.json().get('batches', [])
            batch_options = {f"Batch {b['id']} - {b['batch_name']}": b['id'] for b in batches if b['status'] == 'completed'}
        else:
            batch_options = {}
    except:
        batch_options = {}

    if not batch_options:
        st.warning("No completed batches found for quality check.")
        return

    with st.form("data_quality_form"):
        selected_batch_label = st.selectbox(
            "Select Batch to check",
            options=list(batch_options.keys()),
            key="quality_batch_select",
            help="Choose a completed batch to analyze its data quality."
        )
        
        submit_quality = st.form_submit_button("🔍 Analyze Data Quality", type="primary", use_container_width=True)

    if submit_quality:
        batch_id = batch_options[selected_batch_label]
        check_data_quality(batch_id)


def check_data_quality(batch_id: int):
    """Check data quality for a batch."""
    try:
        with st.spinner("Checking data quality..."):
            # First get batch data
            from backend.models import PredictionBatch
            from backend.database import SessionLocal

            db = SessionLocal()
            batch = db.query(PredictionBatch).filter(PredictionBatch.id == batch_id).first()

            if not batch or not batch.raw_applications:
                st.warning("Batch not found or has no data")
                return

            # Convert to DataFrame
            data_list = [raw_app.raw_data for raw_app in batch.raw_applications if raw_app.raw_data]
            df = pd.DataFrame(data_list)

            # Call API for quality check with sanitized data
            response = requests.post(
                f"{API_BASE_URL}/monitoring/quality",
                json=sanitize_for_json({
                    "dataframe_dict": df.to_dict(orient='list'),
                    "check_missing": True,
                    "check_range": True,
                    "check_schema": False
                }),
                timeout=30
            )

            if response.status_code == 200:
                quality_result = response.json()

                # Status card
                status_color = "green" if quality_result['valid'] else "orange"
                status_icon = "✅" if quality_result['valid'] else "⚠️"
                st.markdown(
                    f"<h3 style='color: {status_color}'>{status_icon} {quality_result['summary']}</h3>",
                    unsafe_allow_html=True
                )

                # Missing values
                if quality_result['missing_values']:
                    st.subheader("📊 Missing Values Analysis")
                    
                    missing_df = pd.DataFrame(
                        list(quality_result['missing_values'].items()),
                        columns=['Feature', 'Missing %']
                    )
                    missing_df = missing_df.sort_values('Missing %', ascending=False)
                    
                    # Show summary stats
                    high_missing = missing_df[missing_df['Missing %'] > 20]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Features", len(missing_df))
                    with col2:
                        st.metric("High Missing (>20%)", len(high_missing), 
                                delta=f"{len(high_missing)/len(missing_df)*100:.1f}%" if len(missing_df) > 0 else "0%",
                                delta_color="inverse")
                    with col3:
                        st.metric("Average Missing", f"{missing_df['Missing %'].mean():.1f}%")
                    
                    # Show top 15 features with missing values
                    st.markdown("**Top 15 Features by Missing Values:**")
                    top_missing = missing_df.head(15).copy()
                    top_missing['Status'] = top_missing['Missing %'].apply(
                        lambda x: '🔴 Critical' if x > 50 else ('🟠 High' if x > 20 else '🟢 Acceptable')
                    )
                    st.dataframe(
                        top_missing[['Feature', 'Missing %', 'Status']], 
                        width="stretch",
                        hide_index=True
                    )
                    
                    # Show features with >20% missing in expander
                    if len(high_missing) > 0:
                        with st.expander(f"📋 View all {len(high_missing)} features with >20% missing"):
                            st.dataframe(
                                high_missing[['Feature', 'Missing %']], 
                                width="stretch",
                                hide_index=True
                            )

                # Out of range
                if quality_result['out_of_range']:
                    st.subheader("Out-of-Range Values")
                    issues = []
                    for col, info in quality_result['out_of_range'].items():
                        issues.append({
                            'Feature': col,
                            'Count': info['out_of_range_count'],
                            'Percentage': f"{info['out_of_range_pct']:.2f}%",
                            'Status': info['status']
                        })
                    issues_df = pd.DataFrame(issues)
                    st.dataframe(issues_df, width="stretch")

            else:
                st.error(f"Quality check failed: {response.json().get('detail')}")

    except Exception as e:
        st.error(f"Error checking quality: {str(e)}")


def render_drift_history():
    """Render drift detection history."""
    st.subheader("📈 Drift History")

    st.info("View historical drift detection results for features over time")

    feature_name = st.selectbox(
        "Select feature to view history",
        options=[
            'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
            'AMT_CREDIT', 'AMT_INCOME', 'DAYS_BIRTH', 'DAYS_EMPLOYED'
        ]
    )

    try:
        response = requests.get(
            f"{API_BASE_URL}/monitoring/drift/history/{feature_name}",
            params={"limit": 30},
            timeout=15
        )

        if response.status_code == 200:
            history = response.json()

            if history['records']:
                # Convert to DataFrame for visualization
                records = history['records']
                history_df = pd.DataFrame(records)

                # Plot drift scores over time
                st.line_chart(
                    history_df.set_index('recorded_at')['drift_score'],
                    width="stretch"
                )

                # Data table
                st.dataframe(history_df, width="stretch")

            else:
                st.info(f"No drift history available for {feature_name}")

        else:
            st.warning("Drift history not available")

    except Exception as e:
        st.info(f"Drift history feature: {str(e)}")


def render_system_health_tab():
    """Render the system health tab."""
    st.markdown("### ⚙️ System Health")

    try:
        # API Health
        health_response = requests.get(f"{API_BASE_URL}/health", timeout=15)
        db_health_response = requests.get(f"{API_BASE_URL}/health/database", timeout=15)

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

    # NEW: Backend API Resources
    st.markdown("---")
    st.subheader("🚀 Backend API Resources")
    
    try:
        res_response = requests.get(f"{API_BASE_URL}/health/resources", timeout=2)
        if res_response.status_code == 200:
            res_data = res_response.json()
            proc = res_data.get("process", {})
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("API Memory", f"{(proc.get('memory_rss_mb') or 0):.1f} MB")
            with col2:
                st.metric("API CPU", f"{(proc.get('cpu_percent') or 0):.1f}%")
            with col3:
                uptime = proc.get('uptime_seconds', 0)
                if uptime > 3600:
                    uptime_str = f"{uptime/3600:.1f}h"
                elif uptime > 60:
                    uptime_str = f"{uptime/60:.1f}m"
                else:
                    uptime_str = f"{uptime:.0f}s"
                st.metric("API Uptime", uptime_str)
            with col4:
                st.metric("API Threads", proc.get('threads', 0))
                
            # Show raw JSON in expander for debugging
            with st.expander("View Full Resource JSON"):
                st.json(res_data)
        else:
            st.warning("⚠️ API resource metrics temporarily unavailable")
    except Exception as e:
        st.info("Could not fetch backend resource metrics. Is the API running?")

    # System information
    st.markdown("---")
    st.subheader("💻 Machine Information (Streamlit Host)")

    import platform
    import os

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
