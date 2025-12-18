"""Batch Predictions Page for Credit Scoring Dashboard.

This page allows users to:
- Upload CSV files with multiple applications
- Process batch predictions
- Download results with SHAP explanations
- Generate detailed HTML reports with waterfall plots
"""

import base64
import io
import sys
import zipfile
from datetime import datetime
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logger = logging.getLogger(__name__)

from streamlit_app.config import API_BASE_URL

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
END_USER_TESTS_DIR = DATA_DIR / "end_user_tests"


def render_batch_predictions():
    """Render the batch predictions interface."""
    # Create two tabs: Upload & Predict, Download Reports
    batch_labels = ["Upload & Predict", "Download Reports"]
    
    if 'batch_active_tab' not in st.session_state:
        st.session_state.batch_active_tab = batch_labels[0]
        
    tabs = st.tabs(batch_labels, default=st.session_state.batch_active_tab)

    with tabs[0]:
        st.session_state.batch_active_tab = batch_labels[0]
        render_upload_tab()
        
    with tabs[1]:
        st.session_state.batch_active_tab = batch_labels[1]
        render_download_reports_tab()


def render_upload_tab():
    """Render the file upload and prediction tab."""
    # Direct multi-file upload (includes sample templates)
    render_multi_file_upload()


def render_multi_file_upload():
    """Render multi-file upload for raw CSV data with left/right layout."""
    st.markdown("### Upload All 7 CSV Files")
    st.markdown("Upload all related data files for comprehensive batch prediction with SHAP explanations.")
    
    # Sample Templates section (moved here)
    with st.expander("📥 Download Sample Templates", expanded=False):
        render_sample_data_section()

    # Create left/right layout
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("##### File Upload")
        # Single file uploader that accepts multiple files
        uploaded_files = st.file_uploader(
            "Browse and select all 7 CSV files",
            type=['csv'],
            accept_multiple_files=True,
            help="Select all 7 required CSV files: application.csv, bureau.csv, bureau_balance.csv, previous_application.csv, credit_card_balance.csv, installments_payments.csv, POS_CASH_balance.csv",
            key="multi_file_upload"
        )

    # Required file names
    required_files = {
        'application': None,
        'bureau': None,
        'bureau_balance': None,
        'previous_application': None,
        'credit_card_balance': None,
        'installments_payments': None,
        'pos_cash_balance': None
    }

    # Map uploaded files to required names - process from newest to oldest to ensure fresh files are prioritized
    if uploaded_files:
        # Reset mapping for each run to avoid accumulation
        for k in required_files:
            required_files[k] = None
            
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name.lower()
            if 'application' in file_name and 'previous' not in file_name:
                required_files['application'] = uploaded_file
            elif 'bureau_balance' in file_name:
                required_files['bureau_balance'] = uploaded_file
            elif 'bureau' in file_name:
                required_files['bureau'] = uploaded_file
            elif 'previous_application' in file_name or 'previous' in file_name:
                required_files['previous_application'] = uploaded_file
            elif 'credit_card' in file_name:
                required_files['credit_card_balance'] = uploaded_file
            elif 'installments' in file_name:
                required_files['installments_payments'] = uploaded_file
            elif 'pos_cash' in file_name or 'pos' in file_name:
                required_files['pos_cash_balance'] = uploaded_file

    # Show upload status
    uploaded_count = sum(1 for f in required_files.values() if f is not None)

    with col_right:
        st.markdown("##### Upload Status")
        if uploaded_files:
            st.write(f"**Files matched:** {uploaded_count}/7")

            # Show which files are matched in a clean format
            for name in ['application', 'bureau', 'bureau_balance', 'previous_application',
                         'credit_card_balance', 'installments_payments', 'pos_cash_balance']:
                if required_files[name]:
                    st.markdown(f"✅ **{name}** - `{required_files[name].name}`")
                else:
                    st.markdown(f"❌ **{name}** - *missing*")
        else:
            st.info("No files uploaded yet. Select all 7 CSV files from your computer.")

    # Process section below
    if uploaded_count == 7:
        st.markdown("---")

        # Preview application data
        try:
            df = pd.read_csv(required_files['application'])
            required_files['application'].seek(0)

            st.success(f"✅ All files loaded! {len(df)} applications found")

            with st.expander("Preview Application Data", expanded=False):
                st.dataframe(df.head(10), width="stretch")

            # Auto-generate batch name (no user input needed)
            batch_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Process button
            if st.button("🚀 Process Batch", width="stretch", type="primary", key="process_multi"):
                process_multi_file_batch(required_files, batch_name)

        except Exception as e:
            st.error(f"Error reading files: {str(e)}")
    elif uploaded_count > 0:
        st.warning(f"Please upload all 7 required CSV files. Missing: {7 - uploaded_count} files")


def process_multi_file_batch(files: dict, batch_name: str):
    """Process batch prediction with all 7 CSV files."""
    with st.spinner("Processing batch..."):
        try:
            logger.info(f"Starting batch prediction: {batch_name}")
            # Reset all file pointers
            for f in files.values():
                f.seek(0)

            # Prepare files for API
            api_files = {
                "application": (files['application'].name, files['application'], "text/csv"),
                "bureau": (files['bureau'].name, files['bureau'], "text/csv"),
                "bureau_balance": (files['bureau_balance'].name, files['bureau_balance'], "text/csv"),
                "previous_application": (files['previous_application'].name, files['previous_application'], "text/csv"),
                "credit_card_balance": (files['credit_card_balance'].name, files['credit_card_balance'], "text/csv"),
                "installments_payments": (files['installments_payments'].name, files['installments_payments'], "text/csv"),
                "pos_cash_balance": (files['pos_cash_balance'].name, files['pos_cash_balance'], "text/csv"),
            }
            
            logger.info(f"Sending POST request to {API_BASE_URL}/batch/predict")
            logger.info(f"Files: {list(api_files.keys())}")

            response = requests.post(
                f"{API_BASE_URL}/batch/predict",
                files=api_files,
                timeout=600  # 10 minutes for large batches with SHAP
            )
            
            logger.info(f"API Response Status: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                display_batch_results(result, batch_name)
            else:
                try:
                    error_detail = response.json().get('detail', 'Unknown error')
                except Exception:
                    error_detail = f"HTTP {response.status_code}: {response.text[:500]}"
                st.error(f"Batch processing failed: {error_detail}")

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Make sure the API server is running.")
            st.info("Start the API with: `uvicorn api.app:app --reload`")
        except ValueError as e:
            st.error(f"Invalid API response: {str(e)}")
            st.info("The API returned a non-JSON response. Check API logs for details.")
        except Exception as e:
            st.error(f"Error: {str(e)}")


def process_single_file_batch(uploaded_file, df: pd.DataFrame, batch_name: str):
    """Process batch prediction with single preprocessed file."""
    with st.spinner(f"Processing {len(df)} applications..."):
        try:
            # Reset file pointer
            uploaded_file.seek(0)

            # Send to API
            files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
            data = {"batch_name": batch_name}

            response = requests.post(
                f"{API_BASE_URL}/batch/predict",
                files=files,
                data=data,
                timeout=300  # 5 minutes timeout
            )

            if response.status_code == 200:
                result = response.json()
                display_batch_results(result, batch_name)
            else:
                error_detail = response.json().get('detail', 'Unknown error')
                st.error(f"Batch processing failed: {error_detail}")

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Make sure the API server is running.")
            st.info("Start the API with: `uvicorn api.app:app --reload`")
        except Exception as e:
            st.error(f"Error: {str(e)}")


def display_batch_results(result: dict, batch_name: str):
    """Display batch processing results."""
    st.success("✅ Batch processed successfully!")

    # Get predictions to calculate stats if not provided
    predictions = result.get('predictions', [])
    
    # Calculate actual statistics from predictions
    n_predictions = len(predictions) if predictions else result.get('n_predictions', result.get('total_processed', 0))
    
    # Calculate processing time
    proc_time = result.get('processing_time_seconds', 0)
    if proc_time == 0 and 'start_time' in result and 'end_time' in result:
        from datetime import datetime
        try:
            start = datetime.fromisoformat(result['start_time'])
            end = datetime.fromisoformat(result['end_time'])
            proc_time = (end - start).total_seconds()
        except:
            pass
    
    # Calculate average probability from predictions if not provided
    avg_prob = result.get('average_probability')
    if (avg_prob is None or avg_prob == 0) and predictions:
        probs = [p.get('probability', 0) for p in predictions if isinstance(p, dict)]
        avg_prob = sum(probs) / len(probs) if probs else 0
    
    # Display summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Processed", n_predictions)
    with col2:
        st.metric("Processing Time", f"{proc_time:.2f}s" if proc_time > 0 else "< 1s")
    with col3:
        st.metric("Avg Probability", f"{avg_prob:.1%}" if avg_prob and avg_prob > 0 else "N/A")

    # Calculate risk distribution from predictions if not provided
    risk_dist = result.get('risk_distribution', {})
    if (not risk_dist or all(v == 0 for v in risk_dist.values())) and predictions:
        risk_dist = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0}
        for p in predictions:
            if isinstance(p, dict):
                risk = p.get('risk_level', 'MEDIUM')
                if risk in ['HIGH', 'CRITICAL']:
                    risk_dist['HIGH'] += 1
                elif risk == 'MEDIUM':
                    risk_dist['MEDIUM'] += 1
                else:
                    risk_dist['LOW'] += 1

    # Risk distribution (3 levels only)
    st.markdown("### Risk Distribution")
    high_count = risk_dist.get('HIGH', 0) + risk_dist.get('CRITICAL', 0)

    col1, col2, col3 = st.columns(3)
    col1.metric("🟢 LOW", risk_dist.get('LOW', 0))
    col2.metric("🟡 MEDIUM", risk_dist.get('MEDIUM', 0))
    col3.metric("🔴 HIGH", high_count)

    # Store results for download
    predictions = result.get('predictions', [])
    if predictions:
        st.session_state.last_batch_results = predictions
        st.session_state.last_batch_id = result.get('batch_id')
        st.session_state.last_batch_name = batch_name

        # Show results table with Default/No Default column
        st.markdown("### Predictions")
        if isinstance(predictions[0], dict):
            results_df = pd.DataFrame(predictions)
        else:
            # Handle pydantic models
            results_df = pd.DataFrame([p.dict() if hasattr(p, 'dict') else p for p in predictions])

        # Add Decision column (Default / No Default)
        if 'prediction' in results_df.columns:
            results_df['decision'] = results_df['prediction'].apply(
                lambda x: "Default" if x == 1 else "No Default"
            )

        display_cols = ['sk_id_curr', 'probability', 'decision', 'risk_level']
        display_cols = [c for c in display_cols if c in results_df.columns]
        st.dataframe(results_df[display_cols], width="stretch")

        st.info("📥 Go to **Download Reports** tab to download Excel file and detailed analysis.")


def render_sample_data_section():
    """Render section for downloading sample data templates."""
    st.markdown("""
    Download sample templates to test batch predictions. These include all linked data files 
    from the original training set.
    """)

    st.markdown("#### Sample Templates")
    st.markdown("Includes a mix of default and non-default cases for testing.")

    if SAMPLES_DIR.exists():
        sample_files = list(SAMPLES_DIR.glob("*.csv"))
        if sample_files:
            # Create and offer direct ZIP download
            zip_buffer = create_zip_from_directory(SAMPLES_DIR)
            st.download_button(
                label="Download Sample Templates (ZIP)",
                data=zip_buffer,
                file_name="sample_templates.zip",
                mime="application/zip",
                key="download_samples_zip",
                width="stretch"
            )
        else:
            st.warning("No sample files found. Run scripts/create_sample_templates.py")
    else:
        st.warning("Sample templates not found. Run scripts/create_sample_templates.py")


def create_zip_from_directory(directory: Path) -> bytes:
    """Create a ZIP file from all CSV files in a directory."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in directory.glob("*.csv"):
            zip_file.write(file_path, file_path.name)
        # Also include README if exists
        readme_path = directory / "README.md"
        if readme_path.exists():
            zip_file.write(readme_path, "README.md")
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


@st.cache_data(ttl=600)
def generate_batch_excel(batch_id: int) -> bytes:
    """Generate Excel file with predictions and SHAP values for a batch."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/batch/history/{batch_id}/download",
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            predictions = result.get('predictions', [])

            if predictions:
                # Create predictions DataFrame
                predictions_df = pd.DataFrame(predictions)

                # Add Decision column
                if 'prediction' in predictions_df.columns:
                    predictions_df['decision'] = predictions_df['prediction'].apply(
                        lambda x: "Default" if x == 1 else "No Default"
                    )

                # Create SHAP features DataFrame
                shap_data = []
                for pred in predictions:
                    sk_id = pred.get('SK_ID_CURR')
                    shap_values = pred.get('shap_values', {})

                    if shap_values:
                        sorted_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:10]

                        row = {'SK_ID_CURR': sk_id}
                        for i, (feature, value) in enumerate(sorted_features, 1):
                            row[f'feature_{i}'] = feature
                            row[f'shap_{i}'] = value
                        shap_data.append(row)

                shap_df = pd.DataFrame(shap_data) if shap_data else pd.DataFrame()

                # Create Excel file with two sheets
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    # Sheet 1: Predictions with Decision column
                    pred_cols = ['SK_ID_CURR', 'probability', 'decision', 'risk_level']
                    pred_cols = [c for c in pred_cols if c in predictions_df.columns]
                    predictions_df[pred_cols].to_excel(writer, index=False, sheet_name='Predictions')

                    # Sheet 2: SHAP Top 10 Features
                    if not shap_df.empty:
                        shap_df.to_excel(writer, index=False, sheet_name='SHAP_Top10_Features')

                excel_buffer.seek(0)
                return excel_buffer.getvalue()

        return None
    except Exception:
        return None


@st.cache_data(ttl=600)
def generate_batch_html_report(batch_id: int, batch_name: str) -> str:
    """Generate HTML report for a batch."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/batch/history/{batch_id}/download",
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            predictions = result.get('predictions', [])

            if predictions:
                return generate_detailed_html_report(predictions, batch_name)

        return None
    except Exception:
        return None


def render_download_reports_tab():
    """Render the download reports tab (formerly Batch History)."""
    st.markdown("### Download Reports")

    try:
        response = requests.get(f"{API_BASE_URL}/batch/history", timeout=30)

        if response.status_code == 200:
            data = response.json()

            if data.get('success') and data.get('batches'):
                batches = data['batches']

                # Sort by ID descending (most recent first)
                batches = sorted(batches, key=lambda x: x['id'], reverse=True)

                st.write(f"**Total Batches:** {data.get('count', 0)}")

                for batch in batches:
                    status_text = "[OK]" if batch['status'] == 'completed' else "[...]" if batch['status'] == 'processing' else "[X]"
                    
                    # Custom formatting for Batch Name: Batch_ID Date Time
                    batch_id = batch.get('id', 'Unknown')
                    created_at_str = batch.get('created_at')
                    if created_at_str:
                        try:
                            # Handle potential Z or +offset in isoformat
                            dt_str = created_at_str.replace('Z', '+00:00')
                            dt = datetime.fromisoformat(dt_str)
                            formatted_dt = dt.strftime('%Y-%m-%d %H:%M:%S')
                            batch_name_display = f"Batch_{batch_id} {formatted_dt}"
                        except Exception:
                            batch_name_display = batch.get('batch_name') or f"Batch_{batch_id}"
                    else:
                        batch_name_display = batch.get('batch_name') or f"Batch_{batch_id}"

                    with st.expander(f"{status_text} {batch_name_display}", expanded=False):
                        # Batch details
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.write(f"**Status:** {batch.get('status', 'unknown')}")
                            st.write(f"**Total Applications:** {batch.get('total_applications', 0)}")

                        with col2:
                            st.write(f"**Processed:** {batch.get('processed_applications', 0)}")
                            proc_time_val = batch.get('processing_time_seconds') or 0
                            st.write(f"**Processing Time:** {proc_time_val:.2f}s")

                        with col3:
                            avg_prob_val = batch.get('avg_probability') or 0
                            st.write(f"**Avg Probability:** {avg_prob_val:.2%}")
                            risk_dist = batch.get('risk_distribution', {})
                            st.write(f"**High Risk:** {risk_dist.get('HIGH', 0) + risk_dist.get('CRITICAL', 0)}")

                        st.markdown("---")

                        # Download buttons in one row
                        col_excel, col_html = st.columns(2)

                        with col_excel:
                            st.markdown("##### Excel Results")
                            st.markdown("*Predictions + SHAP values*")
                            excel_data = generate_batch_excel(batch['id'])
                            if excel_data:
                                # Use user-friendly filename
                                st.download_button(
                                    label="Download Excel",
                                    data=excel_data,
                                    file_name=f"{batch_name_display}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key=f"excel_{batch['id']}",
                                    width="stretch"
                                )

                        with col_html:
                            st.markdown("##### Detail Analysis")
                            st.markdown("*HTML report with SHAP plots*")
                            # Generate HTML and provide download button
                            html_report = generate_batch_html_report(batch['id'], batch_name_display)
                            if html_report:
                                # Use user-friendly filename
                                st.download_button(
                                    label="Download HTML Report",
                                    data=html_report,
                                    file_name=f"{batch_name_display}.html",
                                    mime="text/html",
                                    key=f"html_{batch['id']}",
                                    width="stretch"
                                )
            else:
                st.info("No batch history found. Process your first batch in the **Upload & Predict** tab!")

        else:
            st.error(f"Failed to fetch history: {response.status_code}")

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Make sure the API server is running.")
    except Exception as e:
        st.error(f"Error: {str(e)}")


def generate_detailed_html_report(predictions: list, batch_name: str) -> str:
    """Generate detailed HTML report with collapsible sections and waterfall plots."""
    # Summary statistics
    total = len(predictions)
    high_risk_count = sum(1 for p in predictions if p.get('risk_level') in ['HIGH', 'CRITICAL'])
    
    # Calculate average probability safely
    probs = [p.get('probability') for p in predictions]
    probs = [p for p in probs if p is not None]
    avg_prob = np.mean(probs) if probs else 0

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Default Risk Detail Analysis - {batch_name}</title>
        <style>
            * {{ box-sizing: border-box; }}
            body {{ 
                font-family: 'Segoe UI', Arial, sans-serif; 
                margin: 0;
                padding: 20px 40px;
                background: #f5f7fa;
                color: #333;
            }}
            .header {{
                background: linear-gradient(135deg, #1f77b4, #2ca02c);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
            }}
            h1 {{ margin: 0 0 10px 0; }}
            .summary {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .summary-card {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                text-align: center;
            }}
            .summary-card-wide {{
                grid-column: span 1;
            }}
            .summary-value {{
                font-size: 32px;
                font-weight: bold;
                color: #1f77b4;
            }}
            .summary-label {{
                font-size: 14px;
                color: #666;
                margin-top: 5px;
            }}
            .collapsible {{
                background: white;
                border: none;
                border-radius: 10px;
                margin-bottom: 15px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            .collapsible-header {{
                background: #f8f9fa;
                padding: 15px 20px;
                cursor: pointer;
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid #eee;
            }}
            .collapsible-header:hover {{ background: #e9ecef; }}
            .collapsible-content {{
                padding: 20px;
                display: none;
            }}
            .collapsible.active .collapsible-content {{ display: block; }}
            .collapsible-header .toggle {{
                font-size: 18px;
                transition: transform 0.3s;
            }}
            .collapsible.active .toggle {{ transform: rotate(90deg); }}
            .risk-badge {{
                padding: 5px 12px;
                border-radius: 20px;
                font-weight: bold;
                font-size: 12px;
            }}
            .risk-low {{ background: #d4edda; color: #155724; }}
            .risk-medium {{ background: #fff3cd; color: #856404; }}
            .risk-high {{ background: #ffe5d0; color: #d35400; }}
            .risk-critical {{ background: #f8d7da; color: #721c24; }}
            .details-grid {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
            }}
            .detail-item {{ margin-bottom: 10px; }}
            .detail-label {{ color: #666; font-size: 12px; }}
            .detail-value {{ font-weight: bold; font-size: 16px; }}
            .waterfall-container {{
                margin-top: 20px;
                padding: 15px;
                background: #f8f9fa;
                border-radius: 8px;
            }}
            .waterfall-title {{ 
                font-weight: bold; 
                margin-bottom: 15px;
                color: #333;
            }}
            .waterfall-bar {{
                display: flex;
                align-items: center;
                margin-bottom: 8px;
                font-size: 13px;
            }}
            .waterfall-label {{
                width: 200px;
                text-align: right;
                padding-right: 10px;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }}
            .waterfall-bar-container {{
                flex: 1;
                height: 24px;
                background: #e9ecef;
                position: relative;
                border-radius: 4px;
                overflow: hidden;
            }}
            .waterfall-bar-fill {{
                height: 100%;
                position: absolute;
                border-radius: 4px;
            }}
            .waterfall-bar-positive {{ background: #dc3545; }}
            .waterfall-bar-negative {{ background: #28a745; }}
            .waterfall-value {{
                width: 80px;
                text-align: right;
                padding-left: 10px;
                font-weight: bold;
            }}
            .positive {{ color: #dc3545; }}
            .negative {{ color: #28a745; }}
            .legend {{
                display: flex;
                gap: 20px;
                margin-top: 15px;
                font-size: 12px;
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                gap: 5px;
            }}
            .legend-color {{
                width: 16px;
                height: 16px;
                border-radius: 3px;
            }}
            .expand-all {{
                background: #1f77b4;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                margin-bottom: 20px;
            }}
            .expand-all:hover {{ background: #1565a0; }}
            .download-btn {{
                background: #28a745;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                margin-left: 10px;
                text-decoration: none;
                display: inline-block;
            }}
            .download-btn:hover {{ background: #218838; }}
            .toolbar {{
                display: flex;
                align-items: center;
                gap: 10px;
                margin-bottom: 20px;
            }}
        </style>
        <script>
            function toggleSection(element) {{
                element.parentElement.classList.toggle('active');
            }}
            function toggleAll() {{
                const sections = document.querySelectorAll('.collapsible');
                const allExpanded = Array.from(sections).every(s => s.classList.contains('active'));
                sections.forEach(s => {{
                    if (allExpanded) {{
                        s.classList.remove('active');
                    }} else {{
                        s.classList.add('active');
                    }}
                }});
            }}
            function downloadReport() {{
                const blob = new Blob([document.documentElement.outerHTML], {{type: 'text/html'}});
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = '{batch_name}.html';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }}
            function copyUrl() {{
                navigator.clipboard.writeText(window.location.href).then(() => {{
                    alert('URL copied to clipboard!');
                }});
            }}
        </script>
    </head>
    <body>
        <div class="header">
            <h1>Default Risk Detail Analysis</h1>
            <p>Batch: {batch_name} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="summary">
            <div class="summary-card">
                <div class="summary-value">{total}</div>
                <div class="summary-label">Total Applications</div>
            </div>
            <div class="summary-card">
                <div class="summary-value">{avg_prob:.1%}</div>
                <div class="summary-label">Avg Default Probability</div>
            </div>
            <div class="summary-card">
                <div class="summary-value">{high_risk_count}</div>
                <div class="summary-label">High Risk Applications</div>
            </div>
            <div class="summary-card">
                <div class="summary-value">{total - high_risk_count}</div>
                <div class="summary-label">Low/Medium Risk</div>
            </div>
        </div>
        
        <div class="toolbar">
            <button class="expand-all" onclick="toggleAll()">Expand/Collapse All</button>
            <button class="download-btn" onclick="downloadReport()">Download HTML</button>
            <button class="download-btn" onclick="copyUrl()" style="background: #6c757d;">Copy URL</button>
        </div>
        
        <h2>Application Details</h2>
    """

    # Generate collapsible section for each application
    for pred in predictions:
        sk_id = pred.get('SK_ID_CURR') or pred.get('sk_id_curr') or 'Unknown'
        probability = pred.get('probability') or 0
        risk_level = pred.get('risk_level', 'UNKNOWN')
        shap_values = pred.get('shap_values', {})
        top_features = pred.get('top_features', [])

        # Risk badge class
        risk_class = f"risk-{risk_level.lower() if risk_level else 'unknown'}"

        html += f"""
        <div class="collapsible">
            <div class="collapsible-header" onclick="toggleSection(this)">
                <div>
                    <strong>SK_ID_CURR: {sk_id}</strong>
                    <span class="risk-badge {risk_class}">{risk_level}</span>
                </div>
                <div style="display: flex; align-items: center; gap: 15px;">
                    <span>Probability: <strong>{probability:.1%}</strong></span>
                    <span class="toggle">&#9654;</span>
                </div>
            </div>
            <div class="collapsible-content">
                <div class="details-grid">
                    <div class="detail-item">
                        <div class="detail-label">Default Probability</div>
                        <div class="detail-value">{probability:.2%}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Risk Level</div>
                        <div class="detail-value">{risk_level}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Prediction</div>
                        <div class="detail-value">{'Default' if pred.get('prediction', 0) == 1 else 'No Default'}</div>
                    </div>
                </div>
        """

        # Generate waterfall plot for SHAP values
        if shap_values:
            expected_val = pred.get('expected_value', 0)
            html += generate_waterfall_html(shap_values, top_features, expected_val)
        else:
            html += """
                <div class="waterfall-container">
                    <p style="color: #666;">SHAP values not available for this prediction.</p>
                </div>
            """

        html += """
            </div>
        </div>
        """

    html += """
    </body>
    </html>
    """

    return html


def _format_feature_label(feature: str, feature_type: str, missing: bool) -> str:
    """Format feature label with type and missing indicator."""
    prefix = "[M] " if missing else ""
    return f"{prefix}{feature} ({feature_type})"


def _format_value(value) -> str:
    if value is None:
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
    except Exception:
        pass

    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        if abs(value) >= 1:
            return f"{value:,.2f}"
        return f"{value:.4f}"
    return str(value)


def generate_waterfall_html(shap_values: dict, top_features: list | None = None, expected_value: float = 0) -> str:
    """Generate a TRUE cumulative waterfall HTML chart with logit-to-prob transformation if needed."""
    
    # Sigmoid function for probability conversion
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Determine if we are in log-odds space (logits) or probability space
    # If base value > 1 or < -1, it's definitely logits.
    # If expected_value is e.g. 0.1, it could be either (but probability 10% is common).
    # We check if the sum of base + shap is outside [0, 1]
    is_logit = False
    if expected_value > 1 or expected_value < 0 or abs(sum(shap_values.values())) > 1:
        is_logit = True

    # Build enriched feature list
    enriched = []
    if top_features:
        for item in top_features:
            enriched.append({
                'feature': item.get('feature'),
                'shap_value': item.get('shap_value', 0),
                'value': item.get('value')
            })
    else:
        for feature, value in shap_values.items():
            enriched.append({
                'feature': feature,
                'shap_value': value,
                'value': None
            })

    # Sort and cap
    enriched = sorted(enriched, key=lambda x: abs(x['shap_value']), reverse=True)[:10]
    
    # Calculate cumulative points for scaling
    # If logit, we calculate points in probability space for the plot
    if is_logit:
        current_logit = expected_value
        points = [sigmoid(current_logit)]
        for item in enriched:
            current_logit += item['shap_value']
            points.append(sigmoid(current_logit))
        
        top_set = {e['feature'] for e in enriched}
        others_sum = sum(v for k, v in shap_values.items() if k not in top_set)
        if others_sum != 0:
            current_logit += others_sum
            points.append(sigmoid(current_logit))
    else:
        current_val = expected_value
        points = [current_val]
        for item in enriched:
            current_val += item['shap_value']
            points.append(current_val)
        
        top_set = {e['feature'] for e in enriched}
        others_sum = sum(v for k, v in shap_values.items() if k not in top_set)
        if others_sum != 0:
            current_val += others_sum
            points.append(current_val)
        
    min_p, max_p = min(points), max(points)
    total_range = max(abs(max_p - min_p), 0.0001)
    
    padding = total_range * 0.1
    min_p -= padding
    max_p += padding
    total_range = max_p - min_p

    html = """
        <div class="waterfall-container">
            <div class="waterfall-title">Cumulative Waterfall (SHAP)</div>
            <div style="margin-top: 15px;">
    """

    # Add Base Value row
    base_val_display = sigmoid(expected_value) if is_logit else expected_value
    base_pos = (base_val_display - min_p) / total_range * 100
    html += f"""
        <div class="waterfall-bar" style="background: #eee; border-radius: 4px; margin-bottom: 12px; padding: 5px;">
            <div class="waterfall-label"><strong>[Base Probability]</strong></div>
            <div class="waterfall-bar-container" style="background: transparent;">
                <div style="position: absolute; left: 0; width: {base_pos}%; height: 100%; border-right: 2px solid #666;"></div>
            </div>
            <div class="waterfall-value">{base_val_display:.2%}</div>
        </div>
    """

    # Running sum for positioning
    running_total_logit = expected_value if is_logit else None
    running_total_prob = expected_value if not is_logit else sigmoid(expected_value)
    
    for item in enriched:
        feature = item['feature']
        val = item['shap_value']
        
        start_p = (running_total_prob - min_p) / total_range * 100
        
        if is_logit:
            running_total_logit += val
            running_total_prob = sigmoid(running_total_logit)
            # For display, show probability impact (approx)
            display_impact = sigmoid(running_total_logit) - sigmoid(running_total_logit - val)
        else:
            running_total_prob += val
            display_impact = val
            
        end_p = (running_total_prob - min_p) / total_range * 100
        
        left = min(start_p, end_p)
        width = abs(start_p - end_p)
        
        bar_class = "waterfall-bar-positive" if display_impact > 0 else "waterfall-bar-negative"
        value_class = "positive" if display_impact > 0 else "negative"

        html += f"""
            <div class="waterfall-bar">
                <div class="waterfall-label">{feature.replace('_', ' ')}</div>
                <div class="waterfall-bar-container">
                    <div class="waterfall-bar-fill {bar_class}" style="left: {left}%; width: {width}%;"></div>
                </div>
                <div class="waterfall-value {value_class}">{display_impact:+.2%}</div>
            </div>
        """

    # Final Prediction row
    html += f"""
        <div class="waterfall-bar" style="border-top: 2px solid #333; margin-top: 10px; padding-top: 10px;">
            <div class="waterfall-label"><strong>Final Probability</strong></div>
            <div class="waterfall-bar-container" style="background: transparent;">
                <div style="position: absolute; left: 0; width: {(running_total_prob - min_p)/total_range*100}%; height: 100%; background: #1f77b4; opacity: 0.3; border-radius: 4px;"></div>
            </div>
            <div class="waterfall-value" style="font-weight: bold; color: #1f77b4;">{running_total_prob:.2%}</div>
        </div>
    """

    html += "</div></div>"
    return html

