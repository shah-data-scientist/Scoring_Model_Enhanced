"""Single Prediction Page for Credit Scoring Dashboard.

This page allows users to:
- Enter client application details
- Get individual credit risk predictions
- View SHAP explanations
"""

import json
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_app.config import API_BASE_URL


def load_feature_config():
    """Load the critical raw features configuration."""
    config_path = PROJECT_ROOT / "config" / "critical_raw_features.json"
    try:
        with open(config_path) as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Feature config not found: {config_path}")
        return {"features": []}


def render_single_prediction():
    """Render the single prediction interface."""
    # Load feature configuration
    feature_config = load_feature_config()
    features = feature_config.get('features', [])

    if not features:
        st.error("No features configured. Please check the configuration.")
        return

    st.markdown("### Enter Application Details")

    # Create two columns for input
    col1, col2 = st.columns(2)

    # Dictionary to store input values
    input_values = {}

    # Default values for common features
    defaults = {
        "SK_ID_CURR": 100001,
        "AMT_INCOME_TOTAL": 150000.0,
        "AMT_CREDIT": 500000.0,
        "AMT_ANNUITY": 25000.0,
        "AMT_GOODS_PRICE": 450000.0,
        "DAYS_BIRTH": -12000,  # ~33 years old
        "DAYS_EMPLOYED": -2000,  # ~5.5 years
        "CNT_FAM_MEMBERS": 2.0,
        "EXT_SOURCE_1": 0.5,
        "EXT_SOURCE_2": 0.5,
        "EXT_SOURCE_3": 0.5,
    }

    # Render input fields
    for i, feature in enumerate(features[:20]):  # Show first 20 for usability
        col = col1 if i % 2 == 0 else col2

        with col:
            default_val = defaults.get(feature, 0.0)

            # Handle different feature types
            if feature == "SK_ID_CURR":
                input_values[feature] = st.number_input(
                    feature,
                    min_value=100000,
                    max_value=999999,
                    value=int(default_val),
                    step=1,
                    help="Unique application ID"
                )
            elif "DAYS" in feature:
                input_values[feature] = st.number_input(
                    feature,
                    min_value=-30000,
                    max_value=0,
                    value=int(default_val),
                    step=1,
                    help="Days relative to application date (negative = past)"
                )
            elif "AMT" in feature:
                input_values[feature] = st.number_input(
                    feature,
                    min_value=0.0,
                    max_value=10000000.0,
                    value=float(default_val),
                    step=1000.0,
                    help="Amount in currency"
                )
            elif "EXT_SOURCE" in feature:
                input_values[feature] = st.slider(
                    feature,
                    min_value=0.0,
                    max_value=1.0,
                    value=float(default_val),
                    step=0.01,
                    help="External data source score (0-1)"
                )
            else:
                input_values[feature] = st.number_input(
                    feature,
                    value=float(default_val),
                    help=f"Feature: {feature}"
                )

    # Add remaining features with default values (hidden)
    for feature in features[20:]:
        input_values[feature] = defaults.get(feature, 0.0)

    st.markdown("---")

    # Prediction button
    if st.button("🔮 Get Prediction", use_container_width=True, type="primary"):
        with st.spinner("Getting prediction..."):
            try:
                # Prepare request
                payload = {"features": input_values}

                response = requests.post(
                    f"{API_BASE_URL}/predict",
                    json=payload,
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()

                    # Display results
                    st.success("Prediction completed!")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        prob = result.get('probability', 0)
                        st.metric(
                            "Default Probability",
                            f"{prob:.1%}",
                            help="Probability of loan default"
                        )

                    with col2:
                        risk_level = result.get('risk_level', 'UNKNOWN')
                        risk_colors = {
                            'LOW': '🟢',
                            'MEDIUM': '🟡',
                            'HIGH': '🟠',
                            'CRITICAL': '🔴'
                        }
                        st.metric(
                            "Risk Level",
                            f"{risk_colors.get(risk_level, '⚪')} {risk_level}",
                            help="Risk category based on probability thresholds"
                        )

                    with col3:
                        prediction = result.get('prediction', 0)
                        st.metric(
                            "Prediction",
                            "⚠️ DEFAULT" if prediction == 1 else "✅ NO DEFAULT",
                            help="Binary prediction (0=No Default, 1=Default)"
                        )

                    # SHAP explanation if available
                    if 'shap_values' in result:
                        st.markdown("### 📊 Feature Importance (SHAP)")

                        shap_values = result.get('shap_values', {})
                        if shap_values:
                            # Create DataFrame for display
                            shap_df = pd.DataFrame([
                                {'Feature': k, 'SHAP Value': v}
                                for k, v in sorted(
                                    shap_values.items(),
                                    key=lambda x: abs(x[1]),
                                    reverse=True
                                )[:15]
                            ])

                            st.bar_chart(shap_df.set_index('Feature'))

                else:
                    error_detail = response.json().get('detail', 'Unknown error')
                    st.error(f"Prediction failed: {error_detail}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure the API server is running.")
                st.info("Start the API with: `uvicorn api.app:app --reload`")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Show sample data option
    with st.expander("📝 Use Sample Data"):
        st.markdown("Click below to load sample application data:")

        if st.button("Load Low Risk Sample"):
            st.session_state.sample_type = "low_risk"
            st.rerun()

        if st.button("Load High Risk Sample"):
            st.session_state.sample_type = "high_risk"
            st.rerun()
