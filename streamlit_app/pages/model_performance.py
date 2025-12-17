"""Model Performance Page - Restructured Layout

Landing Page Overview (visible without scrolling):
- Left Side: Threshold Selection + Key Metrics  
- Right Side: Confusion Matrix

Below:
- Threshold Analysis (two graphs)
- One collapsible section with all explanations consolidated
"""

import sys
from pathlib import Path
import logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import requests

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

from src.config import CONFIG
from streamlit_app.config import API_BASE_URL

METRICS_ENDPOINT = f"{API_BASE_URL}/metrics/precomputed"


@st.cache_data(ttl=3600)
def fetch_metrics_from_api_cached():
    """Fetch precomputed metrics from API - FAST HTTP call only.
    
    This function is cached. It only handles the happy path.
    Exceptions raised here will NOT be cached, allowing retries.
    """
    response = requests.get(METRICS_ENDPOINT, timeout=30)
    response.raise_for_status()
    data = response.json()
    logger.info(f"Fetched metrics for {len(data['metrics'])} thresholds from API")
    return {
        'all_metrics': data['metrics'],
        'metrics_df': pd.DataFrame(data['metrics_df']),
        'optimal_threshold': data['optimal_threshold'],
        'data_count': data['data_count'],
        'thresholds': data['thresholds']
    }


def fetch_metrics_from_api():
    """Wrapper to handle API fetch with error handling.
    
    This function is NOT cached, so errors are not persisted.
    It calls the cached function for the heavy lifting.
    """
    try:
        return fetch_metrics_from_api_cached()
    except requests.exceptions.ConnectionError:
        return {'error': 'connection', 'detail': 'Cannot connect to API server'}
    except requests.exceptions.Timeout:
        return {'error': 'timeout', 'detail': 'API request timed out'}
    except requests.exceptions.HTTPError as e:
        return {'error': 'http', 'detail': str(e), 'response': e.response.json() if e.response else None}
    except Exception as e:
        logger.error(f"API fetch error: {e}")
        return {'error': 'unknown', 'detail': str(e)}


def plot_confusion_matrix(cm):
    """Plot confusion matrix with row and column percentages."""
    cm_array = np.array(cm)
    
    # Calculate row-based percentages (percentage of actual class)
    row_sums = cm_array.sum(axis=1)[:, np.newaxis]
    cm_row_pct = np.divide(cm_array.astype('float'), row_sums,
                          out=np.zeros_like(cm_array, dtype=float), where=row_sums!=0)
    
    # Calculate column-based percentages (percentage of predicted class)
    col_sums = cm_array.sum(axis=0)[np.newaxis, :]
    cm_col_pct = np.divide(cm_array.astype('float'), col_sums,
                          out=np.zeros_like(cm_array, dtype=float), where=col_sums!=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_row_pct, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)

    for i in range(2):
        for j in range(2):
            # Display both row % and column %
            text = ax.text(j, i, f'{cm_row_pct[i,j]:.1%}\n{cm_col_pct[i,j]:.1%}',
                          ha="center", va="center", 
                          color="black" if cm_row_pct[i,j] < 0.5 else "white",
                          fontsize=11, fontweight='bold')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['No Default', 'Default'])
    ax.set_yticklabels(['No Default', 'Default'])
    ax.set_ylabel('Actual', fontsize=11)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_title('Confusion Matrix (Row% / Col%)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def create_threshold_analysis_plots(metrics_df, cost_optimal, current_threshold):
    """Create threshold analysis plots."""
    fig_thresh, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left plot: Precision/Recall/F1
    axes[0].plot(metrics_df['Threshold'], metrics_df['Recall'], label='Recall', marker='o', linewidth=2)
    axes[0].plot(metrics_df['Threshold'], metrics_df['Precision'], label='Precision', marker='s', linewidth=2)
    axes[0].plot(metrics_df['Threshold'], metrics_df['F1 Score'], label='F1 Score', marker='^', linewidth=2)
    axes[0].axvline(current_threshold, color='red', linestyle='--', linewidth=2, label=f'Current ({current_threshold:.2f})')
    axes[0].axvline(cost_optimal, color='green', linestyle=':', linewidth=2, label=f'Optimal ({cost_optimal:.2f})')
    axes[0].set_xlabel('Threshold', fontsize=10)
    axes[0].set_ylabel('Score', fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].set_title('Precision, Recall, F1 vs Threshold', fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)

    # Right plot: Business Cost
    axes[1].plot(metrics_df['Threshold'], metrics_df['Business Cost'], marker='o', color='red', linewidth=2)
    axes[1].axvline(current_threshold, color='blue', linestyle='--', linewidth=2, label=f'Current ({current_threshold:.2f})')
    axes[1].axvline(cost_optimal, color='green', linestyle=':', linewidth=2, label=f'Optimal ({cost_optimal:.2f})')
    axes[1].set_xlabel('Threshold', fontsize=10)
    axes[1].set_ylabel('Business Cost', fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].set_title('Business Cost vs Threshold', fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 1)

    plt.tight_layout()
    return fig_thresh


def render_model_performance():
    """Render the model performance page - restructured layout."""
    logger.info("Rendering Model Performance page")
    
    st.title("📈 Model Performance")
    
    # Fetch precomputed metrics from API
    api_data = fetch_metrics_from_api()

    if api_data is None or 'error' in api_data:
        if api_data and 'error' in api_data:
            error_type = api_data.get('error')
            if error_type == 'connection':
                st.error("❌ Cannot connect to API. Please ensure the API server is running.")
                st.code("poetry run uvicorn api.app:app --reload --port 8000", language="bash")
            elif error_type == 'http':
                response_data = api_data.get('response')
                if response_data:
                    detail = response_data.get('detail', 'Unknown error')
                else:
                    detail = api_data.get('detail', 'Unknown error')
                st.warning(f"⚠️ API Error: {detail}")
                st.info("The metrics endpoint returned an error. This usually means prediction data needs to be generated first. Try running a batch prediction to generate the metrics data.")
            else:
                st.error(f"❌ Error: {api_data.get('detail', 'Unknown error')}")
        else:
            st.error("❌ Cannot connect to API. Please ensure the API server is running.")
            st.code("poetry run uvicorn api.app:app --reload --port 8000", language="bash")
        return

    # Extract data
    all_metrics = api_data['all_metrics']
    metrics_df = api_data['metrics_df']
    cost_optimal = api_data['optimal_threshold']
    data_count = api_data['data_count']

    st.caption(f"✅ {data_count:,} predictions loaded from API")

    # Get config
    cost_fn = CONFIG['business']['cost_fn']
    cost_fp = CONFIG['business']['cost_fp']
    f_beta = CONFIG['business']['f_beta']

    # ==========================================================================
    # SECTION 1: LANDING PAGE - NO SCROLLING REQUIRED
    # Left: Threshold + Metrics | Right: Confusion Matrix
    # ==========================================================================
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        # Threshold Selection
        st.subheader("🎯 Threshold Selection")
        
        threshold = st.slider(
            "Decision Threshold",
            0.01, 0.99, 0.33, 0.01,
            help="Probability cutoff for classifying as Default",
            key="threshold_slider"
        )
        
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            st.metric("Current", f"{threshold:.2f}")
        with col_opt2:
            st.metric("Optimal", f"{cost_optimal:.2f}", 
                     delta=f"{threshold - cost_optimal:+.2f}" if abs(threshold - cost_optimal) > 0.01 else None)

        # Find closest threshold in API data
        threshold_key = str(round(threshold, 2))
        if threshold_key not in all_metrics:
            available_thresholds = [float(k) for k in all_metrics.keys()]
            closest_threshold = min(available_thresholds, key=lambda x: abs(x - threshold))
            threshold_key = str(round(closest_threshold, 2))

        # INSTANT lookup
        metrics = all_metrics[threshold_key]
        
        st.markdown("---")
        
        # Key Performance Metrics
        st.subheader("📊 Key Metrics")
        
        col1, col2 = st.columns(2)
        col1.metric("💰 Business Cost", f"{metrics['cost']:,}")
        col2.metric("🎯 Recall", f"{metrics['recall']:.1%}")
        
        col3, col4 = st.columns(2)
        col3.metric("✅ Precision", f"{metrics['precision']:.1%}")
        col4.metric(f"📈 F{f_beta} Score", f"{metrics['fbeta']:.3f}")
        
        col5, col6 = st.columns(2)
        col5.metric("📏 Accuracy", f"{metrics['accuracy']:.1%}")
        col6.metric("⚖️ F1 Score", f"{metrics['f1']:.3f}")

    with col_right:
        # Confusion Matrix
        st.subheader(f"📈 Confusion Matrix (T={threshold:.2f})")
        
        cm = metrics['cm']
        fig = plot_confusion_matrix(cm)
        st.pyplot(fig)
        plt.close(fig)
        
        # Breakdown
        tn, fp, fn, tp = metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp']
        col_a, col_b = st.columns(2)
        with col_a:
            st.write(f"✔️ TN: {tn:,}")
            st.write(f"⚠️ FP: {fp:,}")
        with col_b:
            st.write(f"❌ FN: {fn:,}")
            st.write(f"✔️ TP: {tp:,}")

    # ==========================================================================
    # SECTION 2: THRESHOLD ANALYSIS (below landing page)
    # ==========================================================================
    st.markdown("---")
    st.header("🔍 Threshold Analysis")
    
    fig_thresh = create_threshold_analysis_plots(metrics_df, cost_optimal, threshold)
    st.pyplot(fig_thresh)
    plt.close(fig_thresh)
    
    st.info(f"💡 **Cost-Optimal Threshold:** {cost_optimal:.2f} (minimizes business cost)")

    # ==========================================================================
    # SECTION 3: CONSOLIDATED EXPLANATIONS (One Collapsible)
    # ==========================================================================
    st.markdown("---")
    
    with st.expander("📚 **Detailed Guidance & Explanations**", expanded=False):
        
        st.markdown("### 🎯 How to Select the Threshold")
        st.markdown("""
        **Decision Threshold** determines classification:
        - If probability ≥ threshold → **Default** (reject loan)
        - If probability < threshold → **No Default** (approve loan)
        
        **Trade-offs:**
        | Lower Threshold (0.2) | Higher Threshold (0.5) |
        |:---------------------|:----------------------|
        | More risky flags | Fewer risky flags |
        | ⬆️ Higher Recall | ⬇️ Lower Recall |
        | ⬇️ Lower Precision | ⬆️ Higher Precision |
        
        **Recommendation:** Use **0.30-0.35** for balanced risk management.
        """)
        
        st.markdown("---")
        st.markdown("### 📈 How to Read the Confusion Matrix")
        st.markdown("""
        ```
                           PREDICTED
                    No Default  |  Default
                   ─────────────┼──────────
        ACTUAL     |    TN     |    FP    |
        No Default |  (correct)|  (error) |
                   ─────────────┼──────────
        ACTUAL     |    FN     |    TP    |
        Default    |  (error) |  (correct)|
        ```
        
        | Quadrant | Meaning | Cost |
        |:---------|:--------|:-----|
        | **TN** | Good client approved | ✅ Profit |
        | **FP** | Good client rejected | ⚠️ Opportunity cost |
        | **FN** | Bad client approved | ❌ **Major loss** |
        | **TP** | Bad client rejected | ✅ Loss prevented |
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Understanding the Metrics")
        st.markdown(f"""
        | Metric | Formula | Target |
        |:-------|:--------|:-------|
        | **Recall** | TP / (TP + FN) | **Higher** (catch defaults) |
        | **Precision** | TP / (TP + FP) | Higher = fewer false alarms |
        | **F1 Score** | 2×(P×R)/(P+R) | Balance metric |
        | **F{f_beta} Score** | Weighted F-score | **Primary metric** (β={f_beta}) |
        
        **Business Cost** = ({cost_fn} × FN) + ({cost_fp} × FP)
        
        Current: ({cost_fn} × {fn:,}) + ({cost_fp} × {fp:,}) = **{metrics['cost']:,}**
        """)
        
        st.markdown("---")
        st.markdown("### 📋 Detailed Metrics by Threshold")
        st.dataframe(
            metrics_df.style.format({
                'Threshold': '{:.2f}',
                'Recall': '{:.1%}',
                'Precision': '{:.1%}',
                'F1 Score': '{:.3f}',
                'Business Cost': '{:,.0f}',
                'False Negatives': '{:,.0f}',
                'False Positives': '{:,.0f}'
            }),
            use_container_width=True,
            height=300
        )
