"""Model Performance Page - Optimized with Precomputation.

This version precomputes all metrics and plots at startup,
then instantly retrieves them based on the selected threshold.
No recalculation happens when the slider moves.
"""

import sys
from pathlib import Path
import logging

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for faster rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, fbeta_score

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

from src.config import CONFIG

# Configuration
RESULTS_DIR = PROJECT_ROOT / CONFIG['paths']['results']
STATIC_PREDICTIONS_PATH = RESULTS_DIR / 'static_model_predictions.parquet'

# Precomputed thresholds (every 1% from 0.01 to 0.99)
THRESHOLDS = [round(t, 2) for t in np.arange(0.01, 1.0, 0.01)]


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_predictions():
    """Load predictions from parquet (fast) or CSV (slower)."""
    if STATIC_PREDICTIONS_PATH.exists():
        try:
            df = pd.read_parquet(STATIC_PREDICTIONS_PATH)
            return df['TARGET'].values, df['PROBABILITY'].values
        except Exception:
            pass
    
    pred_path = RESULTS_DIR / 'train_predictions.csv'
    if not pred_path.exists():
        return None, None
    
    try:
        df = pd.read_csv(
            pred_path,
            usecols=['TARGET', 'PROBABILITY'],
            dtype={'TARGET': 'int8', 'PROBABILITY': 'float32'}
        )
        # Save as parquet for faster future loads
        df.to_parquet(STATIC_PREDICTIONS_PATH, index=False)
        return df['TARGET'].values, df['PROBABILITY'].values
    except Exception as e:
        logger.error(f"Error loading predictions: {e}")
        return None, None


@st.cache_data(ttl=3600)
def precompute_all_metrics(_y_true, _y_proba, cost_fn, cost_fp, f_beta):
    """Precompute metrics for ALL threshold values at once.
    
    This runs ONCE and caches the results for all thresholds.
    Subsequent slider changes just look up precomputed values.
    """
    results = {}
    
    for threshold in THRESHOLDS:
        y_pred = (_y_proba >= threshold).astype(int)
        cm = confusion_matrix(_y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = f1_score(_y_true, y_pred)
        fbeta_val = fbeta_score(_y_true, y_pred, beta=f_beta)
        accuracy = accuracy_score(_y_true, y_pred)
        cost = cost_fn * fn + cost_fp * fp
        
        results[threshold] = {
            'cm': cm,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
            'recall': recall,
            'precision': precision,
            'f1': f1,
            'fbeta': fbeta_val,
            'accuracy': accuracy,
            'cost': cost
        }
    
    return results


@st.cache_data(ttl=3600)
def create_static_probability_distribution(_y_true, _y_proba):
    """Create the probability distribution plot ONCE.
    
    The threshold lines are NOT included here - they're added dynamically.
    This plot is the same regardless of threshold.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # These don't change with threshold
    ax.hist(_y_proba[_y_true == 0], bins=50, alpha=0.6, 
            label='No Default (Class 0)', color='green', density=True)
    ax.hist(_y_proba[_y_true == 1], bins=50, alpha=0.6, 
            label='Default (Class 1)', color='red', density=True)
    
    ax.set_xlabel("Predicted Probability of Default", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Distribution of Predicted Probabilities by Actual Class", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    return fig


@st.cache_data(ttl=3600)
def create_static_threshold_analysis(_y_true, _y_proba, cost_fn, cost_fp):
    """Create threshold analysis plots ONCE.
    
    Current threshold line is NOT included - added dynamically.
    """
    thresholds = np.arange(0.05, 0.95, 0.05)
    metrics_list = []
    
    for t in thresholds:
        y_pred_t = (_y_proba >= t).astype(int)
        cm_t = confusion_matrix(_y_true, y_pred_t)
        tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
        
        recall_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
        precision_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
        cost_t = cost_fn * fn_t + cost_fp * fp_t
        f1_t = 2 * precision_t * recall_t / (precision_t + recall_t) if (precision_t + recall_t) > 0 else 0
        
        metrics_list.append({
            'Threshold': t,
            'Recall': recall_t,
            'Precision': precision_t,
            'F1 Score': f1_t,
            'Business Cost': cost_t,
            'False Negatives': fn_t,
            'False Positives': fp_t
        })
    
    metrics_df = pd.DataFrame(metrics_list)
    optimal_idx = metrics_df['Business Cost'].idxmin()
    cost_optimal_threshold = metrics_df.loc[optimal_idx, 'Threshold']
    
    return metrics_df, cost_optimal_threshold


def render_model_performance():
    """Render the model performance page - optimized version."""
    logger.info("Rendering Model Performance page")
    st.title("📈 Model Performance")
    st.markdown("Analyze model performance with threshold optimization.")
    
    # Load predictions (cached)
    y_true, y_proba = load_predictions()
    
    if y_true is None:
        st.error("Predictions file not found!")
        st.info("Please run the model training pipeline first.")
        return
    
    st.success(f"✅ Loaded {len(y_true):,} predictions")
    
    # Get config values
    cost_fn = CONFIG['business']['cost_fn']
    cost_fp = CONFIG['business']['cost_fp']
    f_beta = CONFIG['business']['f_beta']
    optimal_threshold = 0.3282
    
    # Precompute ALL metrics once (cached)
    with st.spinner("Initializing metrics (one-time)..."):
        all_metrics = precompute_all_metrics(
            tuple(y_true), tuple(y_proba), cost_fn, cost_fp, f_beta
        )
        metrics_df, cost_optimal = create_static_threshold_analysis(
            tuple(y_true), tuple(y_proba), cost_fn, cost_fp
        )
    
    # ==========================================================================
    # THRESHOLD SELECTION - Slider changes now just look up precomputed values
    # ==========================================================================
    st.markdown("---")
    st.header("🎯 Threshold Selection")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        threshold = st.slider(
            "Decision Threshold",
            min_value=0.01, max_value=0.99, 
            value=0.33, step=0.01,
            help="Probability cutoff for classifying as Default",
            key="threshold_slider"
        )
    with col2:
        st.metric("Optimal Threshold", f"{optimal_threshold:.4f}")
    
    # Round threshold to match precomputed keys
    threshold_key = round(threshold, 2)
    if threshold_key not in all_metrics:
        threshold_key = min(all_metrics.keys(), key=lambda x: abs(x - threshold))
    
    # INSTANT lookup of precomputed metrics
    metrics = all_metrics[threshold_key]
    
    # ==========================================================================
    # KEY METRICS - Just display precomputed values (instant)
    # ==========================================================================
    st.markdown("---")
    st.header("📊 Key Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Business Cost", f"{metrics['cost']:,}")
    col2.metric("🎯 Recall", f"{metrics['recall']:.2%}")
    col3.metric("✅ Precision", f"{metrics['precision']:.2%}")
    col4.metric(f"📈 F{f_beta} Score", f"{metrics['fbeta']:.4f}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📏 Accuracy", f"{metrics['accuracy']:.2%}")
    col2.metric("⚖️ F1 Score", f"{metrics['f1']:.4f}")
    col3.metric("✔️ True Positives", f"{metrics['tp']:,}")
    col4.metric("❌ False Negatives", f"{metrics['fn']:,}")
    
    # ==========================================================================
    # CONFUSION MATRIX - Simple display
    # ==========================================================================
    st.markdown("---")
    st.header(f"📈 Confusion Matrix (Threshold = {threshold:.2f})")
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # Simple confusion matrix display
        cm = metrics['cm']
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        cm_pct = np.divide(cm.astype('float'), row_sums, 
                          out=np.zeros_like(cm, dtype=float), where=row_sums!=0)
        
        annot = np.array([[f'{cm[i,j]}\n({cm_pct[i,j]:.1%})' for j in range(2)] for i in range(2)])
        
        sns.heatmap(cm_pct, annot=annot, fmt='', cmap='Blues', ax=ax_cm,
                    xticklabels=['No Default', 'Default'],
                    yticklabels=['No Default', 'Default'])
        ax_cm.set_ylabel('Actual')
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_title('Confusion Matrix')
        
        st.pyplot(fig_cm)
        plt.close(fig_cm)
    
    with col_right:
        st.markdown("### Breakdown")
        st.write(f"✔️ **True Negatives:** {metrics['tn']:,}")
        st.write(f"⚠️ **False Positives:** {metrics['fp']:,}")
        st.write(f"❌ **False Negatives:** {metrics['fn']:,}")
        st.write(f"✔️ **True Positives:** {metrics['tp']:,}")
    
    # ==========================================================================
    # THRESHOLD ANALYSIS - Static plots with dynamic threshold line
    # ==========================================================================
    st.markdown("---")
    st.header("🔍 Threshold Analysis")
    
    # Create plots with current threshold marked
    fig_thresh, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Precision/Recall/F1
    axes[0].plot(metrics_df['Threshold'], metrics_df['Recall'], label='Recall', marker='o')
    axes[0].plot(metrics_df['Threshold'], metrics_df['Precision'], label='Precision', marker='s')
    axes[0].plot(metrics_df['Threshold'], metrics_df['F1 Score'], label='F1 Score', marker='^')
    axes[0].axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Current ({threshold:.2f})')
    axes[0].axvline(cost_optimal, color='green', linestyle=':', linewidth=2, label=f'Optimal ({cost_optimal:.2f})')
    axes[0].set_xlabel('Threshold')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Precision, Recall, F1 vs Threshold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    
    # Right plot: Business Cost
    axes[1].plot(metrics_df['Threshold'], metrics_df['Business Cost'], marker='o', color='red')
    axes[1].axvline(threshold, color='blue', linestyle='--', linewidth=2, label=f'Current ({threshold:.2f})')
    axes[1].axvline(cost_optimal, color='green', linestyle=':', linewidth=2, label=f'Optimal ({cost_optimal:.2f})')
    axes[1].set_xlabel('Threshold')
    axes[1].set_ylabel('Business Cost')
    axes[1].set_title('Business Cost vs Threshold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 1)
    
    plt.tight_layout()
    st.pyplot(fig_thresh)
    plt.close(fig_thresh)
    
    st.info(f"💡 **Cost-Optimal Threshold:** {cost_optimal:.2f}")
    
    # Metrics table
    with st.expander("📋 Detailed Metrics Table"):
        st.dataframe(
            metrics_df.style.format({
                'Threshold': '{:.2f}',
                'Recall': '{:.2%}',
                'Precision': '{:.2%}',
                'F1 Score': '{:.4f}',
                'Business Cost': '{:,.0f}'
            }),
            width="stretch"
        )
