"""
Model Performance Page for Credit Scoring Dashboard.

This page provides:
- Confusion matrix visualization
- Threshold optimization with guidance
- Business cost analysis
- Comprehensive metric explanations

Uses pre-computed static predictions to avoid repeated loading.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CONFIG

# Configuration
RESULTS_DIR = PROJECT_ROOT / CONFIG['paths']['results']
STATIC_PREDICTIONS_PATH = RESULTS_DIR / 'static_model_predictions.parquet'


@st.cache_data
def load_predictions():
    """Load cached predictions from static file or training predictions."""
    # First try to load from static parquet file (faster)
    if STATIC_PREDICTIONS_PATH.exists():
        try:
            df = pd.read_parquet(STATIC_PREDICTIONS_PATH)
            return df['TARGET'].values, df['PROBABILITY'].values
        except Exception:
            pass
    
    # Fall back to CSV predictions
    pred_path = RESULTS_DIR / 'train_predictions.csv'
    
    if not pred_path.exists():
        return None, None
    
    try:
        df = pd.read_csv(pred_path, usecols=['TARGET', 'PROBABILITY'])
        
        # Save as static parquet for future use (much faster)
        df.to_parquet(STATIC_PREDICTIONS_PATH, index=False)
        
        return df['TARGET'].values, df['PROBABILITY'].values
        
    except Exception as e:
        st.error(f"Error loading predictions: {e}")
        return None, None


def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix with percentages."""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_pct = np.divide(cm.astype('float'), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums!=0)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create annotation labels
    annot = np.array([[f'{cm[i,j]}\n({cm_pct[i,j]:.1%})' for j in range(2)] for i in range(2)])
    
    sns.heatmap(cm_pct, annot=annot, fmt='', cmap='Blues', ax=ax,
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default'])
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title('Confusion Matrix')
    
    return fig, cm


def render_model_performance():
    """Render the model performance page with comprehensive explanations."""
    
    st.title("📈 Model Performance")
    st.markdown("Analyze model performance with threshold optimization and business cost analysis.")
    
    # Load predictions
    y_true, y_proba = load_predictions()
    
    if y_true is None:
        st.error("Predictions file not found!")
        st.info("Please run the model training pipeline first to generate predictions.")
        st.code("poetry run python scripts/pipeline/apply_best_model.py", language="bash")
        return
    
    st.success(f"✅ Loaded {len(y_true):,} predictions from training data")
    
    # ==========================================================================
    # THRESHOLD SELECTION SECTION
    # ==========================================================================
    st.markdown("---")
    st.header("🎯 Threshold Selection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        threshold = st.slider(
            "Decision Threshold",
            0.0, 1.0, 0.328, 0.01,
            help="Probability cutoff for classifying as Default"
        )
    
    with col2:
        optimal_threshold = 0.3282
        st.metric(
            "Optimal Threshold (Training)",
            f"{optimal_threshold:.4f}",
            help="Threshold found during F-beta optimization"
        )
    
    # Threshold explanation
    with st.expander("ℹ️ **How to Select the Threshold - Guidance**", expanded=True):
        st.markdown("""
        ### Understanding the Decision Threshold
        
        The **decision threshold** determines when the model classifies an application as "Default" vs "No Default":
        - If predicted probability ≥ threshold → Classify as **Default** (high risk, reject loan)
        - If predicted probability < threshold → Classify as **No Default** (low risk, approve loan)
        
        ### Trade-offs in Threshold Selection
        
        | Lower Threshold (e.g., 0.2) | Higher Threshold (e.g., 0.5) |
        |:---------------------------|:-----------------------------|
        | More applications flagged as risky | Fewer applications flagged as risky |
        | ⬆️ Higher Recall (catches more defaults) | ⬇️ Lower Recall (misses more defaults) |
        | ⬇️ Lower Precision (more false alarms) | ⬆️ Higher Precision (fewer false alarms) |
        | Safer but more rejected applications | Riskier but more approved applications |
        
        ### Business Logic for Threshold Selection
        
        **For this credit scoring model, we prioritize catching defaults (high recall)** because:
        1. **Cost of False Negatives (FN)** = Approving a client who will default → **VERY EXPENSIVE** (loss of loan amount)
        2. **Cost of False Positives (FP)** = Rejecting a good client → **Less expensive** (lost business opportunity)
        
        The optimal threshold of **0.328** was determined by minimizing the **business cost function**:
        
        ```
        Business Cost = (Cost_FN × False Negatives) + (Cost_FP × False Positives)
        ```
        
        Where Cost_FN >> Cost_FP (defaults are much more costly than rejecting good clients).
        
        ### Recommendation
        - Use threshold around **0.30-0.35** for balanced risk management
        - Use **lower threshold (0.20-0.30)** if you want to be more conservative (catch more defaults)
        - Use **higher threshold (0.40-0.50)** only if you want to maximize loan approvals (accept more risk)
        """)
    
    # Calculate predictions at threshold
    y_pred = (y_proba >= threshold).astype(int)
    
    # Metrics calculation
    from sklearn.metrics import confusion_matrix, f1_score, fbeta_score, accuracy_score
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = f1_score(y_true, y_pred)
    fbeta = fbeta_score(y_true, y_pred, beta=CONFIG['business']['f_beta'])
    accuracy = accuracy_score(y_true, y_pred)
    
    # Business cost
    cost_fn = CONFIG['business']['cost_fn']
    cost_fp = CONFIG['business']['cost_fp']
    cost = cost_fn * fn + cost_fp * fp
    
    # ==========================================================================
    # KEY METRICS SECTION
    # ==========================================================================
    st.markdown("---")
    st.header("📊 Key Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Business Cost", f"{cost:,}", help="Total cost based on FN and FP")
    col2.metric("🎯 Recall (Sensitivity)", f"{recall:.2%}", help="% of actual defaults correctly identified")
    col3.metric("✅ Precision", f"{precision:.2%}", help="% of predicted defaults that are actual defaults")
    col4.metric(f"📈 F{CONFIG['business']['f_beta']} Score", f"{fbeta:.4f}", help="Weighted harmonic mean favoring recall")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📏 Accuracy", f"{accuracy:.2%}", help="Overall correct predictions")
    col2.metric("⚖️ F1 Score", f"{f1:.4f}", help="Harmonic mean of precision and recall")
    col3.metric("✔️ True Positives", f"{tp:,}", help="Correctly identified defaults")
    col4.metric("❌ False Negatives", f"{fn:,}", help="Missed defaults (costly!)")
    
    # Metrics explanation
    with st.expander("ℹ️ **Understanding the Metrics**", expanded=False):
        st.markdown(f"""
        ### Metric Definitions and Interpretation
        
        | Metric | Formula | Interpretation | Target |
        |:-------|:--------|:---------------|:-------|
        | **Recall (Sensitivity)** | TP / (TP + FN) | Of all actual defaults, what % did we catch? | **Higher is better** (catch defaults) |
        | **Precision** | TP / (TP + FP) | Of all predicted defaults, what % were correct? | Higher means fewer false alarms |
        | **Accuracy** | (TP + TN) / Total | Overall correct predictions | Can be misleading with imbalanced data |
        | **F1 Score** | 2 × (P × R) / (P + R) | Balance between precision and recall | Good overall metric |
        | **F{CONFIG['business']['f_beta']} Score** | Weighted F-score | Gives {CONFIG['business']['f_beta']}× more weight to recall | **Our primary metric** |
        
        ### Business Cost Calculation
        
        The business cost represents the financial impact of model errors:
        
        - **Cost of False Negative (FN)** = {cost_fn:,} (approving a defaulter)
        - **Cost of False Positive (FP)** = {cost_fp:,} (rejecting a good client)
        
        **Current Business Cost** = ({cost_fn} × {fn:,} FN) + ({cost_fp} × {fp:,} FP) = **{cost:,}**
        
        ### Why F{CONFIG['business']['f_beta']} Score?
        
        We use F-beta with β={CONFIG['business']['f_beta']} because:
        - β > 1 means we weight **recall more than precision**
        - This reflects our business priority: **catching defaults is more important than avoiding false alarms**
        - A missed default (FN) is much more costly than a false alarm (FP)
        """)
    
    # ==========================================================================
    # CONFUSION MATRIX SECTION
    # ==========================================================================
    st.markdown("---")
    st.header(f"📈 Confusion Matrix at Threshold = {threshold:.3f}")
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        fig, _ = plot_confusion_matrix(y_true, y_pred)
        st.pyplot(fig)
        plt.close()
    
    with col_right:
        st.markdown("### Raw Counts")
        cm_df = pd.DataFrame(
            cm,
            columns=['Pred: No Default', 'Pred: Default'],
            index=['Actual: No Default', 'Actual: Default']
        )
        st.dataframe(cm_df, use_container_width=True)
        
        st.markdown("### Breakdown")
        st.write(f"✔️ **True Negatives:** {tn:,} (correctly approved)")
        st.write(f"⚠️ **False Positives:** {fp:,} (incorrectly rejected)")
        st.write(f"❌ **False Negatives:** {fn:,} (missed defaults)")
        st.write(f"✔️ **True Positives:** {tp:,} (correctly rejected)")
    
    # Confusion matrix explanation
    with st.expander("ℹ️ **How to Read the Confusion Matrix**", expanded=False):
        st.markdown("""
        ### Confusion Matrix Quadrants
        
        ```
                           PREDICTED
                    No Default  |  Default
                   ─────────────┼──────────
        ACTUAL     |    TN     |    FP    |  → All actual No Defaults
        No Default |  (correct)|  (error) |
                   ─────────────┼──────────
        ACTUAL     |    FN     |    TP    |  → All actual Defaults
        Default    |  (error) |  (correct)|
                   ─────────────┴──────────
        ```
        
        ### Reading the Percentages
        
        - **Row percentages** (what we show): "Of all clients who actually [Row Label], what % did we predict as [Column Label]?"
          - Top-right cell: **False Positive Rate** = FP / (TN + FP)
          - Bottom-right cell: **True Positive Rate (Recall)** = TP / (FN + TP)
        
        ### Business Interpretation
        
        | Quadrant | Business Meaning | Cost |
        |:---------|:-----------------|:-----|
        | **True Negative (TN)** | Good client approved → Loan interest earned | ✅ Profit |
        | **False Positive (FP)** | Good client rejected → Lost business | ⚠️ Opportunity cost |
        | **False Negative (FN)** | Bad client approved → **LOAN DEFAULT** | ❌ **Major loss** |
        | **True Positive (TP)** | Bad client rejected → Loss avoided | ✅ Loss prevented |
        """)
    
    # ==========================================================================
    # PROBABILITY DISTRIBUTION SECTION
    # ==========================================================================
    st.markdown("---")
    st.header("📉 Probability Distribution")
    
    fig_hist, ax_hist = plt.subplots(figsize=(12, 5))
    
    # Plot distributions
    ax_hist.hist(y_proba[y_true == 0], bins=50, alpha=0.6, label='No Default (Class 0)', color='green', density=True)
    ax_hist.hist(y_proba[y_true == 1], bins=50, alpha=0.6, label='Default (Class 1)', color='red', density=True)
    ax_hist.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    ax_hist.axvline(optimal_threshold, color='blue', linestyle=':', linewidth=2, label=f'Optimal ({optimal_threshold})')
    ax_hist.set_xlabel("Predicted Probability of Default", fontsize=12)
    ax_hist.set_ylabel("Density", fontsize=12)
    ax_hist.legend(fontsize=10)
    ax_hist.set_title("Distribution of Predicted Probabilities by Actual Class", fontsize=14)
    ax_hist.grid(True, alpha=0.3)
    
    st.pyplot(fig_hist)
    plt.close()
    
    with st.expander("ℹ️ **Understanding the Probability Distribution**", expanded=False):
        st.markdown("""
        ### What This Chart Shows
        
        This histogram shows how the model's predicted probabilities are distributed for each actual class:
        
        - **Green bars (No Default):** Distribution of probabilities for clients who did NOT default
        - **Red bars (Default):** Distribution of probabilities for clients who DID default
        
        ### Ideal Scenario
        
        In a perfect model:
        - All green bars would be at probability = 0 (left side)
        - All red bars would be at probability = 1 (right side)
        - There would be no overlap
        
        ### Reality
        
        In practice, the distributions overlap. The **threshold** (vertical line) determines where we "cut":
        - Everything to the **right** of the threshold → Predicted as Default
        - Everything to the **left** of the threshold → Predicted as No Default
        
        ### How to Use This
        
        - If you move the threshold **left**: More red bars are captured (higher recall), but also more green bars (lower precision)
        - If you move the threshold **right**: Fewer green bars are included (higher precision), but also fewer red bars (lower recall)
        """)
    
    # ==========================================================================
    # THRESHOLD ANALYSIS SECTION
    # ==========================================================================
    st.markdown("---")
    st.header("🎯 Threshold Analysis")
    
    thresholds = np.arange(0.05, 0.95, 0.05)
    metrics_list = []
    
    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        cm_t = confusion_matrix(y_true, y_pred_t)
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
    
    # Find optimal threshold by business cost
    optimal_idx = metrics_df['Business Cost'].idxmin()
    cost_optimal_threshold = metrics_df.loc[optimal_idx, 'Threshold']
    
    # Plot
    fig_thresh, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(metrics_df['Threshold'], metrics_df['Recall'], label='Recall', marker='o', linewidth=2)
    axes[0].plot(metrics_df['Threshold'], metrics_df['Precision'], label='Precision', marker='s', linewidth=2)
    axes[0].plot(metrics_df['Threshold'], metrics_df['F1 Score'], label='F1 Score', marker='^', linewidth=2)
    axes[0].axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Current ({threshold})')
    axes[0].axvline(cost_optimal_threshold, color='green', linestyle=':', linewidth=2, label=f'Cost Optimal ({cost_optimal_threshold:.2f})')
    axes[0].set_xlabel('Threshold', fontsize=12)
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].set_title('Precision, Recall, F1 vs Threshold', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    
    axes[1].plot(metrics_df['Threshold'], metrics_df['Business Cost'], label='Business Cost', marker='o', color='red', linewidth=2)
    axes[1].axvline(threshold, color='blue', linestyle='--', linewidth=2, label=f'Current ({threshold})')
    axes[1].axvline(cost_optimal_threshold, color='green', linestyle=':', linewidth=2, label=f'Cost Optimal ({cost_optimal_threshold:.2f})')
    axes[1].set_xlabel('Threshold', fontsize=12)
    axes[1].set_ylabel('Business Cost', fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].set_title('Business Cost vs Threshold', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 1)
    
    plt.tight_layout()
    st.pyplot(fig_thresh)
    plt.close()
    
    st.info(f"💡 **Cost-Optimal Threshold:** {cost_optimal_threshold:.2f} (minimizes business cost)")
    
    with st.expander("ℹ️ **Understanding the Threshold Charts**", expanded=False):
        st.markdown("""
        ### Left Chart: Precision-Recall Trade-off
        
        - **Recall (blue):** Decreases as threshold increases (fewer defaults caught)
        - **Precision (orange):** Increases as threshold increases (more accurate positive predictions)
        - **F1 Score (green):** Balanced metric, typically peaks at moderate thresholds
        
        The **crossing point** of precision and recall is sometimes called the "break-even point."
        
        ### Right Chart: Business Cost
        
        - Shows the total financial cost at each threshold
        - **U-shaped curve:** Too low or too high thresholds both increase cost
        - **Minimum point:** The optimal threshold for business
        
        ### Why Cost is U-Shaped
        
        - **Low threshold:** Too many false positives (rejecting good clients) → Lost business
        - **High threshold:** Too many false negatives (approving bad clients) → Loan defaults
        - **Optimal:** Balance between these two costs
        """)
    
    # Data table
    with st.expander("📋 Detailed Metrics by Threshold"):
        st.dataframe(
            metrics_df.style.format({
                'Threshold': '{:.2f}',
                'Recall': '{:.2%}',
                'Precision': '{:.2%}',
                'F1 Score': '{:.4f}',
                'Business Cost': '{:,.0f}',
                'False Negatives': '{:,.0f}',
                'False Positives': '{:,.0f}'
            }),
            use_container_width=True
        )
