"""Metrics API Module

Precomputes and serves model performance metrics.
This offloads heavy computation from Streamlit to the API.
"""
from pathlib import Path
from typing import Dict, Any
import logging

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, fbeta_score

# Setup logging
logger = logging.getLogger(__name__)

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
STATIC_PREDICTIONS_PATH = RESULTS_DIR / "static_model_predictions.parquet"
PRECOMPUTED_METRICS_PATH = RESULTS_DIR / "precomputed_metrics.parquet"

# Router
router = APIRouter(prefix="/metrics", tags=["Metrics"])

# Global cache for metrics
_METRICS_CACHE: Dict[str, Any] = {}


class MetricsResponse(BaseModel):
    """Response model for metrics endpoint."""
    thresholds: list[float]
    metrics: dict[str, dict[str, Any]]
    optimal_threshold: float
    metrics_df: dict[str, list[Any]]
    data_count: int
    cached: bool


class ThresholdMetricsResponse(BaseModel):
    """Response model for single threshold metrics."""
    threshold: float
    tn: int
    fp: int
    fn: int
    tp: int
    recall: float
    precision: float
    f1: float
    fbeta: float
    accuracy: float
    cost: int


def load_predictions():
    """Load cached predictions from static file."""
    logger.info("Loading predictions for metrics computation")

    if STATIC_PREDICTIONS_PATH.exists():
        try:
            df = pd.read_parquet(STATIC_PREDICTIONS_PATH)
            logger.info(f"Loaded {len(df)} predictions from parquet")
            return df['TARGET'].values, df['PROBABILITY'].values
        except Exception as e:
            logger.error(f"Error loading parquet: {e}")

    # Fallback to CSV
    pred_path = RESULTS_DIR / 'train_predictions.csv'
    if pred_path.exists():
        try:
            df = pd.read_csv(
                pred_path,
                usecols=['TARGET', 'PROBABILITY'],
                dtype={'TARGET': 'int8', 'PROBABILITY': 'float32'}
            )
            # Save as parquet for future
            df.to_parquet(STATIC_PREDICTIONS_PATH, index=False)
            logger.info(f"Loaded {len(df)} predictions from CSV and saved to parquet")
            return df['TARGET'].values, df['PROBABILITY'].values
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")

    return None, None


def compute_metrics_for_threshold(y_true, y_proba, threshold, cost_fn=10, cost_fp=1, f_beta=3.2):
    """Compute all metrics for a single threshold using vectorized numpy."""
    # Fast vectorized thresholding
    y_pred = (y_proba >= threshold).astype(np.int8)
    
    # Fast confusion matrix elements calculation
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Custom fast F-score calculation
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
        beta_sq = f_beta ** 2
        fbeta_val = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)
    else:
        f1 = 0.0
        fbeta_val = 0.0
        
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    cost = cost_fn * fn + cost_fp * fp

    return {
        'cm': [[int(tn), int(fp)], [int(fn), int(tp)]],
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'recall': float(recall),
        'precision': float(precision),
        'f1': float(f1),
        'fbeta': float(fbeta_val),
        'accuracy': float(accuracy),
        'cost': int(cost)
    }


def precompute_all_metrics(cost_fn=10, cost_fp=1, f_beta=3.2):
    """Precompute metrics for all thresholds and cache them.

    This runs once on API startup and caches results in memory and disk.
    """
    logger.info("Starting metrics precomputation")

    # Check if already cached in memory
    if _METRICS_CACHE.get('computed'):
        logger.info("Returning cached metrics from memory")
        return _METRICS_CACHE

    # Check if precomputed file exists
    if PRECOMPUTED_METRICS_PATH.exists():
        try:
            logger.info("Loading precomputed metrics from disk")
            df = pd.read_parquet(PRECOMPUTED_METRICS_PATH)

            # Reconstruct metrics dictionary
            all_metrics = {}
            for _, row in df.iterrows():
                threshold = round(row['threshold'], 2)
                all_metrics[threshold] = {
                    'cm': [[row['cm_00'], row['cm_01']], [row['cm_10'], row['cm_11']]],
                    'tn': int(row['tn']),
                    'fp': int(row['fp']),
                    'fn': int(row['fn']),
                    'tp': int(row['tp']),
                    'recall': float(row['recall']),
                    'precision': float(row['precision']),
                    'f1': float(row['f1']),
                    'fbeta': float(row['fbeta']),
                    'accuracy': float(row['accuracy']),
                    'cost': int(row['cost'])
                }

            # Compute threshold analysis
            metrics_df, optimal_threshold = compute_threshold_analysis_from_metrics(all_metrics, cost_fn, cost_fp)

            # Cache in memory
            _METRICS_CACHE.update({
                'all_metrics': all_metrics,
                'metrics_df': metrics_df,
                'optimal_threshold': optimal_threshold,
                'data_count': int(df['data_count'].iloc[0]) if 'data_count' in df.columns else len(df),
                'computed': True,
                'cached': True
            })

            logger.info(f"Loaded precomputed metrics for {len(all_metrics)} thresholds")
            return _METRICS_CACHE
        except Exception as e:
            logger.warning(f"Error loading precomputed metrics: {e}, recomputing...")

    # Compute from scratch
    logger.info("Computing metrics from scratch")
    y_true, y_proba = load_predictions()

    if y_true is None:
        raise ValueError("Predictions file not found")

    all_metrics = {}

    # Compute for every 2% threshold (0.01 to 0.99) - 50 thresholds
    # Reduced from 99 to 50 for faster computation
    thresholds = np.arange(0.01, 1.0, 0.02)

    for t in thresholds:
        threshold = round(t, 2)
        all_metrics[threshold] = compute_metrics_for_threshold(
            y_true, y_proba, threshold, cost_fn, cost_fp, f_beta
        )

    # Compute threshold analysis (for plots)
    metrics_df, optimal_threshold = compute_threshold_analysis(y_true, y_proba, cost_fn, cost_fp)

    # Save to disk for future use
    save_precomputed_metrics(all_metrics, len(y_true))

    # Cache in memory
    _METRICS_CACHE.update({
        'all_metrics': all_metrics,
        'metrics_df': metrics_df,
        'optimal_threshold': optimal_threshold,
        'data_count': len(y_true),
        'computed': True,
        'cached': False
    })

    logger.info(f"Computed and cached metrics for {len(all_metrics)} thresholds")
    return _METRICS_CACHE


def compute_threshold_analysis(y_true, y_proba, cost_fn=10, cost_fp=1):
    """Compute threshold analysis metrics for plots using vectorized numpy."""
    # Use finer granularity (0.01) to find true optimal threshold
    thresholds = np.arange(0.01, 1.0, 0.01)
    metrics_list = []

    # Vectorized computation for ALL thresholds at once
    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(np.int8)
        
        tp_t = np.sum((y_pred_t == 1) & (y_true == 1))
        tn_t = np.sum((y_pred_t == 0) & (y_true == 0))
        fp_t = np.sum((y_pred_t == 1) & (y_true == 0))
        fn_t = np.sum((y_pred_t == 0) & (y_true == 1))

        recall_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
        precision_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
        cost_t = cost_fn * fn_t + cost_fp * fp_t
        f1_t = 2 * precision_t * recall_t / (precision_t + recall_t) if (precision_t + recall_t) > 0 else 0

        metrics_list.append({
            'Threshold': float(t),
            'Recall': float(recall_t),
            'Precision': float(precision_t),
            'F1 Score': float(f1_t),
            'Business Cost': int(cost_t),
            'False Negatives': int(fn_t),
            'False Positives': int(fp_t)
        })

    metrics_df = pd.DataFrame(metrics_list)
    optimal_idx = metrics_df['Business Cost'].idxmin()
    optimal_threshold = metrics_df.loc[optimal_idx, 'Threshold']

    return metrics_df.to_dict('list'), float(optimal_threshold)


def compute_threshold_analysis_from_metrics(all_metrics, cost_fn=10, cost_fp=1):
    """Compute threshold analysis from precomputed metrics."""
    # Use every 0.01 threshold for analysis (matches precomputed data at 0.02 intervals)
    analysis_thresholds = [round(t, 2) for t in np.arange(0.01, 1.0, 0.01)]

    metrics_list = []
    for t in analysis_thresholds:
        # Find closest threshold in all_metrics
        closest = min(all_metrics.keys(), key=lambda x: abs(x - t))
        metrics = all_metrics[closest]

        metrics_list.append({
            'Threshold': t,
            'Recall': metrics['recall'],
            'Precision': metrics['precision'],
            'F1 Score': metrics['f1'],
            'Business Cost': metrics['cost'],
            'False Negatives': metrics['fn'],
            'False Positives': metrics['fp']
        })

    metrics_df = pd.DataFrame(metrics_list)
    optimal_idx = metrics_df['Business Cost'].idxmin()
    optimal_threshold = metrics_df.loc[optimal_idx, 'Threshold']

    return metrics_df.to_dict('list'), float(optimal_threshold)


def save_precomputed_metrics(all_metrics, data_count):
    """Save precomputed metrics to disk."""
    try:
        rows = []
        for threshold, metrics in all_metrics.items():
            cm = metrics['cm']
            rows.append({
                'threshold': threshold,
                'cm_00': cm[0][0] if isinstance(cm, list) else cm[0, 0],
                'cm_01': cm[0][1] if isinstance(cm, list) else cm[0, 1],
                'cm_10': cm[1][0] if isinstance(cm, list) else cm[1, 0],
                'cm_11': cm[1][1] if isinstance(cm, list) else cm[1, 1],
                'tn': metrics['tn'],
                'fp': metrics['fp'],
                'fn': metrics['fn'],
                'tp': metrics['tp'],
                'recall': metrics['recall'],
                'precision': metrics['precision'],
                'f1': metrics['f1'],
                'fbeta': metrics['fbeta'],
                'accuracy': metrics['accuracy'],
                'cost': metrics['cost'],
                'data_count': data_count
            })

        df = pd.DataFrame(rows)
        df.to_parquet(PRECOMPUTED_METRICS_PATH, index=False)
        logger.info(f"Saved precomputed metrics to {PRECOMPUTED_METRICS_PATH}")
    except Exception as e:
        logger.error(f"Error saving precomputed metrics: {e}")


@router.get("/precomputed")
async def get_precomputed_metrics():
    """Get all precomputed metrics.

    This endpoint returns cached metrics for all thresholds.
    First call may take a few seconds to compute, subsequent calls are instant.
    """
    try:
        cache = precompute_all_metrics()

        # Convert all numeric keys to strings for JSON serialization
        metrics_str_keys = {str(k): v for k, v in cache['all_metrics'].items()}

        return {
            "thresholds": sorted([float(k) for k in metrics_str_keys.keys()]),
            "metrics": metrics_str_keys,
            "optimal_threshold": float(cache['optimal_threshold']),
            "metrics_df": cache['metrics_df'],
            "data_count": int(cache['data_count']),
            "cached": bool(cache['cached'])
        }
    except Exception as e:
        logger.error(f"Error getting precomputed metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# UNUSED: Threshold-specific metrics endpoint not used by Streamlit
# @router.get("/threshold/{threshold}", response_model=ThresholdMetricsResponse)
# async def get_threshold_metrics(threshold: float):
#     """Get metrics for a specific threshold.
#
#     Args:
#         threshold: Threshold value (0.01 to 0.99)
#
#     Returns:
#         Metrics for the specified threshold
#     """
#     if threshold < 0.01 or threshold > 0.99:
#         raise HTTPException(status_code=400, detail="Threshold must be between 0.01 and 0.99")
#
#     try:
#         cache = precompute_all_metrics()
#
#         # Round to 2 decimal places
#         threshold_key = round(threshold, 2)
#
#         # Find closest threshold if exact not found
#         if threshold_key not in cache['all_metrics']:
#             threshold_key = min(cache['all_metrics'].keys(), key=lambda x: abs(x - threshold))
#
#         metrics = cache['all_metrics'][threshold_key]
#
#         return ThresholdMetricsResponse(
#             threshold=threshold_key,
#             **metrics
#         )
#     except Exception as e:
#         logger.error(f"Error getting threshold metrics: {e}")
#         raise HTTPException(status_code=500, detail=str(e))


# UNUSED: Metrics recompute endpoint not used by Streamlit
# @router.post("/recompute")
# async def recompute_metrics():
#     """Recompute all metrics (admin only).
#
#     This triggers a fresh computation of all metrics.
#     Use when predictions data has been updated.
#     """
#     try:
#         # Clear cache
#         _METRICS_CACHE.clear()
#
#         # Delete precomputed file
#         if PRECOMPUTED_METRICS_PATH.exists():
#             PRECOMPUTED_METRICS_PATH.unlink()
#
#         # Recompute
#         cache = precompute_all_metrics()
#
#         return {
#             "status": "success",
#             "message": f"Recomputed metrics for {len(cache['all_metrics'])} thresholds",
#             "data_count": cache['data_count']
#         }
#     except Exception as e:
#         logger.error(f"Error recomputing metrics: {e}")
#         raise HTTPException(status_code=500, detail=str(e))
