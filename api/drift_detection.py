"""Data Drift Detection and Quality Monitoring

Implements statistical drift detection using:
- Kolmogorov-Smirnov (KS) test for continuous features
- Chi-square test for categorical features
- Population Stability Index (PSI)
- Missing value detection
- Out-of-range detection
"""

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy.orm import Session

from backend.models import DataDrift


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def calculate_ks_statistic(reference: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
    """
    Calculate Kolmogorov-Smirnov test statistic.
    
    Tests whether two distributions differ significantly.
    
    Args:
        reference: Reference distribution (training data)
        current: Current distribution (new data)
    
    Returns:
        (ks_statistic, p_value)
    """
    # Remove NaN values
    ref_clean = reference[~np.isnan(reference)]
    curr_clean = current[~np.isnan(current)]
    
    if len(ref_clean) == 0 or len(curr_clean) == 0:
        return 0.0, 1.0
    
    ks_stat, p_value = stats.ks_2samp(ref_clean, curr_clean)
    return float(ks_stat), float(p_value)


def calculate_chi_square(reference: pd.Series, current: pd.Series) -> Tuple[float, float]:
    """
    Calculate Chi-square test statistic for categorical features.
    
    Args:
        reference: Reference categorical distribution
        current: Current categorical distribution
    
    Returns:
        (chi2_statistic, p_value)
    """
    # Get value counts
    ref_counts = reference.value_counts()
    curr_counts = current.value_counts()
    
    # Align indices
    all_categories = set(ref_counts.index) | set(curr_counts.index)
    ref_counts = ref_counts.reindex(all_categories, fill_value=0)
    curr_counts = curr_counts.reindex(all_categories, fill_value=0)
    
    # Normalize to probabilities
    ref_probs = ref_counts / ref_counts.sum() if ref_counts.sum() > 0 else ref_counts
    curr_probs = curr_counts / curr_counts.sum() if curr_counts.sum() > 0 else curr_counts
    
    # Chi-square test
    # Add small value to avoid division by zero
    expected = (ref_probs + 1e-10) * len(current)
    observed = curr_counts.values
    
    chi2_stat = np.sum((observed - expected) ** 2 / expected)
    p_value = 1.0 - stats.chi2.cdf(chi2_stat, len(all_categories) - 1)
    
    return float(chi2_stat), float(p_value)


def calculate_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI).
    
    PSI measures the shift in a variable's distribution:
    - PSI < 0.1: No population shift detected
    - PSI 0.1-0.25: Small population shift
    - PSI > 0.25: Significant population shift
    
    Args:
        reference: Reference distribution
        current: Current distribution
        n_bins: Number of bins for histogram
    
    Returns:
        PSI value
    """
    # Remove NaN values
    ref_clean = reference[~np.isnan(reference)]
    curr_clean = current[~np.isnan(current)]
    
    if len(ref_clean) == 0 or len(curr_clean) == 0:
        return 0.0
    
    # Create bins based on reference distribution
    breakpoints = np.percentile(ref_clean, np.linspace(0, 100, n_bins + 1))
    breakpoints[0] = min(ref_clean.min(), curr_clean.min()) - 1e-6
    breakpoints[-1] = max(ref_clean.max(), curr_clean.max()) + 1e-6
    
    # Histogram both distributions
    ref_counts, _ = np.histogram(ref_clean, bins=breakpoints)
    curr_counts, _ = np.histogram(curr_clean, bins=breakpoints)
    
    # Convert to proportions
    ref_props = (ref_counts + 1e-10) / (ref_counts.sum() + 1e-10)
    curr_props = (curr_counts + 1e-10) / (curr_counts.sum() + 1e-10)
    
    # Calculate PSI
    psi = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))
    
    return float(psi)


# =============================================================================
# DATA QUALITY CHECKS
# =============================================================================

def check_missing_values(df: pd.DataFrame) -> Dict[str, float]:
    """
    Check missing value rates for all columns.
    
    Args:
        df: Input dataframe
    
    Returns:
        Dict mapping column names to missing value percentages
    """
    missing_rates = {}
    for col in df.columns:
        missing_pct = (df[col].isna().sum() / len(df)) * 100
        missing_rates[col] = round(missing_pct, 2)
    
    return missing_rates


def check_out_of_range(
    current_df: pd.DataFrame,
    reference_df: Optional[pd.DataFrame] = None,
    thresholds: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, Any]:
    """
    Check for out-of-range values in numeric features.
    
    Args:
        current_df: Current data to check
        reference_df: Reference data for statistical bounds
        thresholds: Manual thresholds as {col: (min, max)}
    
    Returns:
        Dict with out-of-range information
    """
    issues = {}
    
    for col in current_df.select_dtypes(include=[np.number]).columns:
        if col not in current_df.columns:
            continue
        
        col_data = current_df[col].dropna()
        if len(col_data) == 0:
            continue
        
        current_min = col_data.min()
        current_max = col_data.max()
        out_of_range_count = 0
        out_of_range_pct = 0.0
        
        # Use manual thresholds if provided
        if thresholds and col in thresholds:
            min_val, max_val = thresholds[col]
            out_of_range_count = ((col_data < min_val) | (col_data > max_val)).sum()
            out_of_range_pct = (out_of_range_count / len(col_data)) * 100
            issues[col] = {
                'min': float(current_min),
                'max': float(current_max),
                'expected_min': float(min_val),
                'expected_max': float(max_val),
                'out_of_range_count': int(out_of_range_count),
                'out_of_range_pct': round(out_of_range_pct, 2),
                'status': 'OK' if out_of_range_pct < 5 else 'WARNING'
            }
        
        # Use reference data bounds if provided
        elif reference_df is not None and col in reference_df.columns:
            ref_data = reference_df[col].dropna()
            if len(ref_data) > 0:
                ref_min = ref_data.min()
                ref_max = ref_data.max()
                out_of_range_count = ((col_data < ref_min) | (col_data > ref_max)).sum()
                out_of_range_pct = (out_of_range_count / len(col_data)) * 100
                
                issues[col] = {
                    'min': float(current_min),
                    'max': float(current_max),
                    'ref_min': float(ref_min),
                    'ref_max': float(ref_max),
                    'out_of_range_count': int(out_of_range_count),
                    'out_of_range_pct': round(out_of_range_pct, 2),
                    'status': 'OK' if out_of_range_pct < 5 else 'WARNING'
                }
    
    return issues


def validate_schema(df: pd.DataFrame, expected_columns: List[str]) -> Dict[str, Any]:
    """
    Validate dataframe schema against expected columns.
    
    Args:
        df: Dataframe to validate
        expected_columns: List of expected column names
    
    Returns:
        Validation results
    """
    current_columns = set(df.columns)
    expected_set = set(expected_columns)
    
    missing_cols = expected_set - current_columns
    extra_cols = current_columns - expected_set
    
    return {
        'valid': len(missing_cols) == 0 and len(extra_cols) == 0,
        'missing_columns': list(missing_cols),
        'extra_columns': list(extra_cols),
        'total_columns': len(df.columns),
        'expected_columns': len(expected_columns),
        'match_percentage': round((len(current_columns & expected_set) / len(expected_set)) * 100, 2)
    }


# =============================================================================
# DRIFT DETECTION
# =============================================================================

def detect_feature_drift(
    feature_name: str,
    reference_data: np.ndarray,
    current_data: np.ndarray,
    feature_type: str = 'numeric',
    alert_threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Detect drift in a single feature.
    
    Args:
        feature_name: Name of the feature
        reference_data: Reference (training) distribution
        current_data: Current (new) distribution
        feature_type: 'numeric' or 'categorical'
        alert_threshold: p-value threshold for alert (default 0.05)
    
    Returns:
        Drift detection results
    """
    result = {
        'feature_name': feature_name,
        'feature_type': feature_type,
    }
    
    if feature_type == 'numeric':
        # KS test for numeric features
        ks_stat, p_value = calculate_ks_statistic(reference_data, current_data)
        psi = calculate_psi(reference_data, current_data)
        
        result.update({
            'drift_test': 'KS',
            'ks_statistic': round(ks_stat, 4),
            'p_value': round(p_value, 4),
            'psi': round(psi, 4),
            'is_drifted': p_value < alert_threshold,
            'interpretation': _interpret_numeric_drift(ks_stat, psi, p_value)
        })
        
        # Add statistics
        result['reference_mean'] = float(np.nanmean(reference_data))
        result['current_mean'] = float(np.nanmean(current_data))
        result['reference_std'] = float(np.nanstd(reference_data))
        result['current_std'] = float(np.nanstd(current_data))
    
    else:  # categorical
        chi2_stat, p_value = calculate_chi_square(
            pd.Series(reference_data),
            pd.Series(current_data)
        )
        
        result.update({
            'drift_test': 'Chi-square',
            'chi2_statistic': round(chi2_stat, 4),
            'p_value': round(p_value, 4),
            'is_drifted': p_value < alert_threshold,
            'interpretation': _interpret_categorical_drift(chi2_stat, p_value)
        })
    
    return result


def _interpret_numeric_drift(ks_stat: float, psi: float, p_value: float) -> str:
    """Interpret numeric drift results."""
    if p_value > 0.05:
        status = "✅ No significant drift"
    elif psi < 0.1:
        status = "⚠️ Small drift (KS significant, low PSI)"
    elif psi < 0.25:
        status = "⚠️ Moderate drift"
    else:
        status = "🔴 Significant drift (high PSI)"
    
    return status


def _interpret_categorical_drift(chi2_stat: float, p_value: float) -> str:
    """Interpret categorical drift results."""
    if p_value > 0.05:
        return "✅ No significant drift"
    elif chi2_stat < 10:
        return "⚠️ Small drift detected"
    elif chi2_stat < 25:
        return "⚠️ Moderate drift detected"
    else:
        return "🔴 Significant drift detected"


# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

def save_drift_results(
    db: Session,
    feature_name: str,
    drift_results: Dict[str, Any],
    batch_id: Optional[int] = None
) -> DataDrift:
    """
    Save drift detection results to database.
    
    Args:
        db: Database session
        feature_name: Feature name
        drift_results: Results from detect_feature_drift()
        batch_id: Optional batch ID for context
    
    Returns:
        DataDrift database object
    """
    drift_record = DataDrift(
        feature_name=feature_name,
        drift_score=drift_results.get('psi') or drift_results.get('ks_statistic'),
        drift_type=drift_results.get('drift_test', 'unknown'),
        is_drifted=drift_results.get('is_drifted', False),
        reference_mean=drift_results.get('reference_mean'),
        current_mean=drift_results.get('current_mean'),
        reference_std=drift_results.get('reference_std'),
        current_std=drift_results.get('current_std'),
        batch_id=batch_id,
        n_samples=None
    )
    
    db.add(drift_record)
    db.commit()
    db.refresh(drift_record)
    
    return drift_record


def get_drift_history(
    db: Session,
    feature_name: str,
    limit: int = 30
) -> List[Dict[str, Any]]:
    """
    Get drift detection history for a feature.
    
    Args:
        db: Database session
        feature_name: Feature name
        limit: Number of records to return
    
    Returns:
        List of drift records
    """
    records = db.query(DataDrift).filter(
        DataDrift.feature_name == feature_name
    ).order_by(DataDrift.recorded_at.desc()).limit(limit).all()
    
    return [
        {
            'recorded_at': record.recorded_at.isoformat(),
            'drift_score': record.drift_score,
            'drift_type': record.drift_type,
            'is_drifted': record.is_drifted,
            'reference_mean': record.reference_mean,
            'current_mean': record.current_mean,
        }
        for record in records
    ]
