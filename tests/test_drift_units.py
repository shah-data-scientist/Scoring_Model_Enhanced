import numpy as np
import pandas as pd
import pytest
from api.drift_detection import (
    calculate_ks_statistic,
    calculate_chi_square,
    calculate_psi,
    check_missing_values,
    check_out_of_range,
    detect_feature_drift
)

def test_calculate_ks_statistic_identical():
    data = np.random.normal(0, 1, 100)
    ks, p = calculate_ks_statistic(data, data)
    assert ks == 0.0
    assert p == 1.0

def test_calculate_ks_statistic_different():
    ref = np.random.normal(0, 1, 100)
    curr = np.random.normal(5, 1, 100)
    ks, p = calculate_ks_statistic(ref, curr)
    assert ks > 0.5
    assert p < 0.05

def test_calculate_chi_square_identical():
    ref = pd.Series(['A', 'B', 'C'] * 30)
    curr = pd.Series(['A', 'B', 'C'] * 30)
    chi2, p = calculate_chi_square(ref, curr)
    assert chi2 < 1.0
    assert p > 0.9

def test_calculate_psi_stable():
    ref = np.random.normal(0, 1, 1000)
    curr = np.random.normal(0, 1, 1000)
    psi = calculate_psi(ref, curr)
    assert psi < 0.1

def test_calculate_psi_drifted():
    ref = np.random.normal(0, 1, 1000)
    curr = np.random.normal(1, 1, 1000)
    psi = calculate_psi(ref, curr)
    assert psi > 0.1

def test_check_missing_values():
    df = pd.DataFrame({
        'A': [1, None, 3, 4],
        'B': [None, None, 3, 4]
    })
    missing = check_missing_values(df)
    assert missing['A'] == 25.0
    assert missing['B'] == 50.0

def test_check_out_of_range():
    curr = pd.DataFrame({'val': [1, 2, 10, 4, 5]})
    ref = pd.DataFrame({'val': [1, 2, 3, 4, 5]})
    issues = check_out_of_range(curr, reference_df=ref)
    assert 'val' in issues
    assert issues['val']['out_of_range_count'] == 1

def test_detect_feature_drift_numeric():
    ref = np.random.normal(0, 1, 100)
    curr = np.random.normal(0, 1, 100)
    result = detect_feature_drift("test", ref, curr, feature_type='numeric')
    assert result['feature_name'] == "test"
    assert 'ks_statistic' in result
    assert 'psi' in result

def test_detect_feature_drift_categorical():
    ref = np.array(['A', 'B'] * 50)
    curr = np.array(['A', 'B'] * 50)
    result = detect_feature_drift("test_cat", ref, curr, feature_type='categorical')
    assert result['feature_name'] == "test_cat"
    assert 'chi2_statistic' in result
