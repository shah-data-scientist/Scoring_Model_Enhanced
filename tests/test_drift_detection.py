"""Quick test of drift detection functionality"""

import numpy as np
from api.drift_detection import (
    calculate_ks_statistic,
    calculate_psi,
    detect_feature_drift,
    check_missing_values,
    check_out_of_range,
    validate_schema
)
import pandas as pd


def test_ks_statistic():
    """Test KS statistic calculation."""
    print("\n=== Testing KS Statistic ===")
    
    # Same distribution
    data1 = np.random.normal(0.5, 0.1, 100)
    ks_stat, p_val = calculate_ks_statistic(data1, data1)
    print(f"Identical distributions: KS={ks_stat:.4f}, p={p_val:.4f} (should be ~0)")
    
    # Different distributions
    data2 = np.random.normal(0.5, 0.1, 100)
    data3 = np.random.normal(0.6, 0.1, 100)
    ks_stat, p_val = calculate_ks_statistic(data2, data3)
    print(f"Different distributions: KS={ks_stat:.4f}, p={p_val:.4f} (should be <0.05)")


def test_psi():
    """Test PSI calculation."""
    print("\n=== Testing PSI ===")
    
    # No drift
    data1 = np.random.normal(0.5, 0.1, 100)
    psi = calculate_psi(data1, data1)
    print(f"No drift (same data): PSI={psi:.4f} (should be ~0)")
    
    # Small drift
    data2 = np.random.normal(0.5, 0.1, 100)
    data3 = np.random.normal(0.52, 0.1, 100)
    psi = calculate_psi(data2, data3)
    print(f"Small drift: PSI={psi:.4f} (should be <0.1)")
    
    # Larger drift
    data4 = np.random.normal(0.5, 0.1, 100)
    data5 = np.random.normal(0.7, 0.1, 100)
    psi = calculate_psi(data4, data5)
    print(f"Large drift: PSI={psi:.4f} (should be >0.1)")


def test_feature_drift():
    """Test feature drift detection."""
    print("\n=== Testing Feature Drift Detection ===")
    
    reference = np.random.normal(0.5, 0.1, 100)
    current = np.random.normal(0.52, 0.12, 100)
    
    results = detect_feature_drift(
        'EXT_SOURCE_1',
        reference,
        current,
        feature_type='numeric',
        alert_threshold=0.05
    )
    
    print(f"Feature: {results['feature_name']}")
    print(f"Test: {results['drift_test']}")
    print(f"KS Statistic: {results['ks_statistic']}")
    print(f"PSI: {results['psi']}")
    print(f"P-value: {results['p_value']}")
    print(f"Drifted: {results['is_drifted']}")
    print(f"Status: {results['interpretation']}")


def test_data_quality():
    """Test data quality checks."""
    print("\n=== Testing Data Quality Checks ===")
    
    # Create test DataFrame
    df = pd.DataFrame({
        'amt_credit': [500000, 450000, np.nan, 550000],
        'amt_income': [180000, 160000, 200000, 190000],
        'ext_source': [0.5, 0.6, 0.55, 0.4]
    })
    
    # Test missing values
    print("\nMissing Values:")
    missing = check_missing_values(df)
    for col, pct in missing.items():
        print(f"  {col}: {pct}%")
    
    # Test schema validation
    print("\nSchema Validation:")
    schema = validate_schema(df, ['amt_credit', 'amt_income', 'extra_col'])
    print(f"  Valid: {schema['valid']}")
    print(f"  Match: {schema['match_percentage']}%")
    print(f"  Missing: {schema['missing_columns']}")


def main():
    """Run all tests."""
    print("="*50)
    print("DRIFT DETECTION & QUALITY MONITORING TESTS")
    print("="*50)
    
    test_ks_statistic()
    test_psi()
    test_feature_drift()
    test_data_quality()
    
    print("\n" + "="*50)
    print("✅ All tests completed successfully!")
    print("="*50)


if __name__ == "__main__":
    main()
