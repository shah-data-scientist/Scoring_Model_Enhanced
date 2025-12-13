"""Debug: Extract actual features sent to model for both datasets."""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.preprocessing_pipeline import preprocess_data

END_USER_DIR = PROJECT_ROOT / 'data' / 'end_user_tests'
ANON_DIR = PROJECT_ROOT / 'data' / 'end_user_tests_anonymized'

print("="*80)
print("FEATURE COMPARISON: Original vs Anonymized")
print("="*80)

# Load datasets
def load_data(directory):
    """Load all CSV files from directory."""
    return {
        'application': pd.read_csv(directory / 'application.csv'),
        'bureau': pd.read_csv(directory / 'bureau.csv'),
        'bureau_balance': pd.read_csv(directory / 'bureau_balance.csv'),
        'credit_card_balance': pd.read_csv(directory / 'credit_card_balance.csv'),
        'installments_payments': pd.read_csv(directory / 'installments_payments.csv'),
        'pos_cash_balance': pd.read_csv(directory / 'POS_CASH_balance.csv'),
        'previous_application': pd.read_csv(directory / 'previous_application.csv')
    }

print("\n[1/2] Processing ORIGINAL data...")
orig_data = load_data(END_USER_DIR)
orig_features, orig_ids, orig_info = preprocess_data(orig_data)
print(f"  ✓ Generated {len(orig_features)} rows x {orig_features.shape[1]} features")

print("\n[2/2] Processing ANONYMIZED data...")
anon_data = load_data(ANON_DIR)
anon_features, anon_ids, anon_info = preprocess_data(anon_data)
print(f"  ✓ Generated {len(anon_features)} rows x {anon_features.shape[1]} features")

# Map anonymized IDs back to original
orig_app = orig_data['application']
anon_app = anon_data['application']

# Create mapping based on row position
id_mapping_reverse = {}
for i in range(len(orig_app)):
    id_mapping_reverse[anon_app.iloc[i]['SK_ID_CURR']] = orig_app.iloc[i]['SK_ID_CURR']

# Reorder anonymized features to match original order
anon_features['_orig_id'] = anon_ids.map(id_mapping_reverse)
anon_features = anon_features.sort_values('_orig_id')
anon_features = anon_features.drop(columns=['_orig_id']).reset_index(drop=True)

orig_features = orig_features.sort_values(orig_ids.name if hasattr(orig_ids, 'name') else orig_ids)
orig_features = orig_features.reset_index(drop=True)

# Compare features
print("\n" + "="*80)
print("FEATURE COMPARISON")
print("="*80)

# Check if feature arrays are identical
features_identical = np.allclose(orig_features.values, anon_features.values, rtol=1e-10, atol=1e-10)
print(f"\nFeatures identical: {features_identical}")

if not features_identical:
    # Find differences
    diff_mask = ~np.isclose(orig_features.values, anon_features.values, rtol=1e-10, atol=1e-10)
    diff_rows, diff_cols = np.where(diff_mask)
    
    print(f"\nDifferences found: {len(diff_rows)} cells")
    print(f"Affected rows: {len(set(diff_rows))} out of {len(orig_features)}")
    print(f"Affected features: {len(set(diff_cols))} out of {orig_features.shape[1]}")
    
    # Show first 10 differences
    print(f"\nFirst 10 differences:")
    print(f"{'Row':<6} {'Feature':<40} {'Original':<15} {'Anonymized':<15} {'Diff':<15}")
    print("-"*95)
    
    for i, (row_idx, col_idx) in enumerate(zip(diff_rows[:10], diff_cols[:10])):
        feature_name = orig_features.columns[col_idx]
        orig_val = orig_features.iloc[row_idx, col_idx]
        anon_val = anon_features.iloc[row_idx, col_idx]
        diff = abs(orig_val - anon_val)
        
        print(f"{row_idx:<6} {feature_name:<40} {orig_val:<15.6f} {anon_val:<15.6f} {diff:<15.6f}")
    
    # Group by feature
    unique_cols = set(diff_cols)
    print(f"\n\nAffected features ({len(unique_cols)} total):")
    for col_idx in sorted(unique_cols)[:20]:
        feature_name = orig_features.columns[col_idx]
        rows_affected = len(diff_rows[diff_cols == col_idx])
        print(f"  {feature_name}: {rows_affected} rows affected")
else:
    print("\n  ✓✓✓ ALL FEATURES IDENTICAL! ✓✓✓")
    print("  This means SK_ID_CURR truly has no impact on feature engineering")

print("\n" + "="*80)
