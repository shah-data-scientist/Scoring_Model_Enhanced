"""Compare feature values between X_train.csv and precomputed_features.parquet."""
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Load X_train.csv (original training features)
print("Loading X_train.csv...")
X_train_csv = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'X_train.csv')
print(f"X_train.csv shape: {X_train_csv.shape}")

# Load train_ids to map row indices to SK_ID_CURR
print("Loading train_ids.csv...")
train_ids = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'train_ids.csv')
print(f"train_ids shape: {train_ids.shape}")

# Add SK_ID_CURR to X_train
X_train_csv['SK_ID_CURR'] = train_ids['SK_ID_CURR'].values
print(f"Added SK_ID_CURR to X_train.csv\n")

# Load precomputed features from Parquet
print("Loading precomputed_features.parquet...")
parquet_df = pd.read_parquet(PROJECT_ROOT / 'data' / 'processed' / 'precomputed_features.parquet')
print(f"Parquet shape: {parquet_df.shape}\n")

# Test with the same sample IDs as test_lookup.py
sample_ids = [100002, 100003, 100004]

print("="*80)
print("COMPARING FEATURE VALUES FOR SAMPLE IDS")
print("="*80)

for sample_id in sample_ids:
    print(f"\nSample ID: {sample_id}")

    # Get row from X_train.csv
    csv_row = X_train_csv[X_train_csv['SK_ID_CURR'] == sample_id]
    if len(csv_row) == 0:
        print(f"  Not found in X_train.csv")
        continue
    csv_row = csv_row.iloc[0]

    # Get row from Parquet
    parquet_row = parquet_df[parquet_df['SK_ID_CURR'] == sample_id]
    if len(parquet_row) == 0:
        print(f"  Not found in Parquet")
        continue
    parquet_row = parquet_row.iloc[0]

    # Compare all features
    feature_cols = [c for c in X_train_csv.columns if c != 'SK_ID_CURR']

    differences = []
    for feat in feature_cols:
        csv_val = csv_row[feat]
        parquet_val = parquet_row[feat]

        # Handle NaN comparison
        if pd.isna(csv_val) and pd.isna(parquet_val):
            continue

        if pd.isna(csv_val) or pd.isna(parquet_val):
            differences.append((feat, csv_val, parquet_val))
        elif abs(csv_val - parquet_val) > 0.0001:
            differences.append((feat, csv_val, parquet_val))

    if differences:
        print(f"  [WARNING] {len(differences)} feature value differences!")
        print(f"  First 10 differences:")
        for feat, csv_val, parquet_val in differences[:10]:
            print(f"    {feat}: CSV={csv_val:.6f}, Parquet={parquet_val:.6f}, Diff={abs(csv_val - parquet_val):.6f}")
    else:
        print(f"  [OK] All {len(feature_cols)} features match perfectly!")

print("\n" + "="*80)
