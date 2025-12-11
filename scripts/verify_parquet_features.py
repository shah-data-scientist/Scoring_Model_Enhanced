"""Verify the Parquet file contains the exact features we just loaded."""
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

print("="*80)
print("VERIFYING PARQUET FEATURES")
print("="*80)

# Load the parquet file we just created
parquet_path = PROJECT_ROOT / 'data' / 'processed' / 'precomputed_features.parquet'
print(f"\nLoading: {parquet_path}")
parquet_df = pd.read_parquet(parquet_path)
print(f"Shape: {parquet_df.shape}")
print(f"Columns: {list(parquet_df.columns[:5])} ... {list(parquet_df.columns[-5:])}")

# Load original files for comparison
print("\nLoading original X_train.csv and X_val.csv...")
X_train = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'X_train.csv')
X_val = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'X_val.csv')
train_ids = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'train_ids.csv')
val_ids = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'val_ids.csv')

X_train.insert(0, 'SK_ID_CURR', train_ids['SK_ID_CURR'].values)
X_val.insert(0, 'SK_ID_CURR', val_ids['SK_ID_CURR'].values)

# Combine
X_combined = pd.concat([X_train, X_val], axis=0, ignore_index=True)
print(f"Combined original shape: {X_combined.shape}")

# Test with sample IDs
sample_ids = [100002, 100003, 100004]

print("\n" + "="*80)
print("COMPARING FEATURE VALUES")
print("="*80)

for sample_id in sample_ids:
    print(f"\nSample ID: {sample_id}")

    # Get from parquet
    parquet_row = parquet_df[parquet_df['SK_ID_CURR'] == sample_id]
    if len(parquet_row) == 0:
        print(f"  [ERROR] Not found in Parquet!")
        continue
    parquet_row = parquet_row.iloc[0]

    # Get from combined original
    original_row = X_combined[X_combined['SK_ID_CURR'] == sample_id]
    if len(original_row) == 0:
        print(f"  [ERROR] Not found in X_combined!")
        continue
    original_row = original_row.iloc[0]

    # Compare all features
    feature_cols = [c for c in parquet_df.columns if c != 'SK_ID_CURR']
    differences = []

    for feat in feature_cols:
        parquet_val = parquet_row[feat]
        original_val = original_row[feat]

        # Handle NaN
        if pd.isna(parquet_val) and pd.isna(original_val):
            continue

        if pd.isna(parquet_val) or pd.isna(original_val):
            differences.append((feat, original_val, parquet_val))
        elif abs(parquet_val - original_val) > 0.0001:
            differences.append((feat, original_val, parquet_val))

    if differences:
        print(f"  [ERROR] {len(differences)} feature differences found!")
        print(f"  First 5 differences:")
        for feat, orig, parq in differences[:5]:
            print(f"    {feat}: Original={orig}, Parquet={parq}")
    else:
        print(f"  [OK] All {len(feature_cols)} features match perfectly!")

# Now test predictions
print("\n" + "="*80)
print("TESTING PREDICTIONS WITH DIRECT FEATURE COMPARISON")
print("="*80)

import pickle

# Load model
with open(PROJECT_ROOT / 'models' / 'production_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load training predictions
train_preds = pd.read_csv(PROJECT_ROOT / 'results' / 'train_predictions.csv')
app_train = pd.read_csv(PROJECT_ROOT / 'data' / 'application_train.csv')
train_preds['SK_ID_CURR'] = app_train['SK_ID_CURR'].values

for sample_id in sample_ids:
    # Get features from X_combined (the TRUE features)
    original_row = X_combined[X_combined['SK_ID_CURR'] == sample_id]
    if len(original_row) == 0:
        continue

    # Drop SK_ID_CURR
    features = original_row.drop(columns=['SK_ID_CURR']).values

    # Make prediction
    pred_prob = model.predict_proba(features)[0, 1]

    # Get training prediction
    train_row = train_preds[train_preds['SK_ID_CURR'] == sample_id]
    if len(train_row) == 0:
        print(f"ID {sample_id}: Not found in training predictions")
        continue

    train_prob = train_row['PROBABILITY'].values[0]
    diff = abs(pred_prob - train_prob)

    match_str = "PERFECT MATCH!" if diff < 0.0001 else f"DIFF: {diff:.4f}"
    print(f"  ID {sample_id}: Training={train_prob:.6f}, Direct={pred_prob:.6f} - {match_str}")

print("\n" + "="*80)
