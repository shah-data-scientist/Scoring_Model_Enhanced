"""Diagnose feature order mismatch."""
import pickle
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent

# Load model
with open(PROJECT_ROOT / 'models' / 'production_model.pkl', 'rb') as f:
    model = pickle.load(f)

model_features = model.feature_name_
print(f"Model expects {len(model_features)} features")
print(f"First 10 model features: {model_features[:10]}")
print(f"Last 10 model features: {model_features[-10:]}\n")

# Load precomputed features
parquet_df = pd.read_parquet(PROJECT_ROOT / 'data' / 'processed' / 'precomputed_features.parquet')
print(f"Parquet has {len(parquet_df.columns)} columns")
print(f"Parquet columns: {list(parquet_df.columns)[:10]} ... {list(parquet_df.columns)[-10:]}\n")

# Remove SK_ID_CURR if present
if 'SK_ID_CURR' in parquet_df.columns:
    feature_cols = [c for c in parquet_df.columns if c != 'SK_ID_CURR']
    print(f"After removing SK_ID_CURR: {len(feature_cols)} features\n")
else:
    feature_cols = list(parquet_df.columns)

# Check if feature names match
if len(model_features) != len(feature_cols):
    print("[ERROR] Feature count mismatch!")
    print(f"  Model expects: {len(model_features)}")
    print(f"  Parquet has: {len(feature_cols)}")
else:
    print(f"[OK] Feature count matches: {len(model_features)}")

# Check feature order
mismatches = []
for i, (model_feat, parquet_feat) in enumerate(zip(model_features, feature_cols)):
    if model_feat != parquet_feat:
        mismatches.append((i, model_feat, parquet_feat))

if mismatches:
    print(f"\n[ERROR] Feature order mismatch at {len(mismatches)} positions:")
    for i, model_feat, parquet_feat in mismatches[:20]:
        print(f"  Position {i}: Model='{model_feat}', Parquet='{parquet_feat}'")
else:
    print("\n[OK] Feature names and order match perfectly!")

# Check for missing features
model_set = set(model_features)
parquet_set = set(feature_cols)

missing_in_parquet = model_set - parquet_set
extra_in_parquet = parquet_set - model_set

if missing_in_parquet:
    print(f"\n[ERROR] {len(missing_in_parquet)} features missing in Parquet:")
    print(f"  {list(missing_in_parquet)[:10]}")

if extra_in_parquet:
    print(f"\n[WARNING] {len(extra_in_parquet)} extra features in Parquet:")
    print(f"  {list(extra_in_parquet)[:10]}")
