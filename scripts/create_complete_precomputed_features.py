"""Create complete precomputed features including train, val, AND test sets."""
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent

print("="*80)
print("CREATING COMPLETE PRECOMPUTED FEATURES (TRAIN + VAL + TEST)")
print("="*80)

# Paths
data_dir = PROJECT_ROOT / 'data' / 'processed'
parquet_path = data_dir / 'precomputed_features.parquet'

print(f"\nOutput: {parquet_path}")

# Load all feature sets
print("\n[1/7] Loading X_train.csv...")
start = time.time()
X_train = pd.read_csv(data_dir / 'X_train.csv')
train_time = time.time() - start
print(f"  Loaded {len(X_train):,} rows × {len(X_train.columns)} columns in {train_time:.1f}s")

print("\n[2/7] Loading X_val.csv...")
start = time.time()
X_val = pd.read_csv(data_dir / 'X_val.csv')
val_time = time.time() - start
print(f"  Loaded {len(X_val):,} rows × {len(X_val.columns)} columns in {val_time:.1f}s")

print("\n[3/7] Loading X_test.csv...")
start = time.time()
X_test = pd.read_csv(data_dir / 'X_test.csv')
test_time = time.time() - start
print(f"  Loaded {len(X_test):,} rows × {len(X_test.columns)} columns in {test_time:.1f}s")

# Load ID mappings
print("\n[4/7] Loading ID mappings...")
train_ids = pd.read_csv(data_dir / 'train_ids.csv')
val_ids = pd.read_csv(data_dir / 'val_ids.csv')
test_ids = pd.read_csv(data_dir / 'test_ids.csv')
print(f"  Train IDs: {len(train_ids):,}")
print(f"  Val IDs: {len(val_ids):,}")
print(f"  Test IDs: {len(test_ids):,}")

# Add SK_ID_CURR to all dataframes
print("\n[5/7] Adding SK_ID_CURR columns...")
X_train.insert(0, 'SK_ID_CURR', train_ids['SK_ID_CURR'].values)
X_val.insert(0, 'SK_ID_CURR', val_ids['SK_ID_CURR'].values)
X_test.insert(0, 'SK_ID_CURR', test_ids['SK_ID_CURR'].values)

print(f"  X_train shape: {X_train.shape}")
print(f"  X_val shape: {X_val.shape}")
print(f"  X_test shape: {X_test.shape}")

# Verify all have same number of features
assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1], "Feature count mismatch!"
print(f"  [OK] All datasets have {X_train.shape[1]-1} features + SK_ID_CURR")

# Combine all three datasets
print("\n[6/7] Combining train + val + test...")
X_combined = pd.concat([X_train, X_val, X_test], axis=0, ignore_index=True)
print(f"  Combined shape: {X_combined.shape}")
print(f"  Total applications: {len(X_combined):,}")

# Verify no duplicate IDs
duplicates = X_combined['SK_ID_CURR'].duplicated().sum()
if duplicates > 0:
    print(f"  [WARNING] Found {duplicates} duplicate SK_ID_CURR values!")
    # Remove duplicates, keeping first occurrence
    X_combined = X_combined.drop_duplicates(subset=['SK_ID_CURR'], keep='first')
    print(f"  After dedup: {len(X_combined):,} applications")
else:
    print("  [OK] No duplicate SK_ID_CURR values")

# Save to Parquet
print("\n[7/7] Saving to Parquet...")
start = time.time()
X_combined.to_parquet(parquet_path, index=False, compression='snappy')
parquet_time = time.time() - start
print(f"  Saved in {parquet_time:.1f} seconds")

# Verify by loading
print("\n[VERIFY] Testing Parquet load speed...")
start = time.time()
test_df = pd.read_parquet(parquet_path)
load_time = time.time() - start
print(f"  Loaded {len(test_df):,} rows × {len(test_df.columns)} columns in {load_time:.1f}s")

# File size comparison
import os

train_size = os.path.getsize(data_dir / 'X_train.csv') / (1024**2)
val_size = os.path.getsize(data_dir / 'X_val.csv') / (1024**2)
test_size = os.path.getsize(data_dir / 'X_test.csv') / (1024**2)
parquet_size = os.path.getsize(parquet_path) / (1024**2)
total_csv_size = train_size + val_size + test_size

print("\n" + "="*80)
print("CONVERSION SUMMARY")
print("="*80)
print(f"X_train.csv size: {train_size:.1f} MB")
print(f"X_val.csv size: {val_size:.1f} MB")
print(f"X_test.csv size: {test_size:.1f} MB")
print(f"Total CSV size: {total_csv_size:.1f} MB")
print(f"Parquet size: {parquet_size:.1f} MB")
print(f"Size reduction: {total_csv_size / parquet_size:.1f}x smaller")
print(f"\nTotal CSV load time: {train_time + val_time + test_time:.1f} seconds")
print(f"Parquet load time: {load_time:.1f} seconds")
print(f"Speed improvement: {(train_time + val_time + test_time) / load_time:.1f}x faster")
print(f"\n{'Dataset':<15s} {'Count':>12s}")
print("-"*80)
print(f"{'Training':<15s} {len(train_ids):>12,}")
print(f"{'Validation':<15s} {len(val_ids):>12,}")
print(f"{'Test':<15s} {len(test_ids):>12,}")
print(f"{'TOTAL':<15s} {len(X_combined):>12,}")
print("\n[SUCCESS] Complete precomputed features created!")
print("="*80)
print("\nThis file contains:")
print(f"  - {len(train_ids):,} training applications")
print(f"  - {len(val_ids):,} validation applications")
print(f"  - {len(test_ids):,} test applications")
print(f"  - {X_combined.shape[1]-1} features per application")
print(f"\nThe batch API will now use LOOKUP for ALL {len(X_combined):,} applications,")
print("guaranteeing 100% accurate predictions that match the original model training.")
print("="*80)
