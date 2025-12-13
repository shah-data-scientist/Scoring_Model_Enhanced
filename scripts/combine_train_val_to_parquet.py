"""Combine X_train.csv and X_val.csv (the actual training data) and convert to Parquet."""
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent

print("="*80)
print("COMBINING X_TRAIN + X_VAL AND CONVERTING TO PARQUET")
print("="*80)

# Paths
train_csv = PROJECT_ROOT / 'data' / 'processed' / 'X_train.csv'
val_csv = PROJECT_ROOT / 'data' / 'processed' / 'X_val.csv'
train_ids_path = PROJECT_ROOT / 'data' / 'processed' / 'train_ids.csv'
val_ids_path = PROJECT_ROOT / 'data' / 'processed' / 'val_ids.csv'
parquet_path = PROJECT_ROOT / 'data' / 'processed' / 'precomputed_features.parquet'

print("\nInput files:")
print(f"  - {train_csv}")
print(f"  - {val_csv}")
print(f"\nOutput: {parquet_path}")

# Load X_train
print("\n[1/6] Loading X_train.csv...")
start = time.time()
X_train = pd.read_csv(train_csv)
train_time = time.time() - start
print(f"  Loaded {len(X_train):,} rows × {len(X_train.columns)} columns in {train_time:.1f} seconds")

# Load X_val
print("\n[2/6] Loading X_val.csv...")
start = time.time()
X_val = pd.read_csv(val_csv)
val_time = time.time() - start
print(f"  Loaded {len(X_val):,} rows × {len(X_val.columns)} columns in {val_time:.1f} seconds")

# Load IDs
print("\n[3/6] Loading train_ids.csv and val_ids.csv...")
train_ids = pd.read_csv(train_ids_path)
val_ids = pd.read_csv(val_ids_path)
print(f"  Train IDs: {len(train_ids):,}")
print(f"  Val IDs: {len(val_ids):,}")

# Add SK_ID_CURR to both dataframes
print("\n[4/6] Adding SK_ID_CURR columns...")
X_train.insert(0, 'SK_ID_CURR', train_ids['SK_ID_CURR'].values)
X_val.insert(0, 'SK_ID_CURR', val_ids['SK_ID_CURR'].values)
print(f"  X_train shape: {X_train.shape}")
print(f"  X_val shape: {X_val.shape}")

# Combine both datasets
print("\n[5/6] Combining X_train and X_val...")
X_combined = pd.concat([X_train, X_val], axis=0, ignore_index=True)
print(f"  Combined shape: {X_combined.shape}")
print(f"  Total applications: {len(X_combined):,}")

# Verify no duplicate IDs
duplicates = X_combined['SK_ID_CURR'].duplicated().sum()
if duplicates > 0:
    print(f"  [WARNING] Found {duplicates} duplicate SK_ID_CURR values!")
else:
    print("  [OK] No duplicate SK_ID_CURR values")

# Save to Parquet
print("\n[6/6] Saving to Parquet...")
start = time.time()
X_combined.to_parquet(parquet_path, index=False, compression='snappy')
parquet_time = time.time() - start
print(f"  Saved in {parquet_time:.1f} seconds")

# Verify by loading Parquet
print("\n[VERIFY] Testing Parquet load speed...")
start = time.time()
test_df = pd.read_parquet(parquet_path)
load_time = time.time() - start
print(f"  Loaded {len(test_df):,} rows × {len(test_df.columns)} columns in {load_time:.1f} seconds")

# File size comparison
import os

train_size = os.path.getsize(train_csv) / (1024**2)
val_size = os.path.getsize(val_csv) / (1024**2)
parquet_size = os.path.getsize(parquet_path) / (1024**2)
total_csv_size = train_size + val_size

print("\n" + "="*80)
print("CONVERSION SUMMARY")
print("="*80)
print(f"X_train.csv size: {train_size:.1f} MB")
print(f"X_val.csv size: {val_size:.1f} MB")
print(f"Total CSV size: {total_csv_size:.1f} MB")
print(f"Parquet size: {parquet_size:.1f} MB")
print(f"Size reduction: {total_csv_size / parquet_size:.1f}x smaller")
print(f"\nTotal CSV load time: {train_time + val_time:.1f} seconds")
print(f"Parquet load time: {load_time:.1f} seconds")
print(f"Speed improvement: {(train_time + val_time) / load_time:.1f}x faster")
print(f"\nTotal training applications: {len(X_combined):,}")
print(f"  - From X_train: {len(X_train):,}")
print(f"  - From X_val: {len(X_val):,}")
print("\n[SUCCESS] Combined training data converted to Parquet!")
print("="*80)
