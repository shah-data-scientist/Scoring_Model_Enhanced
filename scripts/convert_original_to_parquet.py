"""Convert original X_train.csv to Parquet format."""
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent

print("="*80)
print("CONVERTING ORIGINAL X_TRAIN.CSV TO PARQUET")
print("="*80)

# Paths
csv_path = PROJECT_ROOT / 'data' / 'processed' / 'X_train.csv'
train_ids_path = PROJECT_ROOT / 'data' / 'processed' / 'train_ids.csv'
parquet_path = PROJECT_ROOT / 'data' / 'processed' / 'precomputed_features.parquet'

print(f"\nInput: {csv_path}")
print(f"Output: {parquet_path}")

# Load CSV and measure time
print("\n[1/4] Loading X_train.csv...")
start = time.time()
df = pd.read_csv(csv_path)
csv_time = time.time() - start
print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns in {csv_time:.1f} seconds")

# Load SK_ID_CURR mapping
print("\n[2/4] Loading train_ids.csv...")
train_ids = pd.read_csv(train_ids_path)
print(f"  Loaded {len(train_ids):,} IDs")

# Add SK_ID_CURR to dataframe
print("\n[3/4] Adding SK_ID_CURR column...")
df.insert(0, 'SK_ID_CURR', train_ids['SK_ID_CURR'].values)
print(f"  Final shape: {df.shape}")

# Save to Parquet
print("\n[4/4] Saving to Parquet...")
start = time.time()
df.to_parquet(parquet_path, index=False, compression='snappy')
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

csv_size = os.path.getsize(csv_path) / (1024**2)
parquet_size = os.path.getsize(parquet_path) / (1024**2)

print("\n" + "="*80)
print("CONVERSION SUMMARY")
print("="*80)
print(f"CSV size: {csv_size:.1f} MB")
print(f"Parquet size: {parquet_size:.1f} MB")
print(f"Size reduction: {csv_size / parquet_size:.1f}x smaller")
print(f"\nCSV load time: {csv_time:.1f} seconds")
print(f"Parquet load time: {load_time:.1f} seconds")
print(f"Speed improvement: {csv_time / load_time:.1f}x faster")
print("\n[SUCCESS] Original X_train.csv converted to Parquet!")
print("="*80)
