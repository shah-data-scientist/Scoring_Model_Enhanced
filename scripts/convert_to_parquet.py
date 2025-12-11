"""
Convert X_train.csv to Parquet format for 10-100x faster loading.

Parquet is a columnar storage format optimized for:
- Fast reading (especially for ML features)
- Efficient compression
- Type preservation
"""
import pandas as pd
from pathlib import Path
import time

print("="*80)
print("CONVERTING X_TRAIN.CSV TO PARQUET FORMAT")
print("="*80)

# Paths
data_dir = Path("data/processed")
csv_path = data_dir / "X_train.csv"
parquet_path = data_dir / "X_train.parquet"

# Check if CSV exists
if not csv_path.exists():
    print(f"ERROR: {csv_path} not found!")
    exit(1)

print(f"\nInput:  {csv_path}")
print(f"Output: {parquet_path}")

# Load CSV and measure time
print("\n[1/3] Loading CSV...")
start = time.time()
df = pd.read_csv(csv_path)
csv_time = time.time() - start
print(f"  Loaded {len(df):,} rows x {len(df.columns)} columns in {csv_time:.2f}s")

# Save as Parquet
print("\n[2/3] Saving as Parquet...")
start = time.time()
df.to_parquet(parquet_path, engine='pyarrow', compression='snappy', index=False)
parquet_time = time.time() - start
print(f"  Saved in {parquet_time:.2f}s")

# Test loading speed
print("\n[3/3] Testing Parquet load speed...")
start = time.time()
df_test = pd.read_parquet(parquet_path)
parquet_load_time = time.time() - start
print(f"  Loaded in {parquet_load_time:.2f}s")

# Compare file sizes
csv_size_mb = csv_path.stat().st_size / (1024 * 1024)
parquet_size_mb = parquet_path.stat().st_size / (1024 * 1024)

# Results
print("\n" + "="*80)
print("CONVERSION COMPLETE")
print("="*80)
print(f"\nFile sizes:")
print(f"  CSV:     {csv_size_mb:,.1f} MB")
print(f"  Parquet: {parquet_size_mb:,.1f} MB ({parquet_size_mb/csv_size_mb*100:.1f}% of original)")

print(f"\nLoad times:")
print(f"  CSV:     {csv_time:.2f}s")
print(f"  Parquet: {parquet_load_time:.2f}s ({csv_time/parquet_load_time:.1f}x faster!)")

print(f"\n✅ Parquet file ready at: {parquet_path}")
print("="*80)
