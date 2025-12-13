"""Demonstrate dataset-level features causing prediction differences."""
import pandas as pd
import numpy as np

print("="*80)
print("WHY AGGREGATIONS DIFFER: Dataset-Level vs Per-Client Features")
print("="*80)

# Example data
data = {
    'SK_ID_CURR': [111761, 222222, 333333, 444444, 555555],
    'OCCUPATION_TYPE': ['Laborers', 'Laborers', 'Managers', 'Managers', 'Laborers'],
    'AMT_INCOME_TOTAL': [100000, 150000, 300000, 400000, 120000]
}

print("\n[TRAINING DATA - 5 clients]")
train_df = pd.DataFrame(data)
print(train_df)

# Compute dataset-level features (TRAINING)
print("\n[TRAINING: Computing group means across ALL 5 clients]")
train_df['group_mean'] = train_df.groupby('OCCUPATION_TYPE')['AMT_INCOME_TOTAL'].transform('mean')
train_df['INCOME_VS_OCCUPATION'] = train_df['AMT_INCOME_TOTAL'] / (train_df['group_mean'] + 1e-5)

print(train_df[['SK_ID_CURR', 'OCCUPATION_TYPE', 'AMT_INCOME_TOTAL', 'group_mean', 'INCOME_VS_OCCUPATION']])

print("\nGroup statistics (TRAINING):")
print("  Laborers: 3 clients, mean income = 123,333")
print("  Managers: 2 clients, mean income = 350,000")

# Now simulate API with ONLY client 111761
print("\n" + "="*80)
print("[API REQUEST - ONLY 1 client uploaded]")
api_data = {
    'SK_ID_CURR': [111761],
    'OCCUPATION_TYPE': ['Laborers'],
    'AMT_INCOME_TOTAL': [100000]
}
api_df = pd.DataFrame(api_data)
print(api_df)

# Compute with ONLY this client's data
print("\n[API: Computing group means from uploaded data ONLY]")
api_df['group_mean'] = api_df.groupby('OCCUPATION_TYPE')['AMT_INCOME_TOTAL'].transform('mean')
api_df['INCOME_VS_OCCUPATION'] = api_df['AMT_INCOME_TOTAL'] / (api_df['group_mean'] + 1e-5)

print(api_df[['SK_ID_CURR', 'OCCUPATION_TYPE', 'AMT_INCOME_TOTAL', 'group_mean', 'INCOME_VS_OCCUPATION']])

print("\nGroup statistics (API - single client):")
print("  Laborers: 1 client, mean income = 100,000")

# Compare
print("\n" + "="*80)
print("COMPARISON FOR CLIENT 111761")
print("="*80)

train_value = train_df[train_df['SK_ID_CURR'] == 111761]['INCOME_VS_OCCUPATION'].iloc[0]
api_value = api_df['INCOME_VS_OCCUPATION'].iloc[0]

print(f"\nFeature: INCOME_VS_OCCUPATION_TYPE_MEAN")
print(f"  Training (with 5 clients): {train_value:.6f}")
print(f"  API (only this client):    {api_value:.6f}")
print(f"  Difference:                {abs(train_value - api_value):.6f}")

print("\n" + "="*80)
print("EXPLANATION")
print("="*80)
print("""
This is a DATASET-LEVEL feature that depends on OTHER clients in the dataset:

Training computation:
  INCOME_VS_OCCUPATION = client_income / mean_income_for_occupation_across_all_clients
  
  For client 111761 (Laborer, 100K income):
    mean_income_laborers = (100K + 150K + 120K) / 3 = 123,333
    INCOME_VS_OCCUPATION = 100,000 / 123,333 = 0.811

API computation (only 1 client uploaded):
  For client 111761 (Laborer, 100K income):
    mean_income_laborers = 100,000 / 1 = 100,000
    INCOME_VS_OCCUPATION = 100,000 / 100,000 = 1.000

**This feature fundamentally CANNOT be computed correctly without the full dataset!**

When you anonymize SK_ID_CURR:
- Known IDs: Use TRAINING features (computed with full dataset context)
- New IDs:   Use LIVE features (computed only from uploaded clients)

These will ALWAYS differ because the group statistics are different!
""")

print("\n" + "="*80)
print("REAL EXAMPLE FROM YOUR DATA")
print("="*80)
print("""
Your test has 50 clients with various occupations:
- Training: Group means computed across ~300K clients
- API (50 clients): Group means computed across ONLY 50 clients

Example:
  - Training dataset: 10,000 "Laborers" → mean income = 150K
  - Your 50 clients: 5 "Laborers" → mean income = 180K
  
Result: INCOME_VS_OCCUPATION differs between cached and live computation!

This affects ~5 features:
  1. INCOME_VS_OCCUPATION_TYPE_MEAN
  2. INCOME_VS_ORGANIZATION_TYPE_MEAN
  3. INCOME_VS_NAME_EDUCATION_TYPE_MEAN
  4. CREDIT_VS_CONTRACT_MEAN
  5. Similar ratio features
""")

print("\n" + "="*80)
print("SOLUTION")
print("="*80)
print("""
Option 1: Remove dataset-level features (use only per-client aggregations)
Option 2: Store global statistics (occupation means from training) and use them for API
Option 3: Disable precomputed cache (use_precomputed=False) → all predictions consistent but slower
Option 4: Accept the difference (production will be slightly different from training)

The cache exists for SPEED, but creates this CONSISTENCY issue.
""")
