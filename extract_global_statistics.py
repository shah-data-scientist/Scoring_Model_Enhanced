"""Extract and save global statistics for dataset-level features.

This ensures the API uses the same group means as training, making predictions
consistent regardless of SK_ID_CURR values.
"""
import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_FILE = DATA_DIR / 'processed' / 'global_statistics.json'

print("="*80)
print("EXTRACTING GLOBAL STATISTICS FROM TRAINING DATA")
print("="*80)

# Load training data
print("\n[1/3] Loading training data...")
train_file = DATA_DIR / 'application_train.csv'

if not train_file.exists():
    print(f"ERROR: Training file not found: {train_file}")
    exit(1)

train_df = pd.read_csv(train_file)
print(f"  ✓ Loaded {len(train_df):,} training records")

# Calculate global statistics for dataset-level features
print("\n[2/3] Computing global statistics...")

global_stats = {}

# 1. Income by occupation type
if 'OCCUPATION_TYPE' in train_df.columns and 'AMT_INCOME_TOTAL' in train_df.columns:
    occupation_income = train_df.groupby('OCCUPATION_TYPE')['AMT_INCOME_TOTAL'].mean().to_dict()
    global_stats['OCCUPATION_TYPE_INCOME_MEAN'] = occupation_income
    print(f"  ✓ Computed income means for {len(occupation_income)} occupation types")

# 2. Income by organization type
if 'ORGANIZATION_TYPE' in train_df.columns and 'AMT_INCOME_TOTAL' in train_df.columns:
    org_income = train_df.groupby('ORGANIZATION_TYPE')['AMT_INCOME_TOTAL'].mean().to_dict()
    global_stats['ORGANIZATION_TYPE_INCOME_MEAN'] = org_income
    print(f"  ✓ Computed income means for {len(org_income)} organization types")

# 3. Income by education type
if 'NAME_EDUCATION_TYPE' in train_df.columns and 'AMT_INCOME_TOTAL' in train_df.columns:
    edu_income = train_df.groupby('NAME_EDUCATION_TYPE')['AMT_INCOME_TOTAL'].mean().to_dict()
    global_stats['NAME_EDUCATION_TYPE_INCOME_MEAN'] = edu_income
    print(f"  ✓ Computed income means for {len(edu_income)} education types")

# 4. Credit by contract type
if 'NAME_CONTRACT_TYPE' in train_df.columns and 'AMT_CREDIT' in train_df.columns:
    contract_credit = train_df.groupby('NAME_CONTRACT_TYPE')['AMT_CREDIT'].mean().to_dict()
    global_stats['NAME_CONTRACT_TYPE_CREDIT_MEAN'] = contract_credit
    print(f"  ✓ Computed credit means for {len(contract_credit)} contract types")

# Save to file
print("\n[3/3] Saving global statistics...")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, 'w') as f:
    json.dump(global_stats, f, indent=2)

print(f"  ✓ Saved to: {OUTPUT_FILE}")

# Show sample statistics
print("\n" + "="*80)
print("SAMPLE STATISTICS")
print("="*80)

if 'OCCUPATION_TYPE_INCOME_MEAN' in global_stats:
    print("\nIncome by Occupation (sample):")
    for occ, mean in list(global_stats['OCCUPATION_TYPE_INCOME_MEAN'].items())[:5]:
        print(f"  {occ}: {mean:,.0f}")

if 'NAME_CONTRACT_TYPE_CREDIT_MEAN' in global_stats:
    print("\nCredit by Contract Type:")
    for contract, mean in global_stats['NAME_CONTRACT_TYPE_CREDIT_MEAN'].items():
        print(f"  {contract}: {mean:,.0f}")

print("\n" + "="*80)
print("✓ Global statistics extracted successfully!")
print("="*80)
print("\nThese statistics will be used by the API to ensure consistent")
print("dataset-level features regardless of SK_ID_CURR values.")
