"""Investigate why anonymized predictions differ."""
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
ORIG_DIR = PROJECT_ROOT / 'data' / 'end_user_tests'
ANON_DIR = PROJECT_ROOT / 'data' / 'end_user_tests_anonymized'

print("Investigating prediction differences...\n")

# Compare application.csv files
orig_app = pd.read_csv(ORIG_DIR / 'application.csv')
anon_app = pd.read_csv(ANON_DIR / 'application.csv')

print(f"Original application.csv: {len(orig_app)} rows")
print(f"  SK_ID_CURR values: {sorted(orig_app['SK_ID_CURR'].values)[:5]}...")
print(f"  Column order: {list(orig_app.columns[:10])}...")

print(f"\nAnonymized application.csv: {len(anon_app)} rows")
print(f"  SK_ID_CURR values: {sorted(anon_app['SK_ID_CURR'].values)[:5]}...")
print(f"  Column order: {list(anon_app.columns[:10])}...")

# Check if rows are in same order
print(f"\n{'Orig ID':<12} {'Anon ID':<12} {'AMT_INCOME':<15} {'AMT_CREDIT':<15} {'Match'}")
print("-"*70)

for i in range(min(10, len(orig_app))):
    orig_id = orig_app.iloc[i]['SK_ID_CURR']
    anon_id = anon_app.iloc[i]['SK_ID_CURR']
    
    orig_income = orig_app.iloc[i]['AMT_INCOME_TOTAL']
    anon_income = anon_app.iloc[i]['AMT_INCOME_TOTAL']
    
    orig_credit = orig_app.iloc[i]['AMT_CREDIT']
    anon_credit = anon_app.iloc[i]['AMT_CREDIT']
    
    match = (orig_income == anon_income) and (orig_credit == anon_credit)
    status = "✓" if match else "✗"
    
    print(f"{orig_id:<12} {anon_id:<12} {orig_income:<15.2f} {anon_credit:<15.2f} {status}")

# Check bureau.csv relationships
print("\n\nChecking bureau.csv relationships...")
orig_bureau = pd.read_csv(ORIG_DIR / 'bureau.csv')
anon_bureau = pd.read_csv(ANON_DIR / 'bureau.csv')

print(f"\nOriginal bureau: First ID has {len(orig_bureau[orig_bureau['SK_ID_CURR']==orig_app.iloc[0]['SK_ID_CURR']])} bureau records")
print(f"Anonymized bureau: First ID has {len(anon_bureau[anon_bureau['SK_ID_CURR']==anon_app.iloc[0]['SK_ID_CURR']])} bureau records")

# Check if bureau data preserved correctly
first_orig_id = orig_app.iloc[0]['SK_ID_CURR']
first_anon_id = anon_app.iloc[0]['SK_ID_CURR']

orig_bureau_subset = orig_bureau[orig_bureau['SK_ID_CURR'] == first_orig_id]
anon_bureau_subset = anon_bureau[anon_bureau['SK_ID_CURR'] == first_anon_id]

if len(orig_bureau_subset) > 0 and len(anon_bureau_subset) > 0:
    print(f"\nFirst client's bureau records comparison:")
    print(f"  Original: {len(orig_bureau_subset)} records, CREDIT_TYPE={orig_bureau_subset.iloc[0]['CREDIT_TYPE']}")
    print(f"  Anonymized: {len(anon_bureau_subset)} records, CREDIT_TYPE={anon_bureau_subset.iloc[0]['CREDIT_TYPE']}")
