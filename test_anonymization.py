"""Test that anonymizing SK_ID_CURR doesn't affect predictions."""
import pandas as pd
import numpy as np
import requests
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).parent
END_USER_DIR = PROJECT_ROOT / 'data' / 'end_user_tests'
ANONYMIZED_DIR = PROJECT_ROOT / 'data' / 'end_user_tests_anonymized'
API_BASE_URL = "http://localhost:8000"

print("="*70)
print("ANONYMIZATION TEST: SK_ID_CURR Impact on Predictions")
print("="*70)

# Step 1: Create anonymized copies
print("\n[1/4] Creating anonymized test data...")
ANONYMIZED_DIR.mkdir(exist_ok=True)

# Read original application.csv to get SK_ID_CURRs
app_df = pd.read_csv(END_USER_DIR / 'application.csv')
original_ids = app_df['SK_ID_CURR'].unique()
print(f"  Original IDs: {sorted(original_ids)[:5]}... ({len(original_ids)} total)")

# Create mapping: original_id -> anonymized_id
np.random.seed(42)
anonymized_ids = np.random.randint(900000, 999999, size=len(original_ids))
id_mapping = dict(zip(original_ids, anonymized_ids))
print(f"  Anonymized IDs: {sorted(anonymized_ids)[:5]}... ({len(anonymized_ids)} total)")

# Anonymize all CSV files
csv_files = {
    'application': 'application.csv',
    'bureau': 'bureau.csv', 
    'bureau_balance': 'bureau_balance.csv',
    'credit_card_balance': 'credit_card_balance.csv',
    'installments_payments': 'installments_payments.csv',
    'pos_cash_balance': 'POS_CASH_balance.csv',
    'previous_application': 'previous_application.csv'
}

for api_name, csv_file in csv_files.items():
    source_path = END_USER_DIR / csv_file
    if not source_path.exists():
        print(f"  ⚠ Skipping {csv_file} (not found)")
        continue
    
    df = pd.read_csv(source_path)
    print(f"  Processing {csv_file}: {len(df)} rows")
    
    # Replace SK_ID_CURR with anonymized IDs
    if 'SK_ID_CURR' in df.columns:
        df['SK_ID_CURR'] = df['SK_ID_CURR'].map(id_mapping)
        anonymized_count = df['SK_ID_CURR'].notna().sum()
        print(f"    Anonymized {anonymized_count} SK_ID_CURR values")
    
    # Save anonymized file
    dest_path = ANONYMIZED_DIR / csv_file
    df.to_csv(dest_path, index=False)
    print(f"    ✓ Saved to {dest_path}")

print(f"\n  ✓ Anonymized data saved to: {ANONYMIZED_DIR}")

# Step 2: Get predictions for original data
print("\n[2/4] Getting predictions for ORIGINAL data...")
files_dict = {}
for api_name, csv_file in csv_files.items():
    file_path = END_USER_DIR / csv_file
    if file_path.exists():
        files_dict[api_name] = open(file_path, 'rb')

try:
    response = requests.post(
        f"{API_BASE_URL}/batch/predict",
        files=files_dict,
        timeout=30
    )
    
    if response.status_code == 200:
        original_predictions = response.json()['predictions']
        print(f"  ✓ Got {len(original_predictions)} predictions")
        original_df = pd.DataFrame(original_predictions)
        print(f"    Sample: ID={original_df.iloc[0]['sk_id_curr']}, "
              f"Pred={original_df.iloc[0]['prediction']}, "
              f"Prob={original_df.iloc[0]['probability']:.4f}")
    else:
        print(f"  ✗ API Error: {response.status_code}")
        print(f"    {response.text[:200]}")
        exit(1)
finally:
    for f in files_dict.values():
        f.close()

# Step 3: Get predictions for anonymized data
print("\n[3/4] Getting predictions for ANONYMIZED data...")
files_dict = {}
for api_name, csv_file in csv_files.items():
    file_path = ANONYMIZED_DIR / csv_file
    if file_path.exists():
        files_dict[api_name] = open(file_path, 'rb')

try:
    response = requests.post(
        f"{API_BASE_URL}/batch/predict",
        files=files_dict,
        timeout=30
    )
    
    if response.status_code == 200:
        anonymized_predictions = response.json()['predictions']
        print(f"  ✓ Got {len(anonymized_predictions)} predictions")
        anonymized_df = pd.DataFrame(anonymized_predictions)
        print(f"    Sample: ID={anonymized_df.iloc[0]['sk_id_curr']}, "
              f"Pred={anonymized_df.iloc[0]['prediction']}, "
              f"Prob={anonymized_df.iloc[0]['probability']:.4f}")
    else:
        print(f"  ✗ API Error: {response.status_code}")
        print(f"    {response.text[:200]}")
        exit(1)
finally:
    for f in files_dict.values():
        f.close()

# Step 4: Compare predictions
print("\n[4/4] Comparing predictions...")
print("-"*70)

# Create reverse mapping for comparison
reverse_mapping = {v: k for k, v in id_mapping.items()}

# Sort both dataframes by original ID for comparison
original_df['original_id'] = original_df['sk_id_curr']
anonymized_df['original_id'] = anonymized_df['sk_id_curr'].map(reverse_mapping)

original_df = original_df.sort_values('original_id').reset_index(drop=True)
anonymized_df = anonymized_df.sort_values('original_id').reset_index(drop=True)

# Compare predictions row by row
print(f"\n{'Original ID':<12} {'Anon ID':<10} {'Orig Pred':<10} {'Anon Pred':<10} {'Orig Prob':<12} {'Anon Prob':<12} {'Match':<8}")
print("-"*70)

all_match = True
prob_diffs = []

for i in range(len(original_df)):
    orig_row = original_df.iloc[i]
    anon_row = anonymized_df.iloc[i]
    
    orig_id = orig_row['original_id']
    anon_id = anon_row['sk_id_curr']
    
    orig_pred = orig_row['prediction']
    anon_pred = anon_row['prediction']
    
    orig_prob = orig_row['probability']
    anon_prob = anon_row['probability']
    
    prob_diff = abs(orig_prob - anon_prob)
    prob_diffs.append(prob_diff)
    
    match = (orig_pred == anon_pred) and (prob_diff < 1e-10)
    status = "✓ YES" if match else "✗ NO"
    
    if not match:
        all_match = False
    
    print(f"{orig_id:<12} {anon_id:<10} {orig_pred:<10} {anon_pred:<10} "
          f"{orig_prob:<12.6f} {anon_prob:<12.6f} {status:<8}")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\n  Total records:           {len(original_df)}")
print(f"  Predictions match:       {sum(original_df['prediction'] == anonymized_df['prediction'])}/{len(original_df)}")
print(f"  Max probability diff:    {max(prob_diffs):.10f}")
print(f"  Mean probability diff:   {np.mean(prob_diffs):.10f}")

if all_match:
    print(f"\n  ✓✓✓ PERFECT MATCH! ✓✓✓")
    print(f"  SK_ID_CURR has NO impact on predictions (as expected)")
    print(f"  The model correctly ignores the identifier column")
else:
    print(f"\n  ✗✗✗ MISMATCH DETECTED! ✗✗✗")
    print(f"  SK_ID_CURR may be impacting predictions (unexpected!)")

print("\n" + "="*70)

# Cleanup option
print(f"\nAnonymized data kept at: {ANONYMIZED_DIR}")
print(f"To delete: shutil.rmtree('{ANONYMIZED_DIR}')")
