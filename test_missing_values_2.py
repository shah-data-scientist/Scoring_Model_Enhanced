"""Test 2: Verify missing values in multiple features are handled consistently."""
import pandas as pd
import numpy as np
import requests
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
END_USER_DIR = PROJECT_ROOT / 'data' / 'end_user_tests'
TEST_DIR = PROJECT_ROOT / 'data' / 'test_missing_multiple'
API_BASE_URL = "http://localhost:8000"

print("="*80)
print("TEST 2: Multiple Missing Values - Consistency Check")
print("="*80)

# Create test directory
TEST_DIR.mkdir(exist_ok=True)

# Load original data
orig_app = pd.read_csv(END_USER_DIR / 'application.csv')

print(f"\n[1/3] Creating clients with various missing value patterns...")

# Create 4 test cases:
# 1. Original (no missing)
# 2. Missing AMT_ANNUITY
# 3. Missing AMT_GOODS_PRICE
# 4. Missing both AMT_ANNUITY and AMT_GOODS_PRICE

test_cases = []

# Case 1: No missing (baseline)
case1 = orig_app.iloc[0].to_dict()
case1['SK_ID_CURR'] = 100001
test_cases.append(('100001', 'No missing', case1))

# Case 2: Missing AMT_ANNUITY
case2 = orig_app.iloc[0].to_dict()
case2['SK_ID_CURR'] = 100002
case2['AMT_ANNUITY'] = np.nan
test_cases.append(('100002', 'Missing AMT_ANNUITY', case2))

# Case 3: Missing AMT_GOODS_PRICE
case3 = orig_app.iloc[0].to_dict()
case3['SK_ID_CURR'] = 100003
case3['AMT_GOODS_PRICE'] = np.nan
test_cases.append(('100003', 'Missing AMT_GOODS_PRICE', case3))

# Case 4: Missing both
case4 = orig_app.iloc[0].to_dict()
case4['SK_ID_CURR'] = 100004
case4['AMT_ANNUITY'] = np.nan
case4['AMT_GOODS_PRICE'] = np.nan
test_cases.append(('100004', 'Missing both', case4))

# Create DataFrame
test_app = pd.DataFrame([case[2] for case in test_cases])
test_app.to_csv(TEST_DIR / 'application.csv', index=False)

print("  Test cases created:")
for id, desc, _ in test_cases:
    print(f"    {id}: {desc}")

# Create supporting files with dummy data
orig_bureau = pd.read_csv(END_USER_DIR / 'bureau.csv')
client_id = orig_app.iloc[0]['SK_ID_CURR']

# Bureau data - replicate for all test clients
bureau_rows = []
for test_id, _, _ in test_cases:
    client_bureau = orig_bureau[orig_bureau['SK_ID_CURR'] == client_id].copy()
    if len(client_bureau) > 0:
        client_bureau['SK_ID_CURR'] = int(test_id)
        bureau_rows.append(client_bureau)

if bureau_rows:
    combined_bureau = pd.concat(bureau_rows, ignore_index=True)
    combined_bureau.to_csv(TEST_DIR / 'bureau.csv', index=False)
    print(f"  ✓ bureau.csv: {len(combined_bureau)} rows")

# Other files - create minimal
csv_files = {
    'bureau_balance': END_USER_DIR / 'bureau_balance.csv',
    'credit_card_balance': END_USER_DIR / 'credit_card_balance.csv',
    'installments_payments': END_USER_DIR / 'installments_payments.csv',
    'POS_CASH_balance': END_USER_DIR / 'POS_CASH_balance.csv',
    'previous_application': END_USER_DIR / 'previous_application.csv'
}

for name, path in csv_files.items():
    if path.exists():
        df = pd.read_csv(path)
        
        if 'SK_ID_CURR' in df.columns:
            # Create dummy rows for each test client
            dummy_rows = []
            for test_id, _, _ in test_cases:
                dummy = {col: (int(test_id) if col == 'SK_ID_CURR' else 0) for col in df.columns}
                dummy_rows.append(dummy)
            combined = pd.DataFrame(dummy_rows)
        else:
            combined = df.head(100)
        
        output_name = 'POS_CASH_balance.csv' if name == 'POS_CASH_balance' else f'{name}.csv'
        combined.to_csv(TEST_DIR / output_name, index=False)

print(f"\n[2/3] Sending to API...")

api_files = {
    'application': 'application.csv',
    'bureau': 'bureau.csv',
    'bureau_balance': 'bureau_balance.csv',
    'credit_card_balance': 'credit_card_balance.csv',
    'installments_payments': 'installments_payments.csv',
    'pos_cash_balance': 'POS_CASH_balance.csv',
    'previous_application': 'previous_application.csv'
}

files_dict = {}
for api_name, csv_file in api_files.items():
    file_path = TEST_DIR / csv_file
    if file_path.exists():
        files_dict[api_name] = open(file_path, 'rb')

try:
    response = requests.post(
        f"{API_BASE_URL}/batch/predict",
        files=files_dict,
        timeout=30
    )
    
    if response.status_code == 200:
        predictions = response.json()['predictions']
        pred_df = pd.DataFrame(predictions)
        
        print(f"  ✓ Got {len(predictions)} predictions")
        
        print("\n[3/3] Results:")
        print("="*80)
        print(f"{'Client ID':<12} {'Pattern':<30} {'Prediction':<12} {'Probability':<15}")
        print("-"*80)
        
        for i, (test_id, desc, _) in enumerate(test_cases):
            row = pred_df[pred_df['sk_id_curr'] == int(test_id)].iloc[0]
            print(f"{test_id:<12} {desc:<30} {row['prediction']:<12} {row['probability']:<15.8f}")
        
        # Analysis
        print("\n" + "="*80)
        print("ANALYSIS")
        print("="*80)
        
        baseline_prob = pred_df[pred_df['sk_id_curr'] == 100001]['probability'].iloc[0]
        
        print(f"\nBaseline (no missing): {baseline_prob:.8f}")
        print("\nDifferences from baseline:")
        
        max_diff = 0
        for i, (test_id, desc, _) in enumerate(test_cases[1:], 1):  # Skip baseline
            prob = pred_df[pred_df['sk_id_curr'] == int(test_id)]['probability'].iloc[0]
            diff = abs(prob - baseline_prob)
            max_diff = max(max_diff, diff)
            print(f"  {desc:<30}: {diff:.8f}")
        
        print(f"\nMaximum difference: {max_diff:.8f}")
        
        if max_diff < 0.001:
            print("\n✓✓✓ EXCELLENT: All predictions within 0.1%")
            print("  Missing values imputed consistently with training medians")
        elif max_diff < 0.01:
            print("\n✓ GOOD: All predictions within 1%")
            print("  Minor differences likely due to feature interactions")
        elif max_diff < 0.05:
            print("\n⚠ ACCEPTABLE: Differences within 5%")
            print("  Some variation present but manageable")
        else:
            print("\n✗ CONCERN: Large differences detected")
            print("  Missing value handling may need investigation")
        
    else:
        print(f"API Error: {response.status_code}")
        print(response.text[:500])
        
finally:
    for f in files_dict.values():
        f.close()

print("\n" + "="*80)
print(f"Test data kept at: {TEST_DIR}")
