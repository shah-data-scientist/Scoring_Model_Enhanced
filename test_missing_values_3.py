"""Test 3: Extreme case - client with many missing values."""
import pandas as pd
import numpy as np
import requests
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
END_USER_DIR = PROJECT_ROOT / 'data' / 'end_user_tests'
TEST_DIR = PROJECT_ROOT / 'data' / 'test_missing_extreme'
API_BASE_URL = "http://localhost:8000"

print("="*80)
print("TEST 3: Extreme Missing Values - Robustness Check")
print("="*80)

# Create test directory
TEST_DIR.mkdir(exist_ok=True)

# Load original data
orig_app = pd.read_csv(END_USER_DIR / 'application.csv')

print(f"\n[1/3] Creating clients with extreme missing patterns...")

# Identify numeric columns that can be set to NaN
numeric_cols = orig_app.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col != 'SK_ID_CURR']

print(f"  Found {len(numeric_cols)} numeric columns")

# Test cases
test_cases = []

# Case 1: Original (baseline)
case1 = orig_app.iloc[0].to_dict()
case1['SK_ID_CURR'] = 200001
test_cases.append(('200001', 'Complete (0% missing)', case1, 0))

# Case 2: 10% missing (5 random fields)
case2 = orig_app.iloc[0].to_dict()
case2['SK_ID_CURR'] = 200002
np.random.seed(42)
missing_cols_10 = np.random.choice(numeric_cols, size=min(5, len(numeric_cols)), replace=False)
for col in missing_cols_10:
    case2[col] = np.nan
test_cases.append(('200002', f'10% missing ({len(missing_cols_10)} fields)', case2, len(missing_cols_10)))

# Case 3: 25% missing (12 random fields)
case3 = orig_app.iloc[0].to_dict()
case3['SK_ID_CURR'] = 200003
missing_cols_25 = np.random.choice(numeric_cols, size=min(12, len(numeric_cols)), replace=False)
for col in missing_cols_25:
    case3[col] = np.nan
test_cases.append(('200003', f'25% missing ({len(missing_cols_25)} fields)', case3, len(missing_cols_25)))

# Case 4: 50% missing (25 random fields)
case4 = orig_app.iloc[0].to_dict()
case4['SK_ID_CURR'] = 200004
missing_cols_50 = np.random.choice(numeric_cols, size=min(25, len(numeric_cols)), replace=False)
for col in missing_cols_50:
    case4[col] = np.nan
test_cases.append(('200004', f'50% missing ({len(missing_cols_50)} fields)', case4, len(missing_cols_50)))

# Create DataFrame
test_app = pd.DataFrame([case[2] for case in test_cases])
test_app.to_csv(TEST_DIR / 'application.csv', index=False)

print("\n  Test cases created:")
for id, desc, _, count in test_cases:
    print(f"    {id}: {desc}")

# Create supporting files
orig_bureau = pd.read_csv(END_USER_DIR / 'bureau.csv')
client_id = orig_app.iloc[0]['SK_ID_CURR']

bureau_rows = []
for test_id, _, _, _ in test_cases:
    client_bureau = orig_bureau[orig_bureau['SK_ID_CURR'] == client_id].copy()
    if len(client_bureau) > 0:
        client_bureau['SK_ID_CURR'] = int(test_id)
        bureau_rows.append(client_bureau)

if bureau_rows:
    combined_bureau = pd.concat(bureau_rows, ignore_index=True)
    combined_bureau.to_csv(TEST_DIR / 'bureau.csv', index=False)

# Other files
csv_files = {
    'bureau_balance': 'bureau_balance.csv',
    'credit_card_balance': 'credit_card_balance.csv',
    'installments_payments': 'installments_payments.csv',
    'POS_CASH_balance': 'POS_CASH_balance.csv',
    'previous_application': 'previous_application.csv'
}

for name, filename in csv_files.items():
    orig_path = END_USER_DIR / filename
    if orig_path.exists():
        df = pd.read_csv(orig_path)
        
        if 'SK_ID_CURR' in df.columns:
            dummy_rows = []
            for test_id, _, _, _ in test_cases:
                dummy = {col: (int(test_id) if col == 'SK_ID_CURR' else 0) for col in df.columns}
                dummy_rows.append(dummy)
            combined = pd.DataFrame(dummy_rows)
        else:
            combined = df.head(100)
        
        combined.to_csv(TEST_DIR / filename, index=False)

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
        print(f"{'Client ID':<12} {'Missing Pattern':<30} {'Prediction':<12} {'Probability':<15} {'Risk':<8}")
        print("-"*80)
        
        for i, (test_id, desc, _, _) in enumerate(test_cases):
            row = pred_df[pred_df['sk_id_curr'] == int(test_id)].iloc[0]
            print(f"{test_id:<12} {desc:<30} {row['prediction']:<12} {row['probability']:<15.8f} {row['risk_level']:<8}")
        
        # Analysis
        print("\n" + "="*80)
        print("ROBUSTNESS ANALYSIS")
        print("="*80)
        
        baseline_prob = pred_df[pred_df['sk_id_curr'] == 200001]['probability'].iloc[0]
        baseline_pred = pred_df[pred_df['sk_id_curr'] == 200001]['prediction'].iloc[0]
        
        print(f"\nBaseline (complete data):")
        print(f"  Probability: {baseline_prob:.8f}")
        print(f"  Prediction:  {baseline_pred}")
        
        print(f"\nImpact of missing values:")
        print(f"{'Missing %':<12} {'Prob Diff':<15} {'Pred Changed':<15} {'Assessment':<20}")
        print("-"*65)
        
        all_consistent = True
        max_diff = 0
        
        for i, (test_id, desc, _, count) in enumerate(test_cases[1:], 1):
            prob = pred_df[pred_df['sk_id_curr'] == int(test_id)]['probability'].iloc[0]
            pred = pred_df[pred_df['sk_id_curr'] == int(test_id)]['prediction'].iloc[0]
            
            diff = abs(prob - baseline_prob)
            max_diff = max(max_diff, diff)
            pred_changed = "YES" if pred != baseline_pred else "NO"
            
            if pred_changed == "YES":
                all_consistent = False
            
            # Assessment
            if diff < 0.01 and pred_changed == "NO":
                assessment = "Excellent"
            elif diff < 0.05 and pred_changed == "NO":
                assessment = "Good"
            elif diff < 0.10:
                assessment = "Acceptable"
            else:
                assessment = "Concerning"
            
            missing_pct = f"{(count/len(numeric_cols)*100):.0f}%"
            print(f"{missing_pct:<12} {diff:<15.8f} {pred_changed:<15} {assessment:<20}")
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        if max_diff < 0.01 and all_consistent:
            print("\n✓✓✓ EXCELLENT ROBUSTNESS")
            print("  System handles missing values consistently")
            print("  All predictions within 1% despite up to 50% missing data")
            print("  Training medians provide stable imputation")
        elif max_diff < 0.05 and all_consistent:
            print("\n✓✓ GOOD ROBUSTNESS")
            print("  System handles missing values well")
            print(f"  Max difference: {max_diff:.4f} (within 5%)")
            print("  No classification changes")
        elif all_consistent:
            print("\n✓ ACCEPTABLE ROBUSTNESS")
            print(f"  Max difference: {max_diff:.4f}")
            print("  Predictions remain consistent (no class changes)")
        else:
            print("\n⚠ MODERATE ROBUSTNESS")
            print(f"  Max difference: {max_diff:.4f}")
            print("  Some predictions changed classification")
            print("  Consider reviewing feature importance of missing fields")
        
    else:
        print(f"API Error: {response.status_code}")
        print(response.text[:500])
        
finally:
    for f in files_dict.values():
        f.close()

print("\n" + "="*80)
print(f"Test data kept at: {TEST_DIR}")
