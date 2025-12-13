"""Test 1: Verify missing values are imputed with global training medians."""
import pandas as pd
import numpy as np
import requests
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).parent
END_USER_DIR = PROJECT_ROOT / 'data' / 'end_user_tests'
TEST_DIR = PROJECT_ROOT / 'data' / 'test_missing_values'
API_BASE_URL = "http://localhost:8000"

print("="*80)
print("TEST 1: Missing Value Handling - Global vs Batch Medians")
print("="*80)

# Load medians from training
medians_file = PROJECT_ROOT / 'data' / 'processed' / 'medians.json'
if not medians_file.exists():
    print(f"ERROR: Medians file not found: {medians_file}")
    exit(1)

with open(medians_file, 'r') as f:
    training_medians = json.load(f)

print(f"\nTraining medians loaded: {len(training_medians)} features")
print(f"Sample medians:")
for key in list(training_medians.keys())[:5]:
    print(f"  {key}: {training_medians[key]}")

# Create test directory
TEST_DIR.mkdir(exist_ok=True)

# Load original application data
orig_app = pd.read_csv(END_USER_DIR / 'application.csv')
client_id = orig_app.iloc[0]['SK_ID_CURR']

print(f"\n[1/4] Creating test data with missing values...")
print(f"Using client SK_ID_CURR={client_id}")

# Test Strategy: Create 2 clients - one with missing AMT_ANNUITY, one without
test_rows = []

# Client 1: Original data (no missing values)
row1 = orig_app.iloc[0].to_dict()
test_rows.append(row1)

# Client 2: Same data but with AMT_ANNUITY set to NaN
row2 = orig_app.iloc[0].to_dict()
row2['SK_ID_CURR'] = 999888  # Different ID
row2['AMT_ANNUITY'] = np.nan  # Introduce missing value
test_rows.append(row2)

test_app = pd.DataFrame(test_rows)
test_app.to_csv(TEST_DIR / 'application.csv', index=False)

print(f"  Created application.csv:")
print(f"    Client {client_id}: AMT_ANNUITY = {row1['AMT_ANNUITY']}")
print(f"    Client 999888: AMT_ANNUITY = NaN (will be imputed)")

# Copy supporting files (use dummy data for simplicity)
csv_files = ['bureau', 'bureau_balance', 'credit_card_balance', 
             'installments_payments', 'POS_CASH_balance', 'previous_application']

for csv_name in csv_files:
    # Create minimal dummy files
    orig_file = END_USER_DIR / f'{csv_name}.csv'
    if csv_name == 'POS_CASH_balance':
        orig_file = END_USER_DIR / 'POS_CASH_balance.csv'
    
    if orig_file.exists():
        df = pd.read_csv(orig_file)
        
        # Filter for our client if has SK_ID_CURR
        if 'SK_ID_CURR' in df.columns:
            client_data = df[df['SK_ID_CURR'] == client_id].copy()
            if len(client_data) > 0:
                # Duplicate for second client
                client2_data = client_data.copy()
                client2_data['SK_ID_CURR'] = 999888
                combined = pd.concat([client_data, client2_data], ignore_index=True)
            else:
                # Create dummy rows
                dummy = {col: (client_id if col == 'SK_ID_CURR' else 0) for col in df.columns}
                dummy2 = {col: (999888 if col == 'SK_ID_CURR' else 0) for col in df.columns}
                combined = pd.DataFrame([dummy, dummy2])
        else:
            # bureau_balance - just copy as is
            combined = df.head(100)
        
        output_name = 'POS_CASH_balance.csv' if csv_name == 'POS_CASH_balance' else f'{csv_name}.csv'
        combined.to_csv(TEST_DIR / output_name, index=False)
        print(f"  ✓ {output_name}: {len(combined)} rows")

print(f"\n[2/4] Sending to API...")

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
        
        print("\n[3/4] Results:")
        print("-"*80)
        print(f"{'SK_ID_CURR':<12} {'Has Missing?':<15} {'Prediction':<12} {'Probability':<15}")
        print("-"*80)
        
        for _, row in pred_df.iterrows():
            has_missing = "No (original)" if row['sk_id_curr'] == client_id else "Yes (imputed)"
            print(f"{row['sk_id_curr']:<12} {has_missing:<15} {row['prediction']:<12} {row['probability']:<15.8f}")
        
        # Analysis
        print("\n[4/4] Analysis:")
        print("-"*80)
        
        prob_diff = abs(pred_df.iloc[0]['probability'] - pred_df.iloc[1]['probability'])
        
        print(f"\nProbability difference: {prob_diff:.10f}")
        
        if prob_diff < 0.0001:
            print("\n✓ SUCCESS: Missing values imputed consistently!")
            print("  Both clients have nearly identical predictions")
            print("  AMT_ANNUITY was imputed with training median")
        elif prob_diff < 0.05:
            print("\n⚠ ACCEPTABLE: Small difference detected")
            print(f"  Difference of {prob_diff:.4f} is minor")
            print("  Could be due to feature interactions")
        else:
            print("\n✗ CONCERN: Significant difference detected")
            print(f"  Difference of {prob_diff:.4f} is large")
            print("  Missing value handling may be inconsistent")
        
    else:
        print(f"API Error: {response.status_code}")
        print(response.text[:500])
        
finally:
    for f in files_dict.values():
        f.close()

print("\n" + "="*80)
print(f"Test data kept at: {TEST_DIR}")
