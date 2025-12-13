"""Simple test: Check if predictions differ based on SK_ID_CURR value alone."""
import pandas as pd
from pathlib import Path
import shutil

# Test strategy: Take ONE client, duplicate with different SK_ID_CURR, predict both

PROJECT_ROOT = Path(__file__).parent
END_USER_DIR = PROJECT_ROOT / 'data' / 'end_user_tests'
TEST_DIR = PROJECT_ROOT / 'data' / 'test_single_client'

print("="*80)
print("SINGLE CLIENT TEST: Does SK_ID_CURR affect features/predictions?")
print("="*80)

# Create test directory
TEST_DIR.mkdir(exist_ok=True)

# Pick first client
orig_app = pd.read_csv(END_USER_DIR / 'application.csv')
client_id = orig_app.iloc[0]['SK_ID_CURR']

print(f"\nUsing client SK_ID_CURR={client_id}")

# Strategy: Create 3 versions of SAME client with DIFFERENT IDs
test_ids = [client_id, 888888, 999999]

print(f"\nCreating 3 copies with IDs: {test_ids}")

# Create application.csv with 3 rows (same data, different IDs)
app_row = orig_app[orig_app['SK_ID_CURR'] == client_id].iloc[0].to_dict()
test_rows = []
for test_id in test_ids:
    row = app_row.copy()
    row['SK_ID_CURR'] = test_id
    test_rows.append(row)

test_app = pd.DataFrame(test_rows)
test_app.to_csv(TEST_DIR / 'application.csv', index=False)
print(f"  ✓ application.csv: 3 rows")

# Copy related data for all 3 IDs
for csv_file in ['bureau', 'bureau_balance', 'credit_card_balance', 
                 'installments_payments', 'POS_CASH_balance', 'previous_application']:
    
    full_path = END_USER_DIR / f'{csv_file}.csv'
    if not full_path.exists():
        full_path = END_USER_DIR / f'{csv_file.replace("_", "")}.csv'
    
    if full_path.exists():
        df = pd.read_csv(full_path)
        
        # Check if SK_ID_CURR column exists
        if 'SK_ID_CURR' not in df.columns:
            # bureau_balance doesn't have SK_ID_CURR, just copy as-is for now
            # We'll handle the join properly later
            output_name = 'POS_CASH_balance.csv' if csv_file == 'POS_CASH_balance' else f'{csv_file}.csv'
            df.to_csv(TEST_DIR / output_name, index=False)
            print(f"  ✓ {output_name}: {len(df)} rows (no SK_ID_CURR column, copied as-is)")
            continue
        
        # Filter for original client
        client_data = df[df['SK_ID_CURR'] == client_id].copy()
        
        if len(client_data) > 0:
            # Create 3 copies
            all_data = []
            for test_id in test_ids:
                data_copy = client_data.copy()
                data_copy['SK_ID_CURR'] = test_id
                all_data.append(data_copy)
            
            combined = pd.concat(all_data, ignore_index=True)
            
            # Save with correct filename
            output_name = 'POS_CASH_balance.csv' if csv_file == 'POS_CASH_balance' else f'{csv_file}.csv'
            combined.to_csv(TEST_DIR / output_name, index=False)
            print(f"  ✓ {output_name}: {len(combined)} rows ({len(client_data)} per client)")
        else:
            # Create file with dummy rows (API rejects empty files)
            output_name = 'POS_CASH_balance.csv' if csv_file == 'POS_CASH_balance' else f'{csv_file}.csv'
            
            # Create dummy rows for each test ID
            dummy_rows = []
            for test_id in test_ids:
                dummy_row = {col: (test_id if col == 'SK_ID_CURR' else (0 if pd.api.types.is_numeric_dtype(df[col]) else 'DUMMY')) 
                            for col in df.columns}
                dummy_rows.append(dummy_row)
            
            pd.DataFrame(dummy_rows).to_csv(TEST_DIR / output_name, index=False)
            print(f"  ✓ {output_name}: {len(dummy_rows)} rows (dummy data - client has none)")

print(f"\n✓ Test data created at: {TEST_DIR}")
print("\nNow sending to API...")

# Send to API
import requests

API_BASE_URL = "http://localhost:8000"

csv_files = {
    'application': 'application.csv',
    'bureau': 'bureau.csv', 
    'bureau_balance': 'bureau_balance.csv',
    'credit_card_balance': 'credit_card_balance.csv',
    'installments_payments': 'installments_payments.csv',
    'pos_cash_balance': 'POS_CASH_balance.csv',
    'previous_application': 'previous_application.csv'
}

files_dict = {}
for api_name, csv_file in csv_files.items():
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
        
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        print(f"\n{'SK_ID_CURR':<12} {'Prediction':<12} {'Probability':<15} {'Risk Level':<12}")
        print("-"*80)
        
        for _, row in pred_df.iterrows():
            print(f"{row['sk_id_curr']:<12} {row['prediction']:<12} {row['probability']:<15.8f} {row['risk_level']:<12}")
        
        # Check if all predictions are identical
        probs = pred_df['probability'].values
        preds = pred_df['prediction'].values
        
        print("\n" + "="*80)
        if len(set(probs)) == 1 and len(set(preds)) == 1:
            print("✓✓✓ ALL 3 PREDICTIONS IDENTICAL! ✓✓✓")
            print("SK_ID_CURR has NO impact on predictions")
        else:
            print("✗✗✗ PREDICTIONS DIFFER! ✗✗✗")
            print("SK_ID_CURR somehow affects predictions (unexpected!)")
            print(f"\nProbability differences: {probs}")
        print("="*80)
        
    else:
        print(f"\nAPI Error: {response.status_code}")
        print(response.text[:500])
        
finally:
    for f in files_dict.values():
        f.close()

print(f"\nTest data kept at: {TEST_DIR}")
print(f"To delete: shutil.rmtree('{TEST_DIR}')")
