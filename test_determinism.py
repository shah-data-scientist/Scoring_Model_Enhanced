"""Test if API predictions are deterministic (same data twice)."""
import pandas as pd
import requests
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
END_USER_DIR = PROJECT_ROOT / 'data' / 'end_user_tests'
API_BASE_URL = "http://localhost:8000"

print("="*70)
print("DETERMINISM TEST: Same Data → Same Predictions?")
print("="*70)

csv_files = {
    'application': 'application.csv',
    'bureau': 'bureau.csv', 
    'bureau_balance': 'bureau_balance.csv',
    'credit_card_balance': 'credit_card_balance.csv',
    'installments_payments': 'installments_payments.csv',
    'pos_cash_balance': 'POS_CASH_balance.csv',
    'previous_application': 'previous_application.csv'
}

def get_predictions():
    """Get predictions from API."""
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
            return pd.DataFrame(response.json()['predictions'])
        else:
            print(f"API Error: {response.status_code}")
            return None
    finally:
        for f in files_dict.values():
            f.close()

# Run 1
print("\n[1/3] Getting predictions - RUN 1...")
predictions_run1 = get_predictions()
if predictions_run1 is None:
    print("Failed to get predictions!")
    exit(1)
print(f"  ✓ Got {len(predictions_run1)} predictions")
print(f"    Sample: ID={predictions_run1.iloc[0]['sk_id_curr']}, "
      f"Pred={predictions_run1.iloc[0]['prediction']}, "
      f"Prob={predictions_run1.iloc[0]['probability']:.6f}")

# Run 2
print("\n[2/3] Getting predictions - RUN 2...")
predictions_run2 = get_predictions()
if predictions_run2 is None:
    print("Failed to get predictions!")
    exit(1)
print(f"  ✓ Got {len(predictions_run2)} predictions")
print(f"    Sample: ID={predictions_run2.iloc[0]['sk_id_curr']}, "
      f"Pred={predictions_run2.iloc[0]['prediction']}, "
      f"Prob={predictions_run2.iloc[0]['probability']:.6f}")

# Compare
print("\n[3/3] Comparing predictions...")
print("-"*70)
print(f"\n{'ID':<12} {'Run1 Pred':<10} {'Run2 Pred':<10} {'Run1 Prob':<12} {'Run2 Prob':<12} {'Match':<8}")
print("-"*70)

all_match = True
max_diff = 0

for i in range(len(predictions_run1)):
    id_val = predictions_run1.iloc[i]['sk_id_curr']
    
    pred1 = predictions_run1.iloc[i]['prediction']
    pred2 = predictions_run2.iloc[i]['prediction']
    
    prob1 = predictions_run1.iloc[i]['probability']
    prob2 = predictions_run2.iloc[i]['probability']
    
    prob_diff = abs(prob1 - prob2)
    max_diff = max(max_diff, prob_diff)
    
    match = (pred1 == pred2) and (prob_diff < 1e-10)
    status = "✓ YES" if match else "✗ NO"
    
    if not match:
        all_match = False
    
    if i < 10 or not match:  # Show first 10 or any mismatches
        print(f"{id_val:<12} {pred1:<10} {pred2:<10} {prob1:<12.6f} {prob2:<12.6f} {status:<8}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\n  Total records:           {len(predictions_run1)}")
print(f"  Predictions match:       {sum(predictions_run1['prediction'] == predictions_run2['prediction'])}/{len(predictions_run1)}")
print(f"  Max probability diff:    {max_diff:.15f}")

if all_match:
    print(f"\n  ✓✓✓ PERFECTLY DETERMINISTIC! ✓✓✓")
    print(f"  API produces identical predictions for same input")
else:
    print(f"\n  ✗✗✗ NON-DETERMINISTIC! ✗✗✗")
    print(f"  API produces different predictions for same input")

print("\n" + "="*70)
