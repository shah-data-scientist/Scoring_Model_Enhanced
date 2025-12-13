"""Test predictions with cache disabled to verify the fix works."""
import requests
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
TEST_DIR = PROJECT_ROOT / 'data' / 'test_single_client'
API_BASE_URL = "http://localhost:8000"

print("="*80)
print("TESTING FIX: Predictions with Global Statistics")
print("="*80)

# Check if test data exists
if not TEST_DIR.exists():
    print(f"\nERROR: Test data not found. Run test_single_client.py first.")
    exit(1)

print("\nSending test data to API...")
print("(3 copies of same client with different SK_ID_CURRs)")

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
        
        # Check consistency
        probs = pred_df['probability'].values
        preds = pred_df['prediction'].values
        
        print("\n" + "="*80)
        print("ANALYSIS")
        print("="*80)
        
        # Group by unique probability values
        unique_probs = set(probs)
        
        if len(unique_probs) == 1:
            print("\n✓✓✓ SUCCESS! ALL 3 PREDICTIONS IDENTICAL! ✓✓✓")
            print("\nThe fix worked:")
            print("  - All 3 IDs now use global statistics from training")
            print("  - Dataset-level features are consistent")
            print("  - SK_ID_CURR no longer affects predictions")
        elif len(unique_probs) == 2:
            # Check if known vs unknown split
            id_111761_prob = pred_df[pred_df['sk_id_curr'] == 111761]['probability'].iloc[0]
            other_probs = pred_df[pred_df['sk_id_curr'] != 111761]['probability'].unique()
            
            if len(other_probs) == 1:
                print("\n⚠ PARTIAL SUCCESS - Cache still active")
                print(f"\n  Known ID (111761):   prob = {id_111761_prob:.8f}")
                print(f"  Unknown IDs (others): prob = {other_probs[0]:.8f}")
                print("\n  The two unknown IDs now match each other (good!)")
                print("  But known ID still uses cache (different)")
                print("\n  Solution: Regenerate precomputed features with new code")
                print("           or disable cache (use_precomputed=False)")
            else:
                print("\n✗ UNEXPECTED: Different patterns detected")
        else:
            print("\n✗ ALL 3 PREDICTIONS STILL DIFFER")
            print("  The fix may not be applied yet")
        
        print("\n" + "="*80)
        
    else:
        print(f"\nAPI Error: {response.status_code}")
        print(response.text[:500])
        
finally:
    for f in files_dict.values():
        f.close()
