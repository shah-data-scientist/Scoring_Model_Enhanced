"""Test batch prediction API with end_user_tests files."""
import requests
from pathlib import Path
import json
import pandas as pd

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
TEST_DIR = PROJECT_ROOT / "data" / "end_user_tests"
API_URL = "http://localhost:8000"

def print_header(title):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

print_header("BATCH API TEST - END USER FILES")
print(f"\nTest Data Directory: {TEST_DIR}")
print(f"API URL: {API_URL}")

# Check how many applications in the test data
app_df = pd.read_csv(TEST_DIR / 'application.csv')
print(f"Number of applications: {len(app_df)}")
print(f"Application IDs: {app_df['SK_ID_CURR'].tolist()}")

# Prepare files for upload
print_header("STEP 1: FILE VALIDATION")
files = {
    'application': ('application.csv', open(TEST_DIR / 'application.csv', 'rb'), 'text/csv'),
    'bureau': ('bureau.csv', open(TEST_DIR / 'bureau.csv', 'rb'), 'text/csv'),
    'bureau_balance': ('bureau_balance.csv', open(TEST_DIR / 'bureau_balance.csv', 'rb'), 'text/csv'),
    'previous_application': ('previous_application.csv', open(TEST_DIR / 'previous_application.csv', 'rb'), 'text/csv'),
    'credit_card_balance': ('credit_card_balance.csv', open(TEST_DIR / 'credit_card_balance.csv', 'rb'), 'text/csv'),
    'installments_payments': ('installments_payments.csv', open(TEST_DIR / 'installments_payments.csv', 'rb'), 'text/csv'),
    'pos_cash_balance': ('POS_CASH_balance.csv', open(TEST_DIR / 'POS_CASH_balance.csv', 'rb'), 'text/csv'),
}

try:
    response = requests.post(f"{API_URL}/batch/validate", files=files, timeout=30)

    # Close file handles
    for f in files.values():
        f[1].close()

    if response.status_code == 200:
        data = response.json()
        print(f"[OK] Validation Status: {response.status_code}")
        print(f"[OK] Files Valid: {data.get('success', False)}")

        if 'file_summaries' in data:
            print("\nFile Summaries:")
            for file_name, summary in data['file_summaries'].items():
                print(f"  - {file_name}: {summary.get('rows', '?')} rows")
    else:
        print(f"[FAIL] Validation failed: {response.status_code}")
        print(f"       Error: {response.text[:500]}")
        exit(1)

except Exception as e:
    print(f"[FAIL] Validation error: {str(e)}")
    exit(1)

# Make batch prediction
print_header("STEP 2: BATCH PREDICTION")

# Reopen files for prediction
files = {
    'application': ('application.csv', open(TEST_DIR / 'application.csv', 'rb'), 'text/csv'),
    'bureau': ('bureau.csv', open(TEST_DIR / 'bureau.csv', 'rb'), 'text/csv'),
    'bureau_balance': ('bureau_balance.csv', open(TEST_DIR / 'bureau_balance.csv', 'rb'), 'text/csv'),
    'previous_application': ('previous_application.csv', open(TEST_DIR / 'previous_application.csv', 'rb'), 'text/csv'),
    'credit_card_balance': ('credit_card_balance.csv', open(TEST_DIR / 'credit_card_balance.csv', 'rb'), 'text/csv'),
    'installments_payments': ('installments_payments.csv', open(TEST_DIR / 'installments_payments.csv', 'rb'), 'text/csv'),
    'pos_cash_balance': ('POS_CASH_balance.csv', open(TEST_DIR / 'POS_CASH_balance.csv', 'rb'), 'text/csv'),
}

try:
    print("Making prediction request...")
    response = requests.post(f"{API_URL}/batch/predict", files=files, timeout=120)

    # Close file handles
    for f in files.values():
        f[1].close()

    if response.status_code == 200:
        data = response.json()
        predictions = data['predictions']

        print(f"[OK] Prediction Status: {response.status_code}")
        print(f"[OK] Number of predictions: {len(predictions)}")

        # Risk level distribution
        risk_levels = [p['risk_level'] for p in predictions]
        risk_distribution = {level: risk_levels.count(level) for level in set(risk_levels)}

        print("\nRisk Level Distribution:")
        for level in ['LOW', 'MEDIUM', 'HIGH']:
            count = risk_distribution.get(level, 0)
            pct = (count / len(predictions) * 100) if predictions else 0
            print(f"  {level:8s}: {count:2d} ({pct:5.1f}%)")

        # Probability statistics
        probabilities = [p['probability'] for p in predictions]
        print(f"\nProbability Statistics:")
        print(f"  Min:    {min(probabilities):.4f}")
        print(f"  Max:    {max(probabilities):.4f}")
        print(f"  Mean:   {sum(probabilities)/len(probabilities):.4f}")
        print(f"  Median: {sorted(probabilities)[len(probabilities)//2]:.4f}")

        # Show all predictions
        print_header("ALL PREDICTIONS")
        print(f"{'SK_ID':>10s} {'Prediction':>12s} {'Probability':>12s} {'Risk Level':>12s}")
        print("-" * 70)

        for pred in sorted(predictions, key=lambda x: x['probability'], reverse=True):
            print(f"{pred['sk_id_curr']:10d} "
                  f"{pred['prediction']:12d} "
                  f"{pred['probability']:12.4f} "
                  f"{pred['risk_level']:>12s}")

        # Processing info
        if 'processing_info' in data:
            info = data['processing_info']
            print_header("PROCESSING INFO")
            print(f"Total processed: {info.get('total_processed', 'N/A')}")
            print(f"Features used: {info.get('features_used', 'N/A')}")
            if 'lookup_count' in info:
                print(f"From lookup: {info['lookup_count']}")
            if 'pipeline_count' in info:
                print(f"From pipeline: {info['pipeline_count']}")

        # Save results
        output_file = PROJECT_ROOT / 'results' / 'end_user_test_predictions.csv'
        output_file.parent.mkdir(parents=True, exist_ok=True)

        results_df = pd.DataFrame(predictions)
        results_df.to_csv(output_file, index=False)
        print(f"\n[OK] Results saved to: {output_file}")

    else:
        print(f"[FAIL] Prediction failed: {response.status_code}")
        print(f"       Error: {response.text[:1000]}")
        exit(1)

except Exception as e:
    import traceback
    print(f"[FAIL] Prediction error: {str(e)}")
    traceback.print_exc()
    exit(1)

print_header("TEST COMPLETE")
print("[SUCCESS] All end_user_tests files processed successfully!")
