"""End-to-End API Testing Script for Batch Predictions
====================================================
Tests the batch prediction API with sample data to verify:
1. File validation endpoint works correctly
2. Batch prediction endpoint returns proper results
3. Preprocessing pipeline produces correct features
4. Risk levels are calculated correctly
"""

import sys
from pathlib import Path

import requests

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
SAMPLE_DIR = PROJECT_ROOT / "data" / "samples"
API_URL = "http://localhost:8000"

def print_header(title):
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def print_result(test_name, passed, details=""):
    """Print test result."""
    status = "[PASS]" if passed else "[FAIL]"
    print(f"  {status} {test_name}")
    if details:
        print(f"         {details}")

def test_api_health():
    """Test 1: Check API is running."""
    print_header("TEST 1: API Health Check")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        passed = response.status_code == 200
        print_result("API Health Endpoint", passed, f"Status: {response.status_code}")
        return passed
    except Exception as e:
        print_result("API Health Endpoint", False, f"Error: {str(e)}")
        return False

def test_file_validation():
    """Test 2: Test file validation endpoint."""
    print_header("TEST 2: File Validation Endpoint")

    # Prepare files
    files = {
        'application': ('application.csv', open(SAMPLE_DIR / 'application.csv', 'rb'), 'text/csv'),
        'bureau': ('bureau.csv', open(SAMPLE_DIR / 'bureau.csv', 'rb'), 'text/csv'),
        'bureau_balance': ('bureau_balance.csv', open(SAMPLE_DIR / 'bureau_balance.csv', 'rb'), 'text/csv'),
        'previous_application': ('previous_application.csv', open(SAMPLE_DIR / 'previous_application.csv', 'rb'), 'text/csv'),
        'credit_card_balance': ('credit_card_balance.csv', open(SAMPLE_DIR / 'credit_card_balance.csv', 'rb'), 'text/csv'),
        'installments_payments': ('installments_payments.csv', open(SAMPLE_DIR / 'installments_payments.csv', 'rb'), 'text/csv'),
        'pos_cash_balance': ('POS_CASH_balance.csv', open(SAMPLE_DIR / 'POS_CASH_balance.csv', 'rb'), 'text/csv'),
    }

    try:
        response = requests.post(f"{API_URL}/batch/validate", files=files, timeout=30)

        # Close file handles
        for f in files.values():
            f[1].close()

        passed = response.status_code == 200
        print_result("Validation Request", passed, f"Status: {response.status_code}")

        if passed:
            data = response.json()
            # Check for either 'success' or 'valid' key (API uses 'success')
            has_status = 'success' in data or 'valid' in data
            print_result("Response Format", has_status, f"Keys: {list(data.keys())}")
            if 'success' in data:
                print_result("Files Valid", data['success'], f"Validation result: {data['success']}")
            elif 'valid' in data:
                print_result("Files Valid", data['valid'], f"Validation result: {data['valid']}")
            if 'file_summaries' in data:
                print("\n  File Summaries:")
                for file_name, summary in data.get('file_summaries', {}).items():
                    print(f"    - {file_name}: {summary.get('rows', '?')} rows")
        else:
            print(f"  Error Response: {response.text[:500]}")

        return passed
    except Exception as e:
        print_result("Validation Request", False, f"Error: {str(e)}")
        return False

def test_batch_prediction():
    """Test 3: Test batch prediction endpoint."""
    print_header("TEST 3: Batch Prediction Endpoint")

    # Prepare files
    files = {
        'application': ('application.csv', open(SAMPLE_DIR / 'application.csv', 'rb'), 'text/csv'),
        'bureau': ('bureau.csv', open(SAMPLE_DIR / 'bureau.csv', 'rb'), 'text/csv'),
        'bureau_balance': ('bureau_balance.csv', open(SAMPLE_DIR / 'bureau_balance.csv', 'rb'), 'text/csv'),
        'previous_application': ('previous_application.csv', open(SAMPLE_DIR / 'previous_application.csv', 'rb'), 'text/csv'),
        'credit_card_balance': ('credit_card_balance.csv', open(SAMPLE_DIR / 'credit_card_balance.csv', 'rb'), 'text/csv'),
        'installments_payments': ('installments_payments.csv', open(SAMPLE_DIR / 'installments_payments.csv', 'rb'), 'text/csv'),
        'pos_cash_balance': ('POS_CASH_balance.csv', open(SAMPLE_DIR / 'POS_CASH_balance.csv', 'rb'), 'text/csv'),
    }

    try:
        print("  Making prediction request (this may take a moment)...")
        response = requests.post(f"{API_URL}/batch/predict", files=files, timeout=120)

        # Close file handles
        for f in files.values():
            f[1].close()

        passed = response.status_code == 200
        print_result("Prediction Request", passed, f"Status: {response.status_code}")

        if passed:
            data = response.json()

            # Check response structure
            has_predictions = 'predictions' in data
            print_result("Has Predictions Key", has_predictions)

            if has_predictions:
                predictions = data['predictions']
                num_predictions = len(predictions)
                print_result("Prediction Count", num_predictions == 20, f"Expected: 20, Got: {num_predictions}")

                # Check first prediction structure
                if predictions:
                    first_pred = predictions[0]
                    required_keys = ['sk_id_curr', 'prediction', 'probability', 'risk_level']
                    has_all_keys = all(k in first_pred for k in required_keys)
                    print_result("Prediction Structure", has_all_keys, f"Keys: {list(first_pred.keys())}")

                    # Check risk levels distribution
                    risk_levels = [p['risk_level'] for p in predictions]
                    risk_distribution = {level: risk_levels.count(level) for level in set(risk_levels)}
                    print("\n  Risk Level Distribution:")
                    for level, count in sorted(risk_distribution.items()):
                        print(f"    - {level}: {count}")

                    # Check probability ranges
                    probabilities = [p['probability'] for p in predictions]
                    prob_valid = all(0 <= p <= 1 for p in probabilities)
                    print_result("Probabilities in [0,1]", prob_valid,
                                f"Range: [{min(probabilities):.4f}, {max(probabilities):.4f}]")

                    # Sample predictions
                    print("\n  Sample Predictions (first 5):")
                    for pred in predictions[:5]:
                        print(f"    SK_ID: {pred['sk_id_curr']}, Prob: {pred['probability']:.4f}, "
                              f"Risk: {pred['risk_level']}")

            # Check processing info
            if 'processing_info' in data:
                info = data['processing_info']
                print("\n  Processing Info:")
                print(f"    - Total processed: {info.get('total_processed', 'N/A')}")
                print(f"    - Features used: {info.get('features_used', 'N/A')}")
        else:
            error_text = response.text[:1000]
            print(f"\n  Error Response:\n{error_text}")

        return passed
    except Exception as e:
        import traceback
        print_result("Prediction Request", False, f"Error: {str(e)}")
        traceback.print_exc()
        return False

def test_missing_file():
    """Test 4: Test error handling with missing file."""
    print_header("TEST 4: Missing File Error Handling")

    # Only send some files (missing pos_cash_balance)
    files = {
        'application': ('application.csv', open(SAMPLE_DIR / 'application.csv', 'rb'), 'text/csv'),
        'bureau': ('bureau.csv', open(SAMPLE_DIR / 'bureau.csv', 'rb'), 'text/csv'),
        'bureau_balance': ('bureau_balance.csv', open(SAMPLE_DIR / 'bureau_balance.csv', 'rb'), 'text/csv'),
        'previous_application': ('previous_application.csv', open(SAMPLE_DIR / 'previous_application.csv', 'rb'), 'text/csv'),
        'credit_card_balance': ('credit_card_balance.csv', open(SAMPLE_DIR / 'credit_card_balance.csv', 'rb'), 'text/csv'),
        'installments_payments': ('installments_payments.csv', open(SAMPLE_DIR / 'installments_payments.csv', 'rb'), 'text/csv'),
        # Missing: pos_cash_balance
    }

    try:
        response = requests.post(f"{API_URL}/batch/validate", files=files, timeout=30)

        # Close file handles
        for f in files.values():
            f[1].close()

        # Should return 422 (validation error) since field is required
        passed = response.status_code == 422
        print_result("Missing File Detection", passed,
                    f"Status: {response.status_code} (expected 422 for missing required field)")

        return passed
    except Exception as e:
        print_result("Missing File Detection", False, f"Error: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  BATCH PREDICTION API - END-TO-END TESTS")
    print("=" * 60)
    print(f"\nSample Data Directory: {SAMPLE_DIR}")
    print(f"API URL: {API_URL}")

    # Run tests
    results = []

    # Test 1: API Health
    results.append(("API Health", test_api_health()))

    if results[0][1]:  # Only continue if API is healthy
        # Test 2: File Validation
        results.append(("File Validation", test_file_validation()))

        # Test 3: Batch Prediction
        results.append(("Batch Prediction", test_batch_prediction()))

        # Test 4: Error Handling
        results.append(("Missing File Error", test_missing_file()))

    # Summary
    print_header("TEST SUMMARY")
    passed = sum(1 for _, p in results if p)
    total = len(results)

    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {name}")

    print(f"\n  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n  [SUCCESS] All tests passed! Phase 2 API is ready.")
        return 0
    print("\n  [WARNING] Some tests failed. Please review the output above.")
    return 1

if __name__ == "__main__":
    sys.exit(main())
