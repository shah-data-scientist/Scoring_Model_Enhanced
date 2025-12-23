"""Verify API predictions match expected values for end user tests."""
import os

import requests

# Test batch predictions endpoint with new end user tests
files_dir = 'data/samples'
url = 'http://localhost:8001/batch/predict'

# Build files list with correct parameter names for multipart form
file_mappings = {
    'application.csv': 'application',
    'bureau.csv': 'bureau',
    'bureau_balance.csv': 'bureau_balance',
    'previous_application.csv': 'previous_application',
    'POS_CASH_balance.csv': 'pos_cash_balance',
    'credit_card_balance.csv': 'credit_card_balance',
    'installments_payments.csv': 'installments_payments'
}

file_handles = []
files_list = []
for filename, param_name in file_mappings.items():
    filepath = os.path.join(files_dir, filename)
    print(f"Checking {filepath}... ", end="")
    if os.path.exists(filepath):
        print("found")
        f = open(filepath, 'rb')
        file_handles.append(f)
        files_list.append((param_name, (filename, f, 'text/csv')))
    else:
        print("NOT FOUND")

print(f"\nSending {len(files_list)} files to API...")

try:
    response = requests.post(url, files=files_list, timeout=120)
    if response.status_code == 200:
        results = response.json()
        print('\nBatch prediction results:')
        print('-' * 70)
        low = medium = high = 0
        for pred in results['predictions']:
            risk = pred['risk_level']
            if risk == 'LOW': low += 1
            elif risk == 'MEDIUM': medium += 1
            else: high += 1
            decision = "Default" if pred['prediction'] == 1 else "No Default"
            print(f"ID {pred['sk_id_curr']}: {pred['probability']*100:.1f}% - {risk} - {decision}")
        print('-' * 70)
        print(f'Distribution: LOW={low}, MEDIUM={medium}, HIGH={high}')
    else:
        print(f'Error: {response.status_code}')
        print(response.text)
finally:
    for f in file_handles:
        f.close()
