import requests
import os

API_URL = "http://127.0.0.1:8000/batch/predict"
FILES_DIR = "data/samples"

files = {
    'application': ('application.csv', open(os.path.join(FILES_DIR, 'application.csv'), 'rb'), 'text/csv'),
    'bureau': ('bureau.csv', open(os.path.join(FILES_DIR, 'bureau.csv'), 'rb'), 'text/csv'),
    'bureau_balance': ('bureau_balance.csv', open(os.path.join(FILES_DIR, 'bureau_balance.csv'), 'rb'), 'text/csv'),
    'previous_application': ('previous_application.csv', open(os.path.join(FILES_DIR, 'previous_application.csv'), 'rb'), 'text/csv'),
    'credit_card_balance': ('credit_card_balance.csv', open(os.path.join(FILES_DIR, 'credit_card_balance.csv'), 'rb'), 'text/csv'),
    'installments_payments': ('installments_payments.csv', open(os.path.join(FILES_DIR, 'installments_payments.csv'), 'rb'), 'text/csv'),
    'pos_cash_balance': ('POS_CASH_balance.csv', open(os.path.join(FILES_DIR, 'POS_CASH_balance.csv'), 'rb'), 'text/csv'),
}

print(f"Uploading files from {FILES_DIR} to {API_URL}...")
try:
    response = requests.post(API_URL, files=files, timeout=30)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("Success!")
        print(response.json())
    else:
        print("Error:")
        print(response.text)
except Exception as e:
    print(f"Request failed: {e}")
finally:
    for f in files.values():
        f[1].close()
