import requests
import json

API_URL = "http://127.0.0.1:8000/monitoring/drift/batch/157"

print(f"Testing Drift Detection for Batch 157 at {API_URL}...")

try:
    response = requests.post(API_URL, timeout=60)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("\nResponse Summary:")
        print(f"Batch ID: {data.get('batch_id')}")
        print(f"Features Checked: {data.get('features_checked')}")
        print(f"Features Drifted: {data.get('features_drifted')}")
        
        results = data.get('results', {{}})
        if results:
            print("\nDrift Results Sample (first 3):")
            count = 0
            for k, v in results.items():
                print(f"  {k}: {v.get('interpretation', 'N/A')} (p={v.get('p_value')})")
                count += 1
                if count >= 3:
                    break
        else:
            print("\nWARNING: No results returned in 'results' key.")
    else:
        print("\nError Response:")
        try:
            print(json.dumps(response.json(), indent=2))
        except:
            print(response.text)

except Exception as e:
    print(f"Request failed: {e}")
