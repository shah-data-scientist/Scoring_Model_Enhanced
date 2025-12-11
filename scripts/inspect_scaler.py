
import joblib
import pandas as pd
from pathlib import Path

def inspect_scaler():
    path = Path("data/processed/scaler.joblib")
    scaler = joblib.load(path)
    
    print(f"Features: {scaler.n_features_in_}")
    
    features = scaler.feature_names_in_
    means = scaler.mean_
    scales = scaler.scale_
    
    df = pd.DataFrame({
        'feature': features,
        'mean': means,
        'scale': scales
    })
    
    # Check BUREAU_AMT_CREDIT_SUM_SUM
    target = "BUREAU_AMT_CREDIT_SUM_SUM"
    if target in features:
        print(f"\nStats for {target}:")
        print(df[df['feature'] == target])
    else:
        print(f"{target} not found in scaler")
        
    # Check PREV_AMT_APPLICATION_MIN
    target = "PREV_AMT_APPLICATION_MIN"
    if target in features:
        print(f"\nStats for {target}:")
        print(df[df['feature'] == target])

if __name__ == "__main__":
    inspect_scaler()
