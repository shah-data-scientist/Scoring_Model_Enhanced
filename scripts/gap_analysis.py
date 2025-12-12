
import pandas as pd
import numpy as np
import subprocess
import sys
from pathlib import Path
import joblib
import pickle
import lightgbm

# Add project root to path
sys.path.insert(0, ".")

from api.preprocessing_pipeline import PreprocessingPipeline

def run_script(script_name):
    print(f"Running {script_name}...")
    subprocess.run([sys.executable, script_name], check=True)

def get_predictions(description):
    print(f"\nGenerating predictions for: {description}")
    
    # Load data
    data_dir = Path("data/end_user_tests")
    dataframes = {}
    for fname in ["application.csv", "bureau.csv", "bureau_balance.csv", 
                  "previous_application.csv", "POS_CASH_balance.csv", 
                  "credit_card_balance.csv", "installments_payments.csv"]:
        fpath = data_dir / fname
        if fpath.exists():
            dataframes[fname] = pd.read_csv(fpath)
            
    # Run Pipeline
    pipeline = PreprocessingPipeline(use_precomputed=False) # Force live calculation
    features, ids = pipeline.process(dataframes, keep_sk_id=True)
    
    # Load Model
    model_path = Path("models/production_model.pkl")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    # Predict
    # Drop SK_ID_CURR if present
    if 'SK_ID_CURR' in features.columns:
        X = features.drop(columns=['SK_ID_CURR'])
    else:
        X = features
        
    # Ensure column order matches model
    # LightGBM might complain about column names if they contain special chars, but pipeline cleans them.
    # We should match names if possible.
    # The model.feature_name() gives required names.
    
    probs = model.predict_proba(X)[:, 1]
    
    results = pd.DataFrame({
        'SK_ID_CURR': ids,
        'probability': probs
    })
    
    return results

def main():
    print("="*80)
    print("GAP ANALYSIS: Original vs Anonymized")
    print("="*80)
    
    # 1. Reset to Original Data
    print("\n[Step 1] Regenerating Original Data...")
    # Clear directory first
    for f in Path("data/end_user_tests").glob("*"):
        if f.is_file(): f.unlink()
        
    run_script("scripts/create_relational_test_files.py")
    run_script("scripts/consolidate_test_files.py")
    
    # 2. Predict on Original
    preds_original = get_predictions("Original Data")
    
    # 3. Anonymize
    print("\n[Step 3] Anonymizing Data...")
    run_script("scripts/anonymize_end_user_tests.py")
    
    # Load mapping to compare correctly
    mapping = pd.read_csv("data/end_user_tests/id_mapping.csv")
    id_map = dict(zip(mapping['original_sk_id_curr'], mapping['new_sk_id_curr']))
    
    # 4. Predict on Anonymized
    preds_anon = get_predictions("Anonymized Data")
    
    # 5. Compare
    print("\n[Step 5] Comparing Results...")
    
    # Map original IDs to new IDs to align rows
    preds_original['new_id'] = preds_original['SK_ID_CURR'].map(id_map)
    
    # Merge
    merged = preds_original.merge(
        preds_anon, 
        left_on='new_id',
        right_on='SK_ID_CURR',
        suffixes=('_orig', '_anon')
    )
    
    merged['diff'] = merged['probability_orig'] - merged['probability_anon']
    merged['abs_diff'] = merged['diff'].abs()
    
    print("\n" + "="*80)
    print("GAP ANALYSIS RESULTS")
    print("="*80)
    print(merged[['SK_ID_CURR_orig', 'SK_ID_CURR_anon', 'probability_orig', 'probability_anon', 'diff']])
    
    print("\nStatistics:")
    print(merged['abs_diff'].describe())
    
    max_diff = merged['abs_diff'].max()
    if max_diff < 0.01:
        print("\n✅ PASSED: Differences are negligible (< 1%)")
    else:
        print(f"\n⚠️ WARNING: Max difference is {max_diff:.4f}")

if __name__ == "__main__":
    main()
