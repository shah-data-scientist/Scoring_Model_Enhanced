"""Fast test of lookup-based preprocessing."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pickle
from api.preprocessing_pipeline import PreprocessingPipeline

print("="*80)
print("TESTING LOOKUP-BASED PREPROCESSING")
print("="*80)

# Initialize pipeline (will load precomputed features)
pipeline = PreprocessingPipeline()

# Create a small test with just a few sample IDs from training data
sample_ids = [100002, 100003, 100004]  # Known training IDs

# Create minimal application DataFrame
app_df = pd.DataFrame({
    'SK_ID_CURR': sample_ids
})

# Create minimal empty auxiliary dataframes (won't be used for known IDs)
empty_df = pd.DataFrame()

dataframes = {
    'application.csv': app_df,
    'bureau.csv': empty_df,
    'bureau_balance.csv': empty_df,
    'previous_application.csv': empty_df,
    'POS_CASH_balance.csv': empty_df,
    'credit_card_balance.csv': empty_df,
    'installments_payments.csv': empty_df
}

# Process through pipeline
features_df, sk_ids = pipeline.process(dataframes)

print(f"\nResult: {len(features_df)} applications processed")
print(f"Features shape: {features_df.shape}")

# Load model and make predictions
print("\nMaking predictions...")
with open('models/production_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Drop SK_ID_CURR for prediction
features_only = features_df.drop(columns=['SK_ID_CURR']) if 'SK_ID_CURR' in features_df.columns else features_df

proba = model.predict_proba(features_only.values)[:, 1]

# Load training predictions for comparison
train_preds = pd.read_csv('results/train_predictions.csv')
app_train = pd.read_csv('data/application_train.csv')
train_preds['SK_ID_CURR'] = app_train['SK_ID_CURR'].values

print("\n" + "="*80)
print("COMPARISON: Training vs Lookup-Based Predictions")
print("="*80)

for i, app_id in enumerate(sample_ids):
    train_prob = train_preds[train_preds['SK_ID_CURR'] == app_id]['PROBABILITY'].values[0]
    batch_prob = proba[i]
    diff = abs(batch_prob - train_prob)

    match_str = "PERFECT MATCH!" if diff < 0.0001 else f"DIFF: {diff:.4f}"
    print(f"  ID {app_id}: Training={train_prob:.6f}, Lookup={batch_prob:.6f} - {match_str}")

print("\n" + "="*80)
if all(abs(proba[i] - train_preds[train_preds['SK_ID_CURR'] == app_id]['PROBABILITY'].values[0]) < 0.0001
       for i, app_id in enumerate(sample_ids)):
    print("[SUCCESS] Lookup-based preprocessing produces 100% accurate predictions!")
else:
    print("[WARNING] Some predictions don't match exactly")
print("="*80)
