"""Quick feature comparison - single sample."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pickle
from api.preprocessing_pipeline import PreprocessingPipeline

# Load training features and predictions
X_train = pd.read_csv('data/processed/X_train.csv')
train_preds = pd.read_csv('results/train_predictions.csv')
app_train = pd.read_csv('data/application_train.csv')
train_preds['SK_ID_CURR'] = app_train['SK_ID_CURR'].values

# Get ONE high-risk sample
sample_id = 100008  # First HIGH risk sample
sample_idx = train_preds[train_preds['SK_ID_CURR'] == sample_id].index[0]
train_row = X_train.iloc[sample_idx]
train_prob = train_preds.iloc[sample_idx]['PROBABILITY']

print(f"Sample ID: {sample_id}")
print(f"Training probability: {train_prob:.1%}\n")

# Process through batch pipeline
print("Processing through batch pipeline...")
app = app_train[app_train['SK_ID_CURR'] == sample_id]
bureau = pd.read_csv('data/bureau.csv')
bureau = bureau[bureau['SK_ID_CURR'] == sample_id]
prev_app = pd.read_csv('data/previous_application.csv')
prev_app = prev_app[prev_app['SK_ID_CURR'] == sample_id]
pos_cash = pd.read_csv('data/POS_CASH_balance.csv')
prev_ids = prev_app['SK_ID_PREV'].tolist() if len(prev_app) > 0 else []
pos_cash = pos_cash[pos_cash['SK_ID_PREV'].isin(prev_ids)]
cc = pd.read_csv('data/credit_card_balance.csv')
cc = cc[cc['SK_ID_PREV'].isin(prev_ids)]
inst = pd.read_csv('data/installments_payments.csv')
inst = inst[inst['SK_ID_PREV'].isin(prev_ids)]
bureau_balance = pd.read_csv('data/bureau_balance.csv')
bureau_ids = bureau['SK_ID_BUREAU'].tolist() if len(bureau) > 0 else []
bureau_balance = bureau_balance[bureau_balance['SK_ID_BUREAU'].isin(bureau_ids)]

pipeline = PreprocessingPipeline()
features, _ = pipeline.process({
    'application.csv': app,
    'bureau.csv': bureau,
    'bureau_balance.csv': bureau_balance,
    'previous_application.csv': prev_app,
    'POS_CASH_balance.csv': pos_cash,
    'credit_card_balance.csv': cc,
    'installments_payments.csv': inst
})

batch_row = features.iloc[0]
if 'SK_ID_CURR' in batch_row.index:
    batch_row = batch_row.drop('SK_ID_CURR')

# Make prediction
with open('models/production_model.pkl', 'rb') as f:
    model = pickle.load(f)

batch_prob = model.predict_proba(batch_row.values.reshape(1, -1))[0, 1]
print(f"Batch probability: {batch_prob:.1%}")
print(f"Difference: {batch_prob - train_prob:+.1%}\n")

# Compare top 20 most important features
feat_importance = pd.Series(model.feature_importances_, index=model.feature_name_)
top_features = feat_importance.nlargest(20).index.tolist()

print("="*70)
print("TOP 20 MOST IMPORTANT FEATURES COMPARISON")
print("="*70)
print(f"{'Feature':<30} {'Training':>12} {'Batch':>12} {'Diff':>12}")
print("-"*70)

for feat in top_features:
    train_val = train_row[feat] if feat in train_row.index else 0
    batch_val = batch_row[feat] if feat in batch_row.index else 0

    if pd.isna(train_val):
        train_val = 0
    if pd.isna(batch_val):
        batch_val = 0

    diff = float(batch_val) - float(train_val)

    # Only print if there's a difference
    if abs(diff) > 0.001:
        print(f"{feat:<30} {float(train_val):>12.4f} {float(batch_val):>12.4f} {diff:>12.4f}")
