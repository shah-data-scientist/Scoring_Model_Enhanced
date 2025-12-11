"""Compare training predictions vs batch API predictions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pickle
from api.preprocessing_pipeline import PreprocessingPipeline

# Load all data
print("Loading data...")
app = pd.read_csv('data/application_train.csv')
bureau = pd.read_csv('data/bureau.csv')
bureau_balance = pd.read_csv('data/bureau_balance.csv')
prev_app = pd.read_csv('data/previous_application.csv')
pos_cash = pd.read_csv('data/POS_CASH_balance.csv')
cc_balance = pd.read_csv('data/credit_card_balance.csv')
installments = pd.read_csv('data/installments_payments.csv')
train_preds = pd.read_csv('results/train_predictions.csv')
train_preds['SK_ID_CURR'] = app['SK_ID_CURR'].values

# Get HIGH risk samples
high_risk = train_preds[train_preds['PROBABILITY'] >= 0.50].head(5)
print('\nTraining predictions for HIGH risk samples:')
print(high_risk[['SK_ID_CURR', 'PROBABILITY', 'TARGET']].to_string())

ids = high_risk['SK_ID_CURR'].tolist()

# Filter ALL auxiliary data for these IDs
app_samples = app[app['SK_ID_CURR'].isin(ids)]
bureau_samples = bureau[bureau['SK_ID_CURR'].isin(ids)]
bureau_ids = bureau_samples['SK_ID_BUREAU'].tolist()
bureau_balance_samples = bureau_balance[bureau_balance['SK_ID_BUREAU'].isin(bureau_ids)]
prev_samples = prev_app[prev_app['SK_ID_CURR'].isin(ids)]
prev_ids = prev_samples['SK_ID_PREV'].tolist()
pos_samples = pos_cash[pos_cash['SK_ID_PREV'].isin(prev_ids)]
cc_samples = cc_balance[cc_balance['SK_ID_PREV'].isin(prev_ids)]
inst_samples = installments[installments['SK_ID_PREV'].isin(prev_ids)]

print(f'\nAuxiliary data counts:')
print(f'  Bureau: {len(bureau_samples)}')
print(f'  Bureau Balance: {len(bureau_balance_samples)}')
print(f'  Previous App: {len(prev_samples)}')
print(f'  POS Cash: {len(pos_samples)}')
print(f'  CC Balance: {len(cc_samples)}')
print(f'  Installments: {len(inst_samples)}')

# Process with ALL auxiliary data
print("\nProcessing through batch pipeline...")
pipeline = PreprocessingPipeline()
dataframes = {
    'application.csv': app_samples,
    'bureau.csv': bureau_samples,
    'bureau_balance.csv': bureau_balance_samples,
    'previous_application.csv': prev_samples,
    'POS_CASH_balance.csv': pos_samples,
    'credit_card_balance.csv': cc_samples,
    'installments_payments.csv': inst_samples
}
features, sk_ids = pipeline.process(dataframes)

# Drop SK_ID_CURR if present
if 'SK_ID_CURR' in features.columns:
    features = features.drop('SK_ID_CURR', axis=1)

print(f"\nFeatures shape: {features.shape}")

# Load model and predict
with open('models/production_model.pkl', 'rb') as f:
    model = pickle.load(f)

proba = model.predict_proba(features)[:, 1]

print('\n' + '='*70)
print('COMPARISON: Training vs Batch API predictions (with full aux data):')
print('='*70)
for sk_id, prob in zip(sk_ids, proba):
    train_prob = train_preds[train_preds['SK_ID_CURR'] == sk_id]['PROBABILITY'].values[0]
    diff = prob - train_prob
    print(f'  ID {sk_id}: Training={train_prob:.1%}, Batch={prob:.1%}, Diff={diff:+.1%}')
