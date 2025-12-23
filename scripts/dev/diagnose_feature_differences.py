"""Diagnose why batch pipeline predictions differ from training predictions.
Compare feature values for specific samples.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle

import pandas as pd

from api.preprocessing_pipeline import PreprocessingPipeline


def main():
    print("="*80)
    print("DIAGNOSING FEATURE VALUE DIFFERENCES")
    print("="*80)

    # Load model
    with open('models/production_model.pkl', 'rb') as f:
        model = pickle.load(f)
    model_features = model.feature_name_

    # Load training data with SK_ID
    app_train = pd.read_csv('data/application_train.csv')
    train_preds = pd.read_csv('results/train_predictions.csv')
    train_preds['SK_ID_CURR'] = app_train['SK_ID_CURR'].values

    # Get a HIGH risk sample
    high_risk = train_preds[train_preds['PROBABILITY'] >= 0.50].head(1)
    sample_id = high_risk['SK_ID_CURR'].values[0]
    sample_prob = high_risk['PROBABILITY'].values[0]

    print(f"\nSample ID: {sample_id}")
    print(f"Training probability: {sample_prob:.1%}")

    # Check if we have pre-processed training features
    processed_path = Path('data/processed')
    train_features_path = processed_path / 'X_train.csv'

    if train_features_path.exists():
        print(f"\nLoading pre-processed training features from {train_features_path}...")
        X_train = pd.read_csv(train_features_path)

        # Match by row index (assuming same order as train_preds)
        sample_idx = train_preds[train_preds['SK_ID_CURR'] == sample_id].index[0]
        train_sample_features = X_train.iloc[sample_idx]

        print(f"Training features shape: {X_train.shape}")
        print(f"Sample row index: {sample_idx}")

        # Verify by making prediction
        train_pred = model.predict_proba(train_sample_features.values.reshape(1, -1))[0, 1]
        print(f"Prediction from training features: {train_pred:.1%}")

    else:
        print(f"\n[WARNING] {train_features_path} not found - cannot compare training features")
        train_sample_features = None

    # Now create features through batch pipeline
    print("\n" + "-"*80)
    print("Processing through batch pipeline...")
    print("-"*80)

    # Load all auxiliary data for this sample
    app_sample = app_train[app_train['SK_ID_CURR'] == sample_id]

    bureau = pd.read_csv('data/bureau.csv')
    bureau_sample = bureau[bureau['SK_ID_CURR'] == sample_id]

    bureau_ids = bureau_sample['SK_ID_BUREAU'].tolist()
    bureau_balance = pd.read_csv('data/bureau_balance.csv')
    bureau_balance_sample = bureau_balance[bureau_balance['SK_ID_BUREAU'].isin(bureau_ids)]

    prev_app = pd.read_csv('data/previous_application.csv')
    prev_sample = prev_app[prev_app['SK_ID_CURR'] == sample_id]

    prev_ids = prev_sample['SK_ID_PREV'].tolist()
    pos_cash = pd.read_csv('data/POS_CASH_balance.csv')
    pos_sample = pos_cash[pos_cash['SK_ID_PREV'].isin(prev_ids)]

    cc_balance = pd.read_csv('data/credit_card_balance.csv')
    cc_sample = cc_balance[cc_balance['SK_ID_PREV'].isin(prev_ids)]

    installments = pd.read_csv('data/installments_payments.csv')
    inst_sample = installments[installments['SK_ID_PREV'].isin(prev_ids)]

    print(f"\nAuxiliary data for sample {sample_id}:")
    print(f"  Bureau: {len(bureau_sample)} records")
    print(f"  Bureau Balance: {len(bureau_balance_sample)} records")
    print(f"  Previous App: {len(prev_sample)} records")
    print(f"  POS Cash: {len(pos_sample)} records")
    print(f"  CC Balance: {len(cc_sample)} records")
    print(f"  Installments: {len(inst_sample)} records")

    # Process through pipeline
    pipeline = PreprocessingPipeline()
    dataframes = {
        'application.csv': app_sample,
        'bureau.csv': bureau_sample if len(bureau_sample) > 0 else None,
        'bureau_balance.csv': bureau_balance_sample if len(bureau_balance_sample) > 0 else None,
        'previous_application.csv': prev_sample if len(prev_sample) > 0 else None,
        'POS_CASH_balance.csv': pos_sample if len(pos_sample) > 0 else None,
        'credit_card_balance.csv': cc_sample if len(cc_sample) > 0 else None,
        'installments_payments.csv': inst_sample if len(inst_sample) > 0 else None,
    }

    features_df, _ = pipeline.process(dataframes)

    if 'SK_ID_CURR' in features_df.columns:
        features_df = features_df.drop('SK_ID_CURR', axis=1)

    batch_sample_features = features_df.iloc[0]

    # Make prediction
    batch_pred = model.predict_proba(batch_sample_features.values.reshape(1, -1))[0, 1]
    print(f"\nBatch pipeline prediction: {batch_pred:.1%}")
    print(f"Difference from training: {batch_pred - sample_prob:.1%}")

    # Compare feature values
    if train_sample_features is not None:
        print("\n" + "="*80)
        print("FEATURE VALUE COMPARISON (top differences)")
        print("="*80)

        differences = []
        for feat in model_features:
            train_val = train_sample_features[feat] if feat in train_sample_features.index else 0
            batch_val = batch_sample_features[feat] if feat in batch_sample_features.index else 0

            if pd.isna(train_val):
                train_val = 0
            if pd.isna(batch_val):
                batch_val = 0

            diff = abs(float(train_val) - float(batch_val))
            if diff > 0.001:  # Only show significant differences
                differences.append({
                    'feature': feat,
                    'train': float(train_val),
                    'batch': float(batch_val),
                    'diff': diff
                })

        # Sort by difference
        differences.sort(key=lambda x: x['diff'], reverse=True)

        print(f"\nFound {len(differences)} features with differences > 0.001")
        print("\nTop 30 largest differences:")
        print(f"{'Feature':<50} {'Training':>12} {'Batch':>12} {'Diff':>12}")
        print("-"*86)
        for d in differences[:30]:
            print(f"{d['feature']:<50} {d['train']:>12.4f} {d['batch']:>12.4f} {d['diff']:>12.4f}")


if __name__ == '__main__':
    main()
