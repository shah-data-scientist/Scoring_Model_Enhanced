"""
Pre-compute features for all training applications.

This script generates the exact same features used during model training
and saves them for lookup during batch predictions. This ensures that
batch predictions match training predictions exactly.
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from src.data_preprocessing import load_data
from src.feature_engineering import encode_categorical_features, clean_column_names
from src.domain_features import create_domain_features

def main():
    print("="*80)
    print("PRE-COMPUTING FEATURES FOR ALL TRAINING APPLICATIONS")
    print("="*80)
    
    # Load and aggregate all data (same as training)
    print("\nStep 1: Loading and aggregating all data sources...")
    train_df, test_df = load_data(
        data_path=str(PROJECT_ROOT / 'data'),
        use_all_data_sources=True,
        validate=True
    )
    
    print(f"\nTrain shape after aggregation: {train_df.shape}")
    
    # Save SK_ID_CURR before processing
    sk_ids = train_df['SK_ID_CURR'].copy()
    target = train_df['TARGET'].copy()
    
    # Step 2: Create domain features
    print("\nStep 2: Creating domain features...")
    train_df = create_domain_features(train_df)
    print(f"  Shape after domain features: {train_df.shape}")
    
    # Step 3: Encode categorical features
    print("\nStep 3: Encoding categorical features...")
    # We need a dummy test_df for the encoder
    train_df, _ = encode_categorical_features(train_df, train_df.head(1).copy())
    print(f"  Shape after encoding: {train_df.shape}")
    
    # Step 4: Clean column names
    print("\nStep 4: Cleaning column names...")
    train_df = clean_column_names(train_df)
    
    # Step 5: Handle missing values with median imputation
    print("\nStep 5: Imputing missing values...")
    numerical_cols = train_df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if train_df[col].isnull().any():
            median_val = train_df[col].median()
            train_df[col] = train_df[col].fillna(median_val if not pd.isna(median_val) else 0)
    
    # Step 6: Load expected features from model
    print("\nStep 6: Aligning with model features...")
    import pickle
    model_path = PROJECT_ROOT / 'models' / 'production_model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    expected_features = model.feature_name_
    print(f"  Model expects {len(expected_features)} features")
    
    # Ensure SK_ID_CURR and TARGET are not in features
    if 'SK_ID_CURR' in train_df.columns:
        train_df = train_df.drop(columns=['SK_ID_CURR'])
    if 'TARGET' in train_df.columns:
        train_df = train_df.drop(columns=['TARGET'])
    
    current_features = set(train_df.columns)
    expected_set = set(expected_features)
    
    # Add missing features
    missing = expected_set - current_features
    if missing:
        print(f"  Adding {len(missing)} missing features")
        for feat in missing:
            train_df[feat] = 0
    
    # Drop extra features
    extra = current_features - expected_set
    if extra:
        print(f"  Dropping {len(extra)} extra features")
        train_df = train_df.drop(columns=list(extra))
    
    # Reorder to match model
    train_df = train_df[expected_features]
    
    # Add back SK_ID_CURR as index
    train_df.insert(0, 'SK_ID_CURR', sk_ids.values)
    
    print(f"\nFinal shape: {train_df.shape}")
    
    # Save to parquet for fast loading
    output_path = PROJECT_ROOT / 'data' / 'processed' / 'precomputed_features.parquet'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving to {output_path}...")
    train_df.to_parquet(output_path, index=False)
    
    # Also save predictions for verification
    print("\nGenerating predictions for verification...")
    features_only = train_df.drop(columns=['SK_ID_CURR'])
    probabilities = model.predict_proba(features_only)[:, 1]
    
    predictions_df = pd.DataFrame({
        'SK_ID_CURR': sk_ids.values,
        'TARGET': target.values,
        'PROBABILITY': probabilities
    })
    
    predictions_path = PROJECT_ROOT / 'data' / 'processed' / 'precomputed_predictions.csv'
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Saved predictions to {predictions_path}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total applications: {len(train_df)}")
    print(f"Features per application: {len(expected_features)}")
    print(f"\nPrediction distribution:")
    print(f"  LOW (< 30%):     {(probabilities < 0.30).sum():,} ({(probabilities < 0.30).mean():.1%})")
    print(f"  MEDIUM (30-50%): {((probabilities >= 0.30) & (probabilities < 0.50)).sum():,} ({((probabilities >= 0.30) & (probabilities < 0.50)).mean():.1%})")
    print(f"  HIGH (>= 50%):   {(probabilities >= 0.50).sum():,} ({(probabilities >= 0.50).mean():.1%})")
    
    print("\n[SUCCESS] Pre-computed features saved!")
    print("="*80)

if __name__ == '__main__':
    main()
