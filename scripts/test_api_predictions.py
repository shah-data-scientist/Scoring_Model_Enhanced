"""
Test what the API actually predicts for specific applications.
This helps verify that end user tests produce expected results.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.preprocessing_pipeline import PreprocessingPipeline
import mlflow

def test_predictions():
    """Test API predictions for end user test data."""
    end_user_dir = PROJECT_ROOT / 'data' / 'end_user_tests'
    
    # Load end user test data - use filenames as keys (as expected by pipeline)
    files = {
        'application.csv': pd.read_csv(end_user_dir / 'application.csv'),
        'bureau.csv': pd.read_csv(end_user_dir / 'bureau.csv'),
        'bureau_balance.csv': pd.read_csv(end_user_dir / 'bureau_balance.csv'),
        'previous_application.csv': pd.read_csv(end_user_dir / 'previous_application.csv'),
        'POS_CASH_balance.csv': pd.read_csv(end_user_dir / 'POS_CASH_balance.csv'),
        'credit_card_balance.csv': pd.read_csv(end_user_dir / 'credit_card_balance.csv'),
        'installments_payments.csv': pd.read_csv(end_user_dir / 'installments_payments.csv'),
    }
    
    print('Loaded files:')
    for name, df in files.items():
        print(f'  {name}: {len(df)} rows')
    
    # Preprocess
    print('\nPreprocessing...')
    pipeline = PreprocessingPipeline()
    features, sk_ids = pipeline.process(files)
    
    # Remove SK_ID_CURR from features if present (model needs 189 features)
    if 'SK_ID_CURR' in features.columns:
        features = features.drop(columns=['SK_ID_CURR'])
    
    print(f'Preprocessed features shape: {features.shape}')
    
    app_ids = sk_ids.tolist()
    
    # Load model (same as API)
    model_file = PROJECT_ROOT / "models" / "production_model.pkl"
    
    if model_file.exists():
        print(f'\nLoading model from {model_file}')
        import pickle
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        probs = model.predict_proba(features)[:, 1]
        
        print('\n' + '=' * 60)
        print('Actual API Predictions for End User Tests')
        print('=' * 60)
        
        results = []
        for app_id, prob in zip(app_ids, probs):
            risk = 'LOW' if prob < 0.3 else ('MEDIUM' if prob < 0.5 else 'HIGH')
            decision = 'Default' if prob >= 0.5 else 'No Default'
            results.append({'ID': app_id, 'Probability': prob, 'Risk': risk, 'Decision': decision})
            print(f'  ID {app_id}: {prob:.1%} probability - {risk} - {decision}')
        
        # Count by risk level
        print('\n' + '-' * 60)
        low_count = sum(1 for r in results if r['Risk'] == 'LOW')
        med_count = sum(1 for r in results if r['Risk'] == 'MEDIUM')
        high_count = sum(1 for r in results if r['Risk'] == 'HIGH')
        print(f'Summary: {low_count} LOW, {med_count} MEDIUM, {high_count} HIGH')
        
        return results
    else:
        print(f'Model not found at {model_file}')
        return None


if __name__ == '__main__':
    test_predictions()
