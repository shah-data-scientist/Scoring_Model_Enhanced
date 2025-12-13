"""Create end user test data with verified API predictions.

This script:
1. Samples a large pool of candidates from training data
2. Runs them through the actual preprocessing pipeline and model
3. Selects 10 applications with proper risk distribution (3 LOW, 3 MEDIUM, 4 HIGH)
4. Creates the test files with linked data
"""
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.preprocessing_pipeline import PreprocessingPipeline

DATA_DIR = PROJECT_ROOT / 'data'
END_USER_TESTS_DIR = DATA_DIR / 'end_user_tests'
SAMPLES_DIR = DATA_DIR / 'samples'


def load_model():
    """Load the production model."""
    model_file = PROJECT_ROOT / "models" / "production_model.pkl"
    with open(model_file, 'rb') as f:
        return pickle.load(f)


def get_api_predictions(sample_ids, model):
    """Get predictions using the actual API preprocessing pipeline."""
    # Load raw data
    print(f"\nLoading data for {len(sample_ids)} candidates...")

    app_train = pd.read_csv(DATA_DIR / 'application_train.csv')
    app_samples = app_train[app_train['SK_ID_CURR'].isin(sample_ids)].copy()

    bureau = pd.read_csv(DATA_DIR / 'bureau.csv')
    bureau_samples = bureau[bureau['SK_ID_CURR'].isin(sample_ids)]

    bureau_ids = bureau_samples['SK_ID_BUREAU'].tolist() if len(bureau_samples) > 0 else []
    bureau_balance = pd.read_csv(DATA_DIR / 'bureau_balance.csv')
    bureau_balance_samples = bureau_balance[bureau_balance['SK_ID_BUREAU'].isin(bureau_ids)]

    prev_app = pd.read_csv(DATA_DIR / 'previous_application.csv')
    prev_app_samples = prev_app[prev_app['SK_ID_CURR'].isin(sample_ids)]

    prev_ids = prev_app_samples['SK_ID_PREV'].tolist() if len(prev_app_samples) > 0 else []

    pos_cash = pd.read_csv(DATA_DIR / 'POS_CASH_balance.csv')
    pos_cash_samples = pos_cash[pos_cash['SK_ID_PREV'].isin(prev_ids)]

    cc_balance = pd.read_csv(DATA_DIR / 'credit_card_balance.csv')
    cc_balance_samples = cc_balance[cc_balance['SK_ID_PREV'].isin(prev_ids)]

    installments = pd.read_csv(DATA_DIR / 'installments_payments.csv')
    installments_samples = installments[installments['SK_ID_PREV'].isin(prev_ids)]

    # Create files dict for preprocessing
    files = {
        'application.csv': app_samples,
        'bureau.csv': bureau_samples,
        'bureau_balance.csv': bureau_balance_samples,
        'previous_application.csv': prev_app_samples,
        'POS_CASH_balance.csv': pos_cash_samples,
        'credit_card_balance.csv': cc_balance_samples,
        'installments_payments.csv': installments_samples,
    }

    print("Preprocessing...")
    pipeline = PreprocessingPipeline()
    features, sk_ids = pipeline.process(files)

    # Remove SK_ID_CURR from features
    if 'SK_ID_CURR' in features.columns:
        features = features.drop(columns=['SK_ID_CURR'])

    # Get predictions
    probs = model.predict_proba(features)[:, 1]

    # Create results DataFrame
    results = pd.DataFrame({
        'SK_ID_CURR': sk_ids.values,
        'PROBABILITY': probs
    })

    return results


def main():
    print('=' * 60)
    print('Creating End User Tests with Verified API Predictions')
    print('=' * 60)

    # Load model
    print("\nLoading production model...")
    model = load_model()

    # Load training data to get candidate IDs
    print("\nLoading training data...")
    app_train = pd.read_csv(DATA_DIR / 'application_train.csv')

    # Focus on applications with TARGET=1 (defaults) - more likely to be high risk
    defaults = app_train[app_train['TARGET'] == 1]['SK_ID_CURR'].tolist()
    non_defaults = app_train[app_train['TARGET'] == 0]['SK_ID_CURR'].tolist()

    np.random.seed(42)
    # Sample more defaults to find high-risk candidates
    candidate_ids = (
        np.random.choice(defaults, size=min(300, len(defaults)), replace=False).tolist() +
        np.random.choice(non_defaults, size=200, replace=False).tolist()
    )

    # Get API predictions for candidates
    results = get_api_predictions(candidate_ids, model)

    print(f"\nPrediction distribution for {len(results)} candidates:")
    print(f"  LOW (< 15%):      {len(results[results['PROBABILITY'] < 0.15])}")
    print(f"  MEDIUM (15-30%):  {len(results[(results['PROBABILITY'] >= 0.15) & (results['PROBABILITY'] < 0.30)])}")
    print(f"  HIGH (>= 30%):    {len(results[results['PROBABILITY'] >= 0.30])}")

    # Select 3 LOW, 3 MEDIUM, 4 HIGH (with new thresholds)
    low_candidates = results[results['PROBABILITY'] < 0.15].nsmallest(3, 'PROBABILITY')
    medium_candidates = results[(results['PROBABILITY'] >= 0.15) & (results['PROBABILITY'] < 0.30)].sample(n=min(3, len(results[(results['PROBABILITY'] >= 0.15) & (results['PROBABILITY'] < 0.30)])), random_state=42)
    high_candidates = results[results['PROBABILITY'] >= 0.30].nlargest(4, 'PROBABILITY')

    # Handle cases where we don't have enough in each category
    n_low = len(low_candidates)
    n_med = len(medium_candidates)
    n_high = len(high_candidates)

    print("\nAvailable candidates:")
    print(f"  LOW:    {n_low}")
    print(f"  MEDIUM: {n_med}")
    print(f"  HIGH:   {n_high}")

    if n_low < 3 or n_med < 3 or n_high < 4:
        print("\nWARNING: Not enough candidates in all risk categories!")
        print("Need to sample more candidates or adjust selection criteria.")

        # Try with expanded medium range (25-55%)
        print("\nTrying with expanded ranges...")
        low_candidates = results[results['PROBABILITY'] < 0.12].nsmallest(3, 'PROBABILITY')
        medium_candidates = results[(results['PROBABILITY'] >= 0.12) & (results['PROBABILITY'] < 0.25)]
        if len(medium_candidates) >= 3:
            medium_candidates = medium_candidates.sample(n=3, random_state=42)
        high_candidates = results[results['PROBABILITY'] >= 0.25].nlargest(4, 'PROBABILITY')

        n_low = len(low_candidates)
        n_med = len(medium_candidates)
        n_high = len(high_candidates)
        print(f"  LOW (< 12%):    {n_low}")
        print(f"  MEDIUM (12-25%): {n_med}")
        print(f"  HIGH (>= 25%):  {n_high}")

    selected = pd.concat([low_candidates, medium_candidates, high_candidates])

    if len(selected) < 10:
        # Fill remainder with whatever we have
        remaining_needed = 10 - len(selected)
        already_selected = set(selected['SK_ID_CURR'].tolist())
        remaining = results[~results['SK_ID_CURR'].isin(already_selected)]
        # Get a mix sorted by probability to get variety
        remaining_sorted = remaining.sort_values('PROBABILITY')
        indices = np.linspace(0, len(remaining_sorted)-1, remaining_needed, dtype=int)
        additional = remaining_sorted.iloc[indices]
        selected = pd.concat([selected, additional])

    selected_ids = selected['SK_ID_CURR'].tolist()

    print(f"\n{'='*60}")
    print("SELECTED APPLICATIONS:")
    print('='*60)

    for _, row in selected.iterrows():
        prob = row['PROBABILITY']
        risk = 'LOW' if prob < 0.15 else ('MEDIUM' if prob < 0.30 else 'HIGH')
        decision = 'Default' if prob >= 0.30 else 'No Default'
        print(f"  ID {int(row['SK_ID_CURR'])}: {prob:.1%} - {risk} - {decision}")

    # Now create the test files
    print("\n" + '='*60)
    print("Creating test files...")
    print('='*60)

    # Reload data for selected IDs
    app_samples = app_train[app_train['SK_ID_CURR'].isin(selected_ids)]

    bureau = pd.read_csv(DATA_DIR / 'bureau.csv')
    bureau_samples = bureau[bureau['SK_ID_CURR'].isin(selected_ids)]

    bureau_ids = bureau_samples['SK_ID_BUREAU'].tolist()
    bureau_balance = pd.read_csv(DATA_DIR / 'bureau_balance.csv')
    bureau_balance_samples = bureau_balance[bureau_balance['SK_ID_BUREAU'].isin(bureau_ids)]

    prev_app = pd.read_csv(DATA_DIR / 'previous_application.csv')
    prev_app_samples = prev_app[prev_app['SK_ID_CURR'].isin(selected_ids)]

    prev_ids = prev_app_samples['SK_ID_PREV'].tolist()

    pos_cash = pd.read_csv(DATA_DIR / 'POS_CASH_balance.csv')
    pos_cash_samples = pos_cash[pos_cash['SK_ID_PREV'].isin(prev_ids)]

    cc_balance = pd.read_csv(DATA_DIR / 'credit_card_balance.csv')
    cc_balance_samples = cc_balance[cc_balance['SK_ID_PREV'].isin(prev_ids)]

    installments = pd.read_csv(DATA_DIR / 'installments_payments.csv')
    installments_samples = installments[installments['SK_ID_PREV'].isin(prev_ids)]

    # Create directory and save files
    END_USER_TESTS_DIR.mkdir(parents=True, exist_ok=True)

    app_samples.to_csv(END_USER_TESTS_DIR / 'application.csv', index=False)
    print(f"  application.csv: {len(app_samples)} applications")

    bureau_samples.to_csv(END_USER_TESTS_DIR / 'bureau.csv', index=False)
    print(f"  bureau.csv: {len(bureau_samples)} records")

    bureau_balance_samples.to_csv(END_USER_TESTS_DIR / 'bureau_balance.csv', index=False)
    print(f"  bureau_balance.csv: {len(bureau_balance_samples)} records")

    prev_app_samples.to_csv(END_USER_TESTS_DIR / 'previous_application.csv', index=False)
    print(f"  previous_application.csv: {len(prev_app_samples)} records")

    pos_cash_samples.to_csv(END_USER_TESTS_DIR / 'POS_CASH_balance.csv', index=False)
    print(f"  POS_CASH_balance.csv: {len(pos_cash_samples)} records")

    cc_balance_samples.to_csv(END_USER_TESTS_DIR / 'credit_card_balance.csv', index=False)
    print(f"  credit_card_balance.csv: {len(cc_balance_samples)} records")

    installments_samples.to_csv(END_USER_TESTS_DIR / 'installments_payments.csv', index=False)
    print(f"  installments_payments.csv: {len(installments_samples)} records")

    # Count final distribution (with new thresholds)
    low_count = len(selected[selected['PROBABILITY'] < 0.15])
    med_count = len(selected[(selected['PROBABILITY'] >= 0.15) & (selected['PROBABILITY'] < 0.30)])
    high_count = len(selected[selected['PROBABILITY'] >= 0.30])

    # Create README
    readme_content = f"""# End User Tests

This folder contains test data with 10 applications representing different risk levels.
**These predictions have been verified against the actual API.**

## Risk Distribution

| Risk Level | Count | Probability Range | Expected Outcome |
|------------|-------|-------------------|------------------|
| LOW        | {low_count}     | < 15%             | No Default       |
| MEDIUM     | {med_count}     | 15% - 30%         | Mixed            |
| HIGH       | {high_count}     | >= 30%            | Default          |

## Files

- `application.csv` - Main application data ({len(app_samples)} applications)
- `bureau.csv` - Bureau records ({len(bureau_samples)} records)
- `bureau_balance.csv` - Bureau balance history ({len(bureau_balance_samples)} records)
- `previous_application.csv` - Previous loan applications ({len(prev_app_samples)} records)
- `POS_CASH_balance.csv` - POS and cash balance ({len(pos_cash_samples)} records)
- `credit_card_balance.csv` - Credit card balance ({len(cc_balance_samples)} records)
- `installments_payments.csv` - Installment payments ({len(installments_samples)} records)

## Usage

Upload all CSV files to the Batch Predictions page to test the model.
"""
    with open(END_USER_TESTS_DIR / 'README.md', 'w') as f:
        f.write(readme_content)
    print("  README.md created")

    # Create ZIP
    print("\nCreating ZIP archive...")
    shutil.make_archive(
        str(SAMPLES_DIR / 'end_user_tests'),
        'zip',
        END_USER_TESTS_DIR
    )
    print("  end_user_tests.zip created")

    print("\n" + '='*60)
    print(f"SUCCESS: Created {len(selected)} test applications")
    print(f"  LOW: {low_count}, MEDIUM: {med_count}, HIGH: {high_count}")
    print('='*60)


if __name__ == '__main__':
    main()
