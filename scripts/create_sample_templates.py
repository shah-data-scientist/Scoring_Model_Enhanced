"""
Script to create sample data templates and end user test data.

This script creates:
1. Sample templates (20 applications) from training data
2. End User Tests folder (10 applications) for testing
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
END_USER_TESTS_DIR = DATA_DIR / "end_user_tests"


def create_sample_templates():
    """Create sample templates with 20 applications from training data."""
    print("Creating sample templates...")
    
    # Read training data
    app_train = pd.read_csv(DATA_DIR / "application_train.csv")
    
    # Select 20 diverse samples (mix of default and non-default)
    default_samples = app_train[app_train['TARGET'] == 1].sample(n=10, random_state=42)
    non_default_samples = app_train[app_train['TARGET'] == 0].sample(n=10, random_state=42)
    samples = pd.concat([default_samples, non_default_samples]).sample(frac=1, random_state=42)
    
    # Get the SK_ID_CURR values for linked data
    sample_ids = samples['SK_ID_CURR'].tolist()
    
    # Ensure samples directory exists
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save application data
    samples.to_csv(SAMPLES_DIR / "application.csv", index=False)
    print(f"  Created application.csv with {len(samples)} applications")
    
    # Process linked data files
    linked_files = {
        'bureau.csv': 'SK_ID_CURR',
        'previous_application.csv': 'SK_ID_CURR',
    }
    
    for file_name, id_column in linked_files.items():
        src_file = DATA_DIR / file_name
        if src_file.exists():
            df = pd.read_csv(src_file)
            filtered = df[df[id_column].isin(sample_ids)]
            filtered.to_csv(SAMPLES_DIR / file_name, index=False)
            print(f"  Created {file_name} with {len(filtered)} records")
    
    # For bureau_balance, need to first get bureau IDs
    bureau_file = SAMPLES_DIR / "bureau.csv"
    if bureau_file.exists():
        bureau_df = pd.read_csv(bureau_file)
        bureau_ids = bureau_df['SK_ID_BUREAU'].tolist()
        
        bureau_balance_src = DATA_DIR / "bureau_balance.csv"
        if bureau_balance_src.exists():
            df = pd.read_csv(bureau_balance_src)
            filtered = df[df['SK_ID_BUREAU'].isin(bureau_ids)]
            filtered.to_csv(SAMPLES_DIR / "bureau_balance.csv", index=False)
            print(f"  Created bureau_balance.csv with {len(filtered)} records")
    
    # For other files linked to previous_application
    prev_app_file = SAMPLES_DIR / "previous_application.csv"
    if prev_app_file.exists():
        prev_app_df = pd.read_csv(prev_app_file)
        prev_ids = prev_app_df['SK_ID_PREV'].tolist()
        
        prev_linked_files = [
            'POS_CASH_balance.csv',
            'credit_card_balance.csv', 
            'installments_payments.csv'
        ]
        
        for file_name in prev_linked_files:
            src_file = DATA_DIR / file_name
            if src_file.exists():
                df = pd.read_csv(src_file)
                filtered = df[df['SK_ID_PREV'].isin(prev_ids)]
                filtered.to_csv(SAMPLES_DIR / file_name, index=False)
                print(f"  Created {file_name} with {len(filtered)} records")
    
    print(f"\nSample templates created in: {SAMPLES_DIR}")
    return sample_ids


def create_end_user_tests():
    """Create end user test data with 9 applications: 3 low, 3 medium, 3 high risk.
    
    Risk levels based on default probability:
    - LOW: probability < 0.3 (non-default)
    - MEDIUM: 0.3 <= probability < 0.5 (mix)
    - HIGH: probability >= 0.5 (default)
    """
    print("\nCreating end user test data...")
    
    # Read training data with predictions if available
    app_train = pd.read_csv(DATA_DIR / "application_train.csv")
    
    # Load predictions to get probabilities
    results_dir = PROJECT_ROOT / "results"
    pred_path = results_dir / "train_predictions.csv"
    
    if pred_path.exists():
        predictions = pd.read_csv(pred_path)
        # Merge to get probabilities
        if 'SK_ID_CURR' in predictions.columns:
            app_train = app_train.merge(predictions[['SK_ID_CURR', 'PROBABILITY']], on='SK_ID_CURR', how='left')
        elif 'PROBABILITY' in predictions.columns:
            # Assume same order
            app_train['PROBABILITY'] = predictions['PROBABILITY'].values[:len(app_train)]
    else:
        # If no predictions, estimate based on TARGET
        print("  Warning: No predictions file found. Using TARGET as proxy for risk.")
        # Create synthetic probabilities based on TARGET
        np.random.seed(456)
        app_train['PROBABILITY'] = np.where(
            app_train['TARGET'] == 1,
            np.random.uniform(0.4, 0.9, len(app_train)),  # Defaults have higher prob
            np.random.uniform(0.05, 0.4, len(app_train))  # Non-defaults have lower prob
        )
    
    # Fill missing probabilities
    app_train['PROBABILITY'] = app_train['PROBABILITY'].fillna(0.5)
    
    # Select samples based on risk levels
    # LOW risk: probability < 0.3 (3 samples)
    low_risk = app_train[app_train['PROBABILITY'] < 0.3].sample(n=3, random_state=456)
    
    # MEDIUM risk: 0.3 <= probability < 0.5 (3 samples)
    medium_risk = app_train[
        (app_train['PROBABILITY'] >= 0.3) & (app_train['PROBABILITY'] < 0.5)
    ].sample(n=3, random_state=456)
    
    # HIGH risk: probability >= 0.5 (3 samples)
    high_risk = app_train[app_train['PROBABILITY'] >= 0.5].sample(n=3, random_state=456)
    
    # Combine all samples
    samples = pd.concat([low_risk, medium_risk, high_risk]).sample(frac=1, random_state=456)
    
    # Remove PROBABILITY column before saving (it's not in original data)
    if 'PROBABILITY' in samples.columns:
        samples = samples.drop(columns=['PROBABILITY'])
    
    # Get the SK_ID_CURR values
    sample_ids = samples['SK_ID_CURR'].tolist()
    
    print(f"  Selected 9 applications:")
    print(f"    - 3 LOW risk (probability < 30%)")
    print(f"    - 3 MEDIUM risk (30-50%)")
    print(f"    - 3 HIGH risk (probability >= 50%)")
    
    # Ensure end user tests directory exists
    END_USER_TESTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save application data
    samples.to_csv(END_USER_TESTS_DIR / "application.csv", index=False)
    print(f"  Created application.csv with {len(samples)} applications")
    
    # Process linked data files
    linked_files = {
        'bureau.csv': 'SK_ID_CURR',
        'previous_application.csv': 'SK_ID_CURR',
    }
    
    for file_name, id_column in linked_files.items():
        src_file = DATA_DIR / file_name
        if src_file.exists():
            df = pd.read_csv(src_file)
            filtered = df[df[id_column].isin(sample_ids)]
            filtered.to_csv(END_USER_TESTS_DIR / file_name, index=False)
            print(f"  Created {file_name} with {len(filtered)} records")
    
    # For bureau_balance
    bureau_file = END_USER_TESTS_DIR / "bureau.csv"
    if bureau_file.exists():
        bureau_df = pd.read_csv(bureau_file)
        bureau_ids = bureau_df['SK_ID_BUREAU'].tolist()
        
        bureau_balance_src = DATA_DIR / "bureau_balance.csv"
        if bureau_balance_src.exists():
            df = pd.read_csv(bureau_balance_src)
            filtered = df[df['SK_ID_BUREAU'].isin(bureau_ids)]
            filtered.to_csv(END_USER_TESTS_DIR / "bureau_balance.csv", index=False)
            print(f"  Created bureau_balance.csv with {len(filtered)} records")
    
    # For files linked to previous_application
    prev_app_file = END_USER_TESTS_DIR / "previous_application.csv"
    if prev_app_file.exists():
        prev_app_df = pd.read_csv(prev_app_file)
        prev_ids = prev_app_df['SK_ID_PREV'].tolist()
        
        prev_linked_files = [
            'POS_CASH_balance.csv',
            'credit_card_balance.csv',
            'installments_payments.csv'
        ]
        
        for file_name in prev_linked_files:
            src_file = DATA_DIR / file_name
            if src_file.exists():
                df = pd.read_csv(src_file)
                filtered = df[df['SK_ID_PREV'].isin(prev_ids)]
                filtered.to_csv(END_USER_TESTS_DIR / file_name, index=False)
                print(f"  Created {file_name} with {len(filtered)} records")
    
    print(f"\nEnd user test data created in: {END_USER_TESTS_DIR}")
    return sample_ids


def create_readme_files():
    """Create README files for sample directories."""
    
    # Sample templates README
    sample_readme = """# Sample Templates

This folder contains sample data templates with 20 applications for testing batch predictions.

## Files Included

- `application.csv` - Main application data (20 applications)
- `bureau.csv` - Bureau credit data for these applications
- `bureau_balance.csv` - Monthly balance data from bureau
- `previous_application.csv` - Previous credit applications
- `POS_CASH_balance.csv` - POS cash loan balance
- `credit_card_balance.csv` - Credit card balance data
- `installments_payments.csv` - Payment installments data

## Usage

1. Download the files you need
2. Upload to the Batch Predictions page
3. Process to get predictions with SHAP explanations

## Data Description

These samples include a mix of:
- 10 applications that defaulted (TARGET=1)
- 10 applications that did not default (TARGET=0)

This provides a balanced test set for evaluating the model's predictions.
"""
    
    with open(SAMPLES_DIR / "README.md", "w") as f:
        f.write(sample_readme)
    
    # End User Tests README
    end_user_readme = """# End User Tests

This folder contains test data with 9 applications across different risk levels for validation testing.

## Files Included

- `application.csv` - Main application data (9 applications)
- `bureau.csv` - Bureau credit data for these applications
- `bureau_balance.csv` - Monthly balance data from bureau
- `previous_application.csv` - Previous credit applications
- `POS_CASH_balance.csv` - POS cash loan balance
- `credit_card_balance.csv` - Credit card balance data
- `installments_payments.csv` - Payment installments data

## Risk Level Distribution

The 9 applications are distributed across risk levels:

| Risk Level | Count | Probability Range | Expected Outcome |
|------------|-------|-------------------|------------------|
| LOW        | 3     | < 30%             | No Default       |
| MEDIUM     | 3     | 30% - 50%         | Mixed            |
| HIGH       | 3     | >= 50%            | Default          |

## Usage

Use this data to test the complete prediction pipeline:

1. Upload all 7 CSV files to Batch Predictions
2. Verify predictions match expected risk levels
3. Check SHAP explanations for each application
4. Download reports and validate outputs
"""
    
    with open(END_USER_TESTS_DIR / "README.md", "w") as f:
        f.write(end_user_readme)
    
    print("\nREADME files created.")


if __name__ == "__main__":
    print("=" * 60)
    print("Creating Sample Templates and End User Test Data")
    print("=" * 60)
    
    create_sample_templates()
    create_end_user_tests()
    create_readme_files()
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
