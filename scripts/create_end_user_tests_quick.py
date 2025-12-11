"""
Quick script to create end user test data with 3 LOW, 3 MEDIUM, 3 HIGH risk applications.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
END_USER_TESTS_DIR = DATA_DIR / 'end_user_tests'
RESULTS_DIR = PROJECT_ROOT / 'results'

def main():
    print('=' * 50)
    print('Creating End User Test Data')
    print('=' * 50)
    
    print('\nLoading train_predictions.csv...')
    predictions = pd.read_csv(RESULTS_DIR / 'train_predictions.csv')
    print(f'  Loaded {len(predictions)} predictions')

    print('\nLoading application_train.csv...')
    app_train = pd.read_csv(DATA_DIR / 'application_train.csv')
    print(f'  Loaded {len(app_train)} applications')

    # Add probabilities by index (both files are in same order)
    app_train['PROBABILITY'] = predictions['PROBABILITY'].values[:len(app_train)]

    # Select samples based on risk levels: 3 LOW, 3 MEDIUM, 4 HIGH = 10 total
    print('\nSelecting risk-stratified samples...')
    low_risk = app_train[app_train['PROBABILITY'] < 0.3].sample(n=3, random_state=456)
    medium_risk = app_train[(app_train['PROBABILITY'] >= 0.3) & (app_train['PROBABILITY'] < 0.5)].sample(n=3, random_state=456)
    high_risk = app_train[app_train['PROBABILITY'] >= 0.5].sample(n=4, random_state=456)

    samples = pd.concat([low_risk, medium_risk, high_risk])

    print(f'\n  Selected {len(samples)} samples:')
    print(f'    LOW risk (< 30%): {len(low_risk)} applications')
    for _, row in low_risk.iterrows():
        print(f'      ID {row["SK_ID_CURR"]}: {row["PROBABILITY"]:.1%} probability')
    
    print(f'    MEDIUM risk (30-50%): {len(medium_risk)} applications')
    for _, row in medium_risk.iterrows():
        print(f'      ID {row["SK_ID_CURR"]}: {row["PROBABILITY"]:.1%} probability')
    
    print(f'    HIGH risk (>= 50%): {len(high_risk)} applications')
    for _, row in high_risk.iterrows():
        print(f'      ID {row["SK_ID_CURR"]}: {row["PROBABILITY"]:.1%} probability')

    sample_ids = samples['SK_ID_CURR'].tolist()

    # Create end user tests directory
    END_USER_TESTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save application data (without PROBABILITY column)
    samples.drop(columns=['PROBABILITY']).to_csv(END_USER_TESTS_DIR / 'application.csv', index=False)
    print(f'\n  Created application.csv')

    # Process linked data files
    print('\nProcessing linked data files...')
    
    # Bureau
    bureau = pd.read_csv(DATA_DIR / 'bureau.csv')
    bureau_filtered = bureau[bureau['SK_ID_CURR'].isin(sample_ids)]
    bureau_filtered.to_csv(END_USER_TESTS_DIR / 'bureau.csv', index=False)
    print(f'  Created bureau.csv with {len(bureau_filtered)} records')
    
    # Bureau Balance
    bureau_ids = bureau_filtered['SK_ID_BUREAU'].tolist()
    bureau_balance = pd.read_csv(DATA_DIR / 'bureau_balance.csv')
    bureau_balance_filtered = bureau_balance[bureau_balance['SK_ID_BUREAU'].isin(bureau_ids)]
    bureau_balance_filtered.to_csv(END_USER_TESTS_DIR / 'bureau_balance.csv', index=False)
    print(f'  Created bureau_balance.csv with {len(bureau_balance_filtered)} records')
    
    # Previous Application
    prev_app = pd.read_csv(DATA_DIR / 'previous_application.csv')
    prev_app_filtered = prev_app[prev_app['SK_ID_CURR'].isin(sample_ids)]
    prev_app_filtered.to_csv(END_USER_TESTS_DIR / 'previous_application.csv', index=False)
    print(f'  Created previous_application.csv with {len(prev_app_filtered)} records')
    
    prev_ids = prev_app_filtered['SK_ID_PREV'].tolist()
    
    # POS Cash Balance
    pos_cash = pd.read_csv(DATA_DIR / 'POS_CASH_balance.csv')
    pos_cash_filtered = pos_cash[pos_cash['SK_ID_PREV'].isin(prev_ids)]
    pos_cash_filtered.to_csv(END_USER_TESTS_DIR / 'POS_CASH_balance.csv', index=False)
    print(f'  Created POS_CASH_balance.csv with {len(pos_cash_filtered)} records')
    
    # Credit Card Balance
    cc_balance = pd.read_csv(DATA_DIR / 'credit_card_balance.csv')
    cc_balance_filtered = cc_balance[cc_balance['SK_ID_PREV'].isin(prev_ids)]
    cc_balance_filtered.to_csv(END_USER_TESTS_DIR / 'credit_card_balance.csv', index=False)
    print(f'  Created credit_card_balance.csv with {len(cc_balance_filtered)} records')
    
    # Installments Payments
    installments = pd.read_csv(DATA_DIR / 'installments_payments.csv')
    installments_filtered = installments[installments['SK_ID_PREV'].isin(prev_ids)]
    installments_filtered.to_csv(END_USER_TESTS_DIR / 'installments_payments.csv', index=False)
    print(f'  Created installments_payments.csv with {len(installments_filtered)} records')
    
    # Create README
    readme_content = """# End User Tests

This folder contains test data with 10 applications representing different risk levels.

## Risk Distribution

| Risk Level | Count | Probability Range | Expected Outcome |
|------------|-------|-------------------|------------------|
| LOW        | 3     | < 30%             | Likely no default |
| MEDIUM     | 3     | 30% - 50%         | Mixed outcomes    |
| HIGH       | 4     | >= 50%            | Likely default    |

## Files

- `application.csv` - Main application data (10 applications)
- `bureau.csv` - Bureau records
- `bureau_balance.csv` - Bureau balance history
- `previous_application.csv` - Previous loan applications
- `POS_CASH_balance.csv` - POS and cash balance
- `credit_card_balance.csv` - Credit card balance
- `installments_payments.csv` - Installment payments

## Usage

Upload all CSV files to the Batch Predictions page to test the model.
"""
    with open(END_USER_TESTS_DIR / 'README.md', 'w') as f:
        f.write(readme_content)
    print('  Created README.md')
    
    # Create ZIP file
    print('\nCreating ZIP archive...')
    shutil.make_archive(
        str(DATA_DIR / 'samples' / 'end_user_tests'),
        'zip',
        END_USER_TESTS_DIR
    )
    print(f'  Created end_user_tests.zip')
    
    print('\n' + '=' * 50)
    print('End User Tests created successfully!')
    print('=' * 50)


if __name__ == '__main__':
    main()
