"""Create sample CSV files for API testing.

Extracts 20 applications from the training data along with all related
records from auxiliary tables.
"""

import sys
from pathlib import Path

import pandas as pd

# Setup path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DIR = PROJECT_ROOT / "data" / "samples"

# Create samples directory
SAMPLE_DIR.mkdir(exist_ok=True)

def create_sample_files(n_samples=20):
    """Create sample CSV files for testing.

    Args:
        n_samples: Number of applications to sample

    """
    print(f"\n{'='*80}")
    print(f"CREATING SAMPLE DATA FOR API TESTING ({n_samples} applications)")
    print(f"{'='*80}\n")

    # 1. Load and sample application data
    print("Step 1: Loading application data...")
    app_train = pd.read_csv(DATA_DIR / "application_train.csv")

    # Sample N applications
    app_sample = app_train.sample(n=n_samples, random_state=42)
    sampled_ids = set(app_sample['SK_ID_CURR'])

    # Rename to application.csv
    app_sample_out = app_sample.copy()

    # Save
    out_path = SAMPLE_DIR / "application.csv"
    app_sample_out.to_csv(out_path, index=False)
    print(f"  Saved application.csv: {len(app_sample_out)} rows, {len(app_sample_out.columns)} columns")

    # 2. Bureau data
    print("\nStep 2: Extracting bureau data...")
    bureau = pd.read_csv(DATA_DIR / "bureau.csv")
    bureau_sample = bureau[bureau['SK_ID_CURR'].isin(sampled_ids)]
    bureau_sample.to_csv(SAMPLE_DIR / "bureau.csv", index=False)
    print(f"  Saved bureau.csv: {len(bureau_sample)} rows")

    # Bureau balance (linked to bureau via SK_ID_BUREAU)
    if len(bureau_sample) > 0:
        bureau_ids = set(bureau_sample['SK_ID_BUREAU'])
        bureau_balance = pd.read_csv(DATA_DIR / "bureau_balance.csv")
        bureau_balance_sample = bureau_balance[bureau_balance['SK_ID_BUREAU'].isin(bureau_ids)]
        bureau_balance_sample.to_csv(SAMPLE_DIR / "bureau_balance.csv", index=False)
        print(f"  Saved bureau_balance.csv: {len(bureau_balance_sample)} rows")
    else:
        # Create empty file
        pd.DataFrame(columns=['SK_ID_BUREAU', 'MONTHS_BALANCE', 'STATUS']).to_csv(
            SAMPLE_DIR / "bureau_balance.csv", index=False
        )
        print("  Saved bureau_balance.csv: 0 rows (empty)")

    # 3. Previous application
    print("\nStep 3: Extracting previous application data...")
    prev_app = pd.read_csv(DATA_DIR / "previous_application.csv")
    prev_app_sample = prev_app[prev_app['SK_ID_CURR'].isin(sampled_ids)]
    prev_app_sample.to_csv(SAMPLE_DIR / "previous_application.csv", index=False)
    print(f"  Saved previous_application.csv: {len(prev_app_sample)} rows")

    # Get SK_ID_PREV for auxiliary tables
    if len(prev_app_sample) > 0:
        prev_ids = set(prev_app_sample['SK_ID_PREV'])
    else:
        prev_ids = set()

    # 4. Credit card balance
    print("\nStep 4: Extracting credit card balance data...")
    if (DATA_DIR / "credit_card_balance.csv").exists():
        cc_balance = pd.read_csv(DATA_DIR / "credit_card_balance.csv")
        cc_balance_sample = cc_balance[cc_balance['SK_ID_PREV'].isin(prev_ids)] if len(prev_ids) > 0 else pd.DataFrame(columns=cc_balance.columns)
        cc_balance_sample.to_csv(SAMPLE_DIR / "credit_card_balance.csv", index=False)
        print(f"  Saved credit_card_balance.csv: {len(cc_balance_sample)} rows")
    else:
        # Create empty with expected columns
        pd.DataFrame(columns=['SK_ID_PREV', 'SK_ID_CURR', 'MONTHS_BALANCE']).to_csv(
            SAMPLE_DIR / "credit_card_balance.csv", index=False
        )
        print("  Saved credit_card_balance.csv: 0 rows (empty)")

    # 5. Installments payments
    print("\nStep 5: Extracting installments payments data...")
    if (DATA_DIR / "installments_payments.csv").exists():
        installments = pd.read_csv(DATA_DIR / "installments_payments.csv")
        installments_sample = installments[installments['SK_ID_PREV'].isin(prev_ids)] if len(prev_ids) > 0 else pd.DataFrame(columns=installments.columns)
        installments_sample.to_csv(SAMPLE_DIR / "installments_payments.csv", index=False)
        print(f"  Saved installments_payments.csv: {len(installments_sample)} rows")
    else:
        pd.DataFrame(columns=['SK_ID_PREV', 'SK_ID_CURR', 'NUM_INSTALMENT_VERSION']).to_csv(
            SAMPLE_DIR / "installments_payments.csv", index=False
        )
        print("  Saved installments_payments.csv: 0 rows (empty)")

    # 6. POS cash balance
    print("\nStep 6: Extracting POS cash balance data...")
    if (DATA_DIR / "POS_CASH_balance.csv").exists():
        pos_cash = pd.read_csv(DATA_DIR / "POS_CASH_balance.csv")
        pos_cash_sample = pos_cash[pos_cash['SK_ID_PREV'].isin(prev_ids)] if len(prev_ids) > 0 else pd.DataFrame(columns=pos_cash.columns)
        pos_cash_sample.to_csv(SAMPLE_DIR / "POS_CASH_balance.csv", index=False)
        print(f"  Saved POS_CASH_balance.csv: {len(pos_cash_sample)} rows")
    else:
        pd.DataFrame(columns=['SK_ID_PREV', 'SK_ID_CURR', 'MONTHS_BALANCE']).to_csv(
            SAMPLE_DIR / "POS_CASH_balance.csv", index=False
        )
        print("  Saved POS_CASH_balance.csv: 0 rows (empty)")

    print(f"\n{'='*80}")
    print("SAMPLE DATA CREATION COMPLETE")
    print(f"{'='*80}\n")
    print(f"Sample files saved to: {SAMPLE_DIR}")
    print("\nSummary:")
    print(f"  - {n_samples} applications sampled")
    print(f"  - SK_ID_CURR range: {app_sample['SK_ID_CURR'].min()} to {app_sample['SK_ID_CURR'].max()}")
    print("  - All 7 CSV files created")
    print("\nReady for API testing!")

if __name__ == "__main__":
    create_sample_files(n_samples=20)
