"""Create synthetic test data derived from end_user_tests with new IDs."""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
SOURCE_DIR = PROJECT_ROOT / "data" / "end_user_tests"
OUTPUT_DIR = PROJECT_ROOT / "data" / "synthetic_tests"

print("="*80)
print("CREATING SYNTHETIC TEST DATA")
print("="*80)

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Load original end_user_tests data
print(f"\nLoading source data from: {SOURCE_DIR}")
app = pd.read_csv(SOURCE_DIR / 'application.csv')
bureau = pd.read_csv(SOURCE_DIR / 'bureau.csv')
bureau_balance = pd.read_csv(SOURCE_DIR / 'bureau_balance.csv')
prev_app = pd.read_csv(SOURCE_DIR / 'previous_application.csv')
cc = pd.read_csv(SOURCE_DIR / 'credit_card_balance.csv')
installments = pd.read_csv(SOURCE_DIR / 'installments_payments.csv')
pos_cash = pd.read_csv(SOURCE_DIR / 'POS_CASH_balance.csv')

print(f"  Application: {len(app)} rows")
print(f"  Bureau: {len(bureau)} rows")
print(f"  Bureau Balance: {len(bureau_balance)} rows")
print(f"  Previous App: {len(prev_app)} rows")
print(f"  Credit Card: {len(cc)} rows")
print(f"  Installments: {len(installments)} rows")
print(f"  POS Cash: {len(pos_cash)} rows")

# Get original IDs
original_ids = app['SK_ID_CURR'].tolist()
print(f"\nOriginal IDs: {original_ids[:5]}...")

# Create synthetic IDs that don't exist in any dataset
# Use a high starting number to avoid collisions
synthetic_base_id = 9000000
synthetic_ids = [synthetic_base_id + i for i in range(len(app))]

print(f"Synthetic IDs: {synthetic_ids[:5]}...")

# Create ID mapping
id_mapping = dict(zip(original_ids, synthetic_ids))

print("\n[1/7] Creating synthetic application.csv...")
app_synthetic = app.copy()
app_synthetic['SK_ID_CURR'] = app_synthetic['SK_ID_CURR'].map(id_mapping)
app_synthetic.to_csv(OUTPUT_DIR / 'application.csv', index=False)
print(f"  Saved {len(app_synthetic)} applications with IDs: {synthetic_ids}")

print("\n[2/7] Creating synthetic bureau.csv...")
if len(bureau) > 0:
    bureau_synthetic = bureau.copy()
    bureau_synthetic['SK_ID_CURR'] = bureau_synthetic['SK_ID_CURR'].map(id_mapping)

    # Also need to update bureau IDs to avoid conflicts
    bureau_id_mapping = {old: 9000000 + idx for idx, old in enumerate(bureau['SK_ID_BUREAU'].unique())}
    bureau_synthetic['SK_ID_BUREAU'] = bureau_synthetic['SK_ID_BUREAU'].map(bureau_id_mapping)
    bureau_synthetic.to_csv(OUTPUT_DIR / 'bureau.csv', index=False)
    print(f"  Saved {len(bureau_synthetic)} bureau records")
else:
    pd.DataFrame().to_csv(OUTPUT_DIR / 'bureau.csv', index=False)
    print("  Saved empty file")

print("\n[3/7] Creating synthetic bureau_balance.csv...")
if len(bureau_balance) > 0 and len(bureau) > 0:
    bureau_balance_synthetic = bureau_balance.copy()
    bureau_balance_synthetic['SK_ID_BUREAU'] = bureau_balance_synthetic['SK_ID_BUREAU'].map(bureau_id_mapping)
    bureau_balance_synthetic.to_csv(OUTPUT_DIR / 'bureau_balance.csv', index=False)
    print(f"  Saved {len(bureau_balance_synthetic)} bureau_balance records")
else:
    pd.DataFrame().to_csv(OUTPUT_DIR / 'bureau_balance.csv', index=False)
    print("  Saved empty file")

print("\n[4/7] Creating synthetic previous_application.csv...")
if len(prev_app) > 0:
    prev_app_synthetic = prev_app.copy()
    prev_app_synthetic['SK_ID_CURR'] = prev_app_synthetic['SK_ID_CURR'].map(id_mapping)

    # Update previous application IDs
    prev_id_mapping = {old: 9000000 + idx for idx, old in enumerate(prev_app['SK_ID_PREV'].unique())}
    prev_app_synthetic['SK_ID_PREV'] = prev_app_synthetic['SK_ID_PREV'].map(prev_id_mapping)
    prev_app_synthetic.to_csv(OUTPUT_DIR / 'previous_application.csv', index=False)
    print(f"  Saved {len(prev_app_synthetic)} previous_application records")
else:
    pd.DataFrame().to_csv(OUTPUT_DIR / 'previous_application.csv', index=False)
    print("  Saved empty file")

print("\n[5/7] Creating synthetic credit_card_balance.csv...")
if len(cc) > 0:
    cc_synthetic = cc.copy()
    cc_synthetic['SK_ID_PREV'] = cc_synthetic['SK_ID_PREV'].map(prev_id_mapping)
    cc_synthetic.to_csv(OUTPUT_DIR / 'credit_card_balance.csv', index=False)
    print(f"  Saved {len(cc_synthetic)} credit_card_balance records")
else:
    pd.DataFrame().to_csv(OUTPUT_DIR / 'credit_card_balance.csv', index=False)
    print("  Saved empty file")

print("\n[6/7] Creating synthetic installments_payments.csv...")
if len(installments) > 0:
    installments_synthetic = installments.copy()
    installments_synthetic['SK_ID_PREV'] = installments_synthetic['SK_ID_PREV'].map(prev_id_mapping)
    installments_synthetic.to_csv(OUTPUT_DIR / 'installments_payments.csv', index=False)
    print(f"  Saved {len(installments_synthetic)} installments_payments records")
else:
    pd.DataFrame().to_csv(OUTPUT_DIR / 'installments_payments.csv', index=False)
    print("  Saved empty file")

print("\n[7/7] Creating synthetic POS_CASH_balance.csv...")
if len(pos_cash) > 0:
    pos_cash_synthetic = pos_cash.copy()
    pos_cash_synthetic['SK_ID_PREV'] = pos_cash_synthetic['SK_ID_PREV'].map(prev_id_mapping)
    pos_cash_synthetic.to_csv(OUTPUT_DIR / 'POS_CASH_balance.csv', index=False)
    print(f"  Saved {len(pos_cash_synthetic)} POS_CASH_balance records")
else:
    pd.DataFrame().to_csv(OUTPUT_DIR / 'POS_CASH_balance.csv', index=False)
    print("  Saved empty file")

print("\n" + "="*80)
print("SYNTHETIC DATA SUMMARY")
print("="*80)
print(f"Created {len(synthetic_ids)} synthetic applications")
print(f"  Original IDs: {original_ids[0]} - {original_ids[-1]}")
print(f"  Synthetic IDs: {synthetic_ids[0]} - {synthetic_ids[-1]}")
print(f"\nOutput directory: {OUTPUT_DIR}")
print("\nThese applications:")
print("  - Have the same features/patterns as end_user_tests")
print("  - Have NEW SK_ID_CURR values not in train/val/test")
print("  - Will force the API to use FULL preprocessing pipeline")
print("  - Will test if the pipeline works for truly new applications")
print("="*80)

# Save ID mapping for reference
mapping_df = pd.DataFrame({
    'original_id': original_ids,
    'synthetic_id': synthetic_ids
})
mapping_df.to_csv(OUTPUT_DIR / 'id_mapping.csv', index=False)
print(f"\nID mapping saved to: {OUTPUT_DIR / 'id_mapping.csv'}")
