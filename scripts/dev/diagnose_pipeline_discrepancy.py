
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, ".")

from api.preprocessing_pipeline import PreprocessingPipeline


def diagnose_discrepancy():
    print("Starting pipeline discrepancy diagnosis...")

    # 1. Get an original ID
    mapping_path = Path("data/end_user_tests/id_mapping.csv")
    if not mapping_path.exists():
        print("Error: id_mapping.csv not found.")
        return

    mapping_df = pd.read_csv(mapping_path)
    # Pick the first one
    original_id = mapping_df.iloc[0]['original_sk_id_curr']
    print(f"Testing with Original ID: {original_id}")

    # 2. Extract Data for this ID from ORIGINAL sources
    # We only need to find the data for this specific ID.
    data_dir = Path("data")

    dataframes = {}

    # Application
    app_df = pd.read_csv(data_dir / "application_test.csv")
    dataframes['application.csv'] = app_df[app_df['SK_ID_CURR'] == original_id]

    if dataframes['application.csv'].empty:
         print(f"ID {original_id} not found in application_test.csv. Checking train...")
         app_train = pd.read_csv(data_dir / "application_train.csv")
         dataframes['application.csv'] = app_train[app_train['SK_ID_CURR'] == original_id]

    if dataframes['application.csv'].empty:
        print("ID not found in train or test.")
        return

    # Auxiliary tables
    aux_files = {
        'bureau.csv': 'SK_ID_CURR',
        'previous_application.csv': 'SK_ID_CURR',
        'POS_CASH_balance.csv': 'SK_ID_CURR',
        'credit_card_balance.csv': 'SK_ID_CURR',
        'installments_payments.csv': 'SK_ID_CURR'
    }

    # Helper to read and filter
    for fname, key in aux_files.items():
        fpath = data_dir / fname
        if fpath.exists():
            # efficient read if possible, but for diagnosis reading all is okay-ish if file isn't massive
            # actually files are massive. Let's use chunks or just hope it fits.
            # Better: use the pre-filtered files? No, those are anonymized.
            # I'll use chunks.
            print(f"  Scanning {fname}...")
            chunks = []
            found = False
            for chunk in pd.read_csv(fpath, chunksize=50000):
                filtered = chunk[chunk[key] == original_id]
                if not filtered.empty:
                    chunks.append(filtered)
                    found = True

            if found:
                dataframes[fname] = pd.concat(chunks)
            else:
                 # Empty DF with cols
                 dataframes[fname] = pd.read_csv(fpath, nrows=0)
        else:
            print(f"  Warning: {fname} not found")

    # Bureau Balance
    if 'bureau.csv' in dataframes and not dataframes['bureau.csv'].empty:
        bureau_ids = dataframes['bureau.csv']['SK_ID_BUREAU'].unique()
        bb_path = data_dir / "bureau_balance.csv"
        if bb_path.exists():
            print(f"  Scanning bureau_balance.csv for {len(bureau_ids)} bureau IDs...")
            chunks = []
            for chunk in pd.read_csv(bb_path, chunksize=50000):
                 filtered = chunk[chunk['SK_ID_BUREAU'].isin(bureau_ids)]
                 if not filtered.empty:
                     chunks.append(filtered)
            if chunks:
                dataframes['bureau_balance.csv'] = pd.concat(chunks)

    # 3. Run Pipeline A: Precomputed = True
    print("\nRunning Pipeline A (Precomputed Features)...")
    pipeline_a = PreprocessingPipeline(use_precomputed=True)
    features_a, _ = pipeline_a.process(dataframes, keep_sk_id=True)

    # 4. Run Pipeline B: Precomputed = False (Force Live Calculation)
    print("\nRunning Pipeline B (Live Calculation)...")
    pipeline_b = PreprocessingPipeline(use_precomputed=False)
    features_b, _ = pipeline_b.process(dataframes, keep_sk_id=True)

    # 5. Compare
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)

    cols_a = set(features_a.columns)
    cols_b = set(features_b.columns)

    if cols_a != cols_b:
        print("Column mismatch!")
        print(f"In A not B: {cols_a - cols_b}")
        print(f"In B not A: {cols_b - cols_a}")

    common_cols = list(cols_a.intersection(cols_b))
    common_cols.remove('SK_ID_CURR')

    diff_summary = []

    for col in common_cols:
        val_a = features_a.iloc[0][col]
        val_b = features_b.iloc[0][col]

        # Handle NaNs
        if pd.isna(val_a) and pd.isna(val_b):
            continue

        # Numerical comparison
        try:
            diff = abs(val_a - val_b)
            if diff > 1e-5: # Tolerance
                diff_summary.append({
                    'feature': col,
                    'precomputed': val_a,
                    'live': val_b,
                    'diff': diff
                })
        except:
            if val_a != val_b:
                diff_summary.append({
                    'feature': col,
                    'precomputed': val_a,
                    'live': val_b,
                    'diff': 'N/A'
                })

    if diff_summary:
        print(f"\nFound differences in {len(diff_summary)} features!")
        diff_df = pd.DataFrame(diff_summary).sort_values(by='diff', ascending=False)
        print(diff_df.head(20))

        # Save full diff
        diff_df.to_csv("pipeline_discrepancy.csv", index=False)
        print("\nFull discrepancy report saved to pipeline_discrepancy.csv")
    else:
        print("\nNO DIFFERENCES FOUND! The pipelines are consistent.")
        print("If results differ, it might be due to the ID change itself affecting sort order or similar.")

if __name__ == "__main__":
    diagnose_discrepancy()
