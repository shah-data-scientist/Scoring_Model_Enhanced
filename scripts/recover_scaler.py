
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.insert(0, ".")

from api.preprocessing_pipeline import PreprocessingPipeline


def recover_scaler():
    print("Recovering scaler from precomputed vs live features...")

    # Paths
    data_dir = Path("data")
    processed_dir = data_dir / "processed"

    # 1. Load X_train (Scaled) subset and IDs
    print("Loading X_train subset...")
    try:
        # Try parquet first
        x_train_path = processed_dir / "precomputed_features.parquet"
        if x_train_path.exists():
            # Read first 200 rows
            # pandas read_parquet doesn't easily support nrows, but we can read and head
            # Parquet is columnar, reading full file might be fast enough if not huge
            # Or use fastparquet/pyarrow to read row group.
            # Let's try reading full index (IDs) if possible, or just read CSV if parquet fails or is too big
            df_scaled_full = pd.read_parquet(x_train_path)

            # We need SK_ID_CURR.
            if 'SK_ID_CURR' not in df_scaled_full.columns:
                # Load IDs
                ids = pd.read_csv(processed_dir / "train_ids.csv")
                df_scaled_full['SK_ID_CURR'] = ids['SK_ID_CURR']

            # Sample 500 IDs that are also in application_train.csv
            # We'll need to check which ones we can easily extract raw data for.
            # Let's just take the first 500 IDs.
            target_ids = df_scaled_full['SK_ID_CURR'].head(500).tolist()
            df_scaled = df_scaled_full[df_scaled_full['SK_ID_CURR'].isin(target_ids)].set_index('SK_ID_CURR')

        else:
             print("Parquet not found, trying CSV...")
             # ... CSV logic (omitted for brevity, assuming parquet exists as per previous logs)
             return

    except Exception as e:
        print(f"Error loading scaled data: {e}")
        return

    print(f"Target IDs: {target_ids}")

    # 2. Extract Raw Data for these IDs
    print("Extracting raw data...")
    dataframes = {}

    # App Train
    app_train = pd.read_csv(data_dir / "application_train.csv")
    dataframes['application.csv'] = app_train[app_train['SK_ID_CURR'].isin(target_ids)]

    # Aux files
    aux_files = {
        'bureau.csv': 'SK_ID_CURR',
        'previous_application.csv': 'SK_ID_CURR',
        'POS_CASH_balance.csv': 'SK_ID_CURR',
        'credit_card_balance.csv': 'SK_ID_CURR',
        'installments_payments.csv': 'SK_ID_CURR'
    }

    for fname, key in aux_files.items():
        fpath = data_dir / fname
        if fpath.exists():
            # Read chunks to find our IDs
            chunks = []
            for chunk in pd.read_csv(fpath, chunksize=50000):
                filtered = chunk[chunk[key].isin(target_ids)]
                if not filtered.empty:
                    chunks.append(filtered)
            if chunks:
                dataframes[fname] = pd.concat(chunks)
            else:
                 dataframes[fname] = pd.read_csv(fpath, nrows=0)

    # Bureau Balance
    if 'bureau.csv' in dataframes and not dataframes['bureau.csv'].empty:
        bureau_ids = dataframes['bureau.csv']['SK_ID_BUREAU'].unique()
        bb_path = data_dir / "bureau_balance.csv"
        if bb_path.exists():
            chunks = []
            for chunk in pd.read_csv(bb_path, chunksize=50000):
                 filtered = chunk[chunk['SK_ID_BUREAU'].isin(bureau_ids)]
                 if not filtered.empty:
                     chunks.append(filtered)
            if chunks:
                dataframes['bureau_balance.csv'] = pd.concat(chunks)

    # 3. Run Pipeline (Live) -> Raw Features
    print("Running Live Pipeline...")
    pipeline = PreprocessingPipeline(use_precomputed=False)
    # CRITICAL: Disable scaler to get actual raw values!
    pipeline.scaler = None
    # Ensure we get features in correct order
    df_raw, _ = pipeline.process(dataframes, keep_sk_id=True)
    df_raw = df_raw.set_index('SK_ID_CURR')

    # 4. Calculate Scaler Stats
    print("Calculating scaler statistics...")

    # Align rows
    common_ids = list(set(df_raw.index) & set(df_scaled.index))
    df_raw = df_raw.loc[common_ids].sort_index()
    df_scaled = df_scaled.loc[common_ids].sort_index()

    print(f"Using {len(common_ids)} common samples.")

    # Lists to store means and scales
    means = []
    scales = []
    features = []

    for col in df_scaled.columns:
        if col not in df_raw.columns:
            print(f"Warning: {col} missing in raw data")
            continue

        S = df_scaled[col].values
        R = df_raw[col].values

        # S = (R - Mean) / Scale
        # Scale * S = R - Mean
        # R = Scale * S + Mean
        # This is a linear regression: Y=R, X=S. Slope=Scale, Intercept=Mean.

        # Simple Linear Regression:
        # Scale = Cov(R, S) / Var(S) ?
        # Or just:
        # Scale = Std(R) / Std(S)  (since Std(S)=1 ideally, so Scale=Std(R))
        # Mean = Mean(R) - Scale * Mean(S) (since Mean(S)=0 ideally, Mean=Mean(R))

        # Let's verify assumption: is S standardized?
        # If S is StandardScaled, mean(S)~0, std(S)~1.
        # Then Mean_feature = mean(R), Scale_feature = std(R).

        # Let's check correlation to ensure they map linearly
        # corr = np.corrcoef(R, S)[0,1]
        # if abs(corr) < 0.99:
        #     print(f"Warning: {col} correlation {corr} - maybe not linear scaling?")

        # We'll simply compute Mean(R) and Std(R) because that's what StandardScaler does!
        # Wait, NO.
        # The scaler was fitted on the WHOLE training set.
        # My 50 samples might have different mean/std than the whole set.
        # But I need to recover the GLOBAL Mean and Std used for scaling.
        # So I must solve for the parameters that map THIS R to THIS S.

        # Y = mX + c  -> R = Scale * S + Mean
        # We can use polyfit(S, R, 1)

        if len(S) > 1 and np.std(S) > 1e-6:
             slope, intercept = np.polyfit(S, R, 1)
             calculated_scale = slope
             calculated_mean = intercept
        else:
             # Constant value or zero variance
             print(f"  Warning: Low variance in {col}, assuming Scale=1.0")
             calculated_mean = np.mean(R)
             calculated_scale = 1.0 # Default

        means.append(calculated_mean)
        scales.append(calculated_scale)
        features.append(col)

    # 5. Create and Save Scaler
    print("Saving recovered scaler...")
    scaler = StandardScaler()
    scaler.mean_ = np.array(means)
    scaler.scale_ = np.array(scales)
    scaler.var_ = scaler.scale_ ** 2  # Approximate var
    scaler.n_features_in_ = len(features)
    scaler.feature_names_in_ = features

    # We also need to set 'with_mean=True', 'with_std=True'

    scaler_path = processed_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    # Verify
    print("Verification on first feature:")
    f0 = features[0]
    r0 = df_raw[f0].values[0]
    s0 = df_scaled[f0].values[0]
    s_pred = (r0 - means[0]) / scales[0]
    print(f"Feature: {f0}")
    print(f"Raw: {r0}, Scaled(Target): {s0}, Scaled(Recovered): {s_pred}")

if __name__ == "__main__":
    recover_scaler()
