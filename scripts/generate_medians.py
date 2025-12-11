
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys

# Add project root to path
sys.path.insert(0, ".")

from api.preprocessing_pipeline import PreprocessingPipeline

class PartialPipeline(PreprocessingPipeline):
    def process_up_to_imputation(self, dataframes):
        # Copy-paste logic from process() but stop before impute
        application_df = dataframes.get('application.csv')
        bureau_df = dataframes.get('bureau.csv')
        bureau_balance_df = dataframes.get('bureau_balance.csv')
        previous_application_df = dataframes.get('previous_application.csv')
        pos_cash_df = dataframes.get('POS_CASH_balance.csv')
        credit_card_df = dataframes.get('credit_card_balance.csv')
        installments_df = dataframes.get('installments_payments.csv')

        df = self.aggregate_data(
            application_df, bureau_df, bureau_balance_df,
            previous_application_df, pos_cash_df, credit_card_df, installments_df
        )
        df = self.create_engineered_features(df)
        df = self.encode_and_clean(df)
        return df

def generate_medians():
    print("Generating global medians for imputation...")
    data_dir = Path("data")
    
    # Load subset of training data (enough to get stable medians)
    nrows = 10000 
    print(f"Loading {nrows} rows from application_train.csv...")
    
    app_df = pd.read_csv(data_dir / "application_train.csv", nrows=nrows)
    target_ids = app_df['SK_ID_CURR'].tolist()
    
    dataframes = {'application.csv': app_df}
    
    # Load aux data corresponding to these IDs
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
            print(f"  Scanning {fname}...")
            # Efficient chunk reading
            chunks = []
            # We assume IDs are somewhat clustered or we just scan. 
            # scanning 10GB file for 10k IDs is slow.
            # But we need accuracy.
            # Optimization: The sample is from the *start* of the file. 
            # The aux files are not sorted by ID usually.
            # We'll read the first 1M rows of aux files, hoping for overlap. 
            # Or just read all.
            # Let's read first 500k rows. It should provide enough coverage for medians.
            try:
                aux_df = pd.read_csv(fpath, nrows=500000)
                filtered = aux_df[aux_df[key].isin(target_ids)]
                dataframes[fname] = filtered
            except:
                dataframes[fname] = pd.read_csv(fpath, nrows=0)
    
    if 'bureau.csv' in dataframes:
        bureau_ids = dataframes['bureau.csv']['SK_ID_BUREAU'].unique()
        bb_path = data_dir / "bureau_balance.csv"
        if bb_path.exists():
             aux_df = pd.read_csv(bb_path, nrows=1000000)
             dataframes['bureau_balance.csv'] = aux_df[aux_df['SK_ID_BUREAU'].isin(bureau_ids)]

    # Run partial pipeline
    print("Running pipeline up to imputation...")
    pipeline = PartialPipeline(use_precomputed=False)
    df_encoded = pipeline.process_up_to_imputation(dataframes)
    
    # Calculate Medians
    print("Calculating medians...")
    medians = df_encoded.median(numeric_only=True).to_dict()
    
    # Save
    out_path = Path("data/processed/medians.json")
    with open(out_path, 'w') as f:
        json.dump(medians, f, indent=2)
        
    print(f"Saved {len(medians)} medians to {out_path}")

if __name__ == "__main__":
    generate_medians()
