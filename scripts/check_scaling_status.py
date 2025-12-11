
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, ".")
from api.preprocessing_pipeline import PreprocessingPipeline

def check_scaling():
    print("Checking scaling status of BUREAU_AMT_CREDIT_SUM_SUM...")
    
    # 1. Load Parquet
    pq_path = Path("data/processed/precomputed_features.parquet")
    df_pq = pd.read_parquet(pq_path)
    
    # Get IDs
    if 'SK_ID_CURR' not in df_pq.columns:
        ids = pd.read_csv("data/processed/train_ids.csv")
        df_pq['SK_ID_CURR'] = ids['SK_ID_CURR']
        
    # Pick 5 IDs
    sample_ids = df_pq['SK_ID_CURR'].head(5).tolist()
    
    # 2. Get Raw
    # Run pipeline for these 5 IDs
    dataframes = {}
    
    # Load raw data
    app = pd.read_csv("data/application_train.csv")
    dataframes['application.csv'] = app[app['SK_ID_CURR'].isin(sample_ids)]
    
    # Load aux
    aux_files = {
        'bureau.csv': 'SK_ID_CURR',
        'previous_application.csv': 'SK_ID_CURR',
        'POS_CASH_balance.csv': 'SK_ID_CURR', 
        'credit_card_balance.csv': 'SK_ID_CURR',
        'installments_payments.csv': 'SK_ID_CURR'
    }
    
    for fname, key in aux_files.items():
        fpath = Path("data") / fname
        if fpath.exists():
            df = pd.read_csv(fpath)
            dataframes[fname] = df[df[key].isin(sample_ids)]
            
    # Bureau Balance
    if 'bureau.csv' in dataframes:
        bids = dataframes['bureau.csv']['SK_ID_BUREAU'].unique()
        bb = pd.read_csv("data/bureau_balance.csv")
        dataframes['bureau_balance.csv'] = bb[bb['SK_ID_BUREAU'].isin(bids)]
        
    # Run Pipeline (Live)
    # Temporarily disable the scaler in pipeline by setting scaler=None?
    # Or just use PreprocessingPipeline(use_precomputed=False)
    # But wait, I just updated PreprocessingPipeline to ALWAYS scale if scaler exists!
    # I need to verify the RAW (unscaled) values.
    # I can manually disable scaler in the object.
    
    pipeline = PreprocessingPipeline(use_precomputed=False)
    pipeline.scaler = None # Disable scaling
    
    df_raw, _ = pipeline.process(dataframes, keep_sk_id=True)
    
    # Compare
    feat = "BUREAU_AMT_CREDIT_SUM_SUM"
    print(f"\nFeature: {feat}")
    
    for _, row in df_raw.iterrows():
        app_id = row['SK_ID_CURR']
        raw_val = row[feat]
        
        # Get parquet val
        pq_val = df_pq[df_pq['SK_ID_CURR'] == app_id][feat].values[0]
        
        print(f"ID {app_id}: Raw={raw_val:.4f}, Parquet={pq_val:.4f}")
        if abs(raw_val - pq_val) < 1.0:
            print("  -> MATCHES! (Unscaled)")
        else:
            print("  -> SCALED")

if __name__ == "__main__":
    check_scaling()
