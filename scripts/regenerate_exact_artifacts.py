import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
from sklearn.preprocessing import StandardScaler
import sys
import gc
import math

# Add project root to path
sys.path.insert(0, ".")

from api.preprocessing_pipeline import PreprocessingPipeline

class RawFeaturePipeline(PreprocessingPipeline):
    """Pipeline that stops before imputation to allow global stat calculation."""
    def process_to_raw(self, dataframes):
        application_df = dataframes.get('application.csv')
        bureau_df = dataframes.get('bureau.csv')
        bureau_balance_df = dataframes.get('bureau_balance.csv')
        previous_application_df = dataframes.get('previous_application.csv')
        pos_cash_df = dataframes.get('POS_CASH_balance.csv')
        credit_card_df = dataframes.get('credit_card_balance.csv')
        installments_df = dataframes.get('installments_payments.csv')

        print("    Aggregating...")
        df = self.aggregate_data(
            application_df, bureau_df, bureau_balance_df,
            previous_application_df, pos_cash_df, credit_card_df, installments_df
        )
        print("    Domain features...")
        df = self.create_engineered_features(df)
        print("    Encoding...")
        df = self.encode_and_clean(df)
        print("    Aligning...")
        df = self.align_features(df)
        return df

def regenerate_exact_artifacts():
    print("="*80)
    print("REGENERATING EXACT ARTIFACTS (FULL DATASET - CHUNKED)")
    print("="*80)
    
    data_dir = Path("data")
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(exist_ok=True)
    
    # 1. Get All IDs
    print("\n[1/5] Loading IDs from application_train.csv...")
    try:
        app_train_ids = pd.read_csv(data_dir / "application_train.csv", usecols=['SK_ID_CURR'])['SK_ID_CURR'].unique()
        total_ids = len(app_train_ids)
        print(f"  Total IDs: {total_ids}")
        
        # Split into chunks
        CHUNK_SIZE = 50000
        num_chunks = math.ceil(total_ids / CHUNK_SIZE)
        id_chunks = np.array_split(app_train_ids, num_chunks)
        
        print(f"  Split into {num_chunks} chunks of ~{CHUNK_SIZE} IDs")

    except Exception as e:
        print(f"Error loading IDs: {e}")
        return

    # 2. Process Chunks
    print("\n[2/5] Processing chunks...")
    
    processed_dfs = []
    
    for i, chunk_ids in enumerate(id_chunks):
        print(f"\n  Processing Chunk {i+1}/{num_chunks} ({len(chunk_ids)} IDs)...")
        chunk_ids_set = set(chunk_ids)
        
        # Load Data for this chunk
        dataframes = {}
        
        # Application
        try:
            # Efficient reading is hard without scanning, but 50k IDs is small enough to just read and filter if files fit in memory?
            # No, reading full aux files repeatedly is slow.
            # Ideally we'd iterate through files.
            # But let's just try reading and filtering.
            
            # Application (small enough)
            app = pd.read_csv(data_dir / "application_train.csv")
            dataframes['application.csv'] = app[app['SK_ID_CURR'].isin(chunk_ids_set)]
            del app
            
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
                    print(f"    Loading {fname}...")
                    # Read in chunks to filter
                    file_chunks = []
                    for file_chunk in pd.read_csv(fpath, chunksize=500000):
                        filtered = file_chunk[file_chunk[key].isin(chunk_ids_set)]
                        if not filtered.empty:
                            file_chunks.append(filtered)
                    if file_chunks:
                        dataframes[fname] = pd.concat(file_chunks)
                    else:
                        # Empty with cols
                        dataframes[fname] = pd.read_csv(fpath, nrows=0)
            
            # Bureau Balance
            if 'bureau.csv' in dataframes and (data_dir / 'bureau_balance.csv').exists():
                 print("    Loading bureau_balance.csv...")
                 bids = set(dataframes['bureau.csv']['SK_ID_BUREAU'])
                 file_chunks = []
                 for file_chunk in pd.read_csv(data_dir / 'bureau_balance.csv', chunksize=500000):
                     filtered = file_chunk[file_chunk['SK_ID_BUREAU'].isin(bids)]
                     if not filtered.empty:
                         file_chunks.append(filtered)
                 if file_chunks:
                     dataframes['bureau_balance.csv'] = pd.concat(file_chunks)

            # Run Pipeline
            pipeline = RawFeaturePipeline(use_precomputed=False)
            pipeline.scaler = None
            pipeline.medians = None
            
            df_chunk = pipeline.process_to_raw(dataframes)
            
            # Save chunk to disk to save memory
            chunk_path = processed_dir / f"raw_chunk_{i}.parquet"
            if 'SK_ID_CURR' in df_chunk.columns:
                df_chunk = df_chunk.drop(columns=['SK_ID_CURR'])
            
            df_chunk.to_parquet(chunk_path)
            processed_dfs.append(chunk_path)
            
            print(f"    Saved chunk {i} to {chunk_path}")
            
            del dataframes
            del df_chunk
            gc.collect()
            
        except Exception as e:
            print(f"Error processing chunk {i}: {e}")
            return

    # 3. Combine and Calculate Medians
    print("\n[3/5] Combining chunks and calculating medians...")
    try:
        # Load all chunks
        full_df = pd.concat([pd.read_parquet(p) for p in processed_dfs], ignore_index=True)
        print(f"  Full Matrix Shape: {full_df.shape}")
        
        medians = full_df.median(numeric_only=True).to_dict()
        
        medians_path = processed_dir / "medians.json"
        with open(medians_path, 'w') as f:
            json.dump(medians, f, indent=2)
        print(f"  Saved {len(medians)} medians to {medians_path}")
        
    except Exception as e:
        print(f"Error calculating medians: {e}")
        return

    # 4. Impute & Calculate Exact Scaler
    print("\n[4/5] Calculating exact scaler...")
    
    # Impute
    print("  Imputing missing values...")
    for col, median_val in medians.items():
        if col in full_df.columns:
            full_df[col] = full_df[col].fillna(median_val)
            
    full_df = full_df.fillna(0)
    
    print("  Fitting StandardScaler...")
    scaler = StandardScaler()
    scaler.fit(full_df)
    
    scaler_path = processed_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"  Saved scaler to {scaler_path}")
    
    # Cleanup chunks
    print("\n[5/5] Cleaning up...")
    for p in processed_dfs:
        if p.exists():
            p.unlink()
    
    print("\nDONE! Exact artifacts regenerated from full dataset.")

if __name__ == "__main__":
    regenerate_exact_artifacts()