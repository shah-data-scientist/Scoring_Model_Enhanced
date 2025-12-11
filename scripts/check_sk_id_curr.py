
import pandas as pd
from pathlib import Path

def check_sk_id_curr_in_files():
    base_dir = Path(".")
    submission_path = base_dir / "results" / "submission.csv"
    data_dirs = [base_dir / "data", base_dir / "data/samples", base_dir / "data/processed"]
    
    # 1. Get Sample SK_ID_CURR from submission.csv
    print(f"Reading submission file from {submission_path}...")
    try:
        df_sub = pd.read_csv(submission_path)
    except FileNotFoundError:
        print(f"Error: Submission file not found at {submission_path}")
        return

    # Sort by TARGET to select a diverse set of IDs (same logic as before)
    df_sub_sorted = df_sub.sort_values(by="TARGET", ascending=False)
    
    high_risk = df_sub_sorted.head(3)
    low_risk = df_sub_sorted.tail(4)
    mid_idx = len(df_sub_sorted) // 2
    medium_risk = df_sub_sorted.iloc[mid_idx-1 : mid_idx+2]
    
    selected_df = pd.concat([high_risk, medium_risk, low_risk])
    selected_ids = set(selected_df['SK_ID_CURR'].tolist())
    
    print(f"\nChecking for {len(selected_ids)} selected SK_ID_CURR values: {selected_ids}")
    
    # 2. Iterate through data files
    for data_dir in data_dirs:
        if not data_dir.exists():
            continue
            
        print(f"\nScanning directory: {data_dir}")
        for csv_file in data_dir.glob("*.csv"):
            if "submission" in csv_file.name or "end_user_tests" in str(csv_file):
                continue
                
            try:
                # Read just the header first to check for SK_ID_CURR
                header = pd.read_csv(csv_file, nrows=0)
                if 'SK_ID_CURR' in header.columns:
                    # Read the file (optimized: only SK_ID_CURR column)
                    # For very large files, this might still be slow, but better than reading all cols
                    try:
                        df_ids = pd.read_csv(csv_file, usecols=['SK_ID_CURR'])
                        present_ids = set(df_ids['SK_ID_CURR'])
                        
                        found_count = len(selected_ids.intersection(present_ids))
                        
                        print(f"  [MATCH] {csv_file.name}: Found {found_count}/{len(selected_ids)} of the selected IDs.")
                        
                        if found_count > 0 and found_count < len(selected_ids):
                             missing = selected_ids - present_ids
                             print(f"      Missing IDs in this file: {missing}")

                    except Exception as e:
                         print(f"  [ERROR] Could not read SK_ID_CURR from {csv_file.name}: {e}")
                
                elif 'SK_ID_BUREAU' in header.columns and 'SK_ID_CURR' not in header.columns:
                     print(f"  [INFO]  {csv_file.name}: Has SK_ID_BUREAU but no SK_ID_CURR (likely linked via bureau.csv)")
                else:
                    print(f"  [SKIP]  {csv_file.name}: No SK_ID_CURR column.")

            except Exception as e:
                print(f"  [ERROR] processing {csv_file.name}: {e}")

if __name__ == "__main__":
    check_sk_id_curr_in_files()
