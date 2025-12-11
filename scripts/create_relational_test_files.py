
import pandas as pd
from pathlib import Path
import os

def create_relational_test_files():
    # Define paths
    base_dir = Path(".")
    submission_path = base_dir / "results" / "submission.csv"
    data_dir = base_dir / "data"
    output_dir = base_dir / "data" / "end_user_tests"
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Get Selected IDs
    print(f"Reading submission file from {submission_path}...")
    try:
        df_sub = pd.read_csv(submission_path)
    except FileNotFoundError:
        print(f"Error: Submission file not found at {submission_path}")
        return

    # Sort and Select
    df_sub_sorted = df_sub.sort_values(by="TARGET", ascending=False)
    
    high_risk = df_sub_sorted.head(3).copy()
    high_risk['risk_category'] = 'High'
    
    low_risk = df_sub_sorted.tail(4).copy()
    low_risk['risk_category'] = 'Low'
    
    mid_idx = len(df_sub_sorted) // 2
    medium_risk = df_sub_sorted.iloc[mid_idx-1 : mid_idx+2].copy()
    medium_risk['risk_category'] = 'Medium'
    
    selected_meta = pd.concat([high_risk, medium_risk, low_risk])
    selected_ids_set = set(selected_meta['SK_ID_CURR'].tolist())
    
    print(f"Selected {len(selected_ids_set)} applications.")

    # Dictionary to hold the filtered data for each table
    # Key: table name, Value: DataFrame
    extracted_data = {}

    # 2. Process Main Application File
    print("Processing application_test.csv...")
    app_test_path = data_dir / "application_test.csv"
    if app_test_path.exists():
        df_app = pd.read_csv(app_test_path)
        extracted_data['application.csv'] = df_app[df_app['SK_ID_CURR'].isin(selected_ids_set)]
    else:
        print("Error: application_test.csv not found.")
        return

    # 3. Process Related Tables (Direct link via SK_ID_CURR)
    related_files = [
        "bureau.csv",
        "previous_application.csv",
        "POS_CASH_balance.csv",
        "credit_card_balance.csv",
        "installments_payments.csv"
    ]

    for filename in related_files:
        file_path = data_dir / filename
        if file_path.exists():
            print(f"Processing {filename}...")
            # Use chunks for large files
            chunks = []
            try:
                for chunk in pd.read_csv(file_path, chunksize=100000):
                    filtered_chunk = chunk[chunk['SK_ID_CURR'].isin(selected_ids_set)]
                    if not filtered_chunk.empty:
                        chunks.append(filtered_chunk)
                
                if chunks:
                    extracted_data[filename] = pd.concat(chunks)
                else:
                    # Create empty DF with correct columns if no data found
                    header = pd.read_csv(file_path, nrows=0)
                    extracted_data[filename] = header
                    print(f"  No records found for selected IDs in {filename}")
            except Exception as e:
                print(f"  Error processing {filename}: {e}")
        else:
             print(f"  Warning: {filename} not found.")

    # 4. Process Bureau Balance (Indirect link via SK_ID_BUREAU)
    # We need the SK_ID_BUREAU values from the extracted bureau data
    if 'bureau.csv' in extracted_data and not extracted_data['bureau.csv'].empty:
        bureau_ids = set(extracted_data['bureau.csv']['SK_ID_BUREAU'].unique())
        print(f"Processing bureau_balance.csv for {len(bureau_ids)} bureau IDs...")
        
        bb_path = data_dir / "bureau_balance.csv"
        if bb_path.exists():
            chunks = []
            try:
                for chunk in pd.read_csv(bb_path, chunksize=100000):
                    filtered_chunk = chunk[chunk['SK_ID_BUREAU'].isin(bureau_ids)]
                    if not filtered_chunk.empty:
                        chunks.append(filtered_chunk)
                
                if chunks:
                    extracted_data['bureau_balance.csv'] = pd.concat(chunks)
                else:
                     header = pd.read_csv(bb_path, nrows=0)
                     extracted_data['bureau_balance.csv'] = header
                     print("  No records found in bureau_balance.csv")
            except Exception as e:
                print(f"  Error processing bureau_balance.csv: {e}")
        else:
            print("  Warning: bureau_balance.csv not found.")
    else:
        print("  Skipping bureau_balance.csv (no relevant bureau records found).")

    # 5. Save Files for each Application
    print("\nSaving files...")
    
    for _, row in selected_meta.iterrows():
        app_id = int(row['SK_ID_CURR'])
        risk = row['risk_category'].lower()
        base_name = f"application_{app_id}_{risk}_risk"
        
        # Create a folder for each application to keep things organized (optional, but cleaner)
        # Or just prefix files. The user asked for "make files", let's do prefixed files in the main dir.
        
        # Save Main Application Record
        if 'application.csv' in extracted_data:
            app_data = extracted_data['application.csv']
            app_record = app_data[app_data['SK_ID_CURR'] == app_id]
            if not app_record.empty:
                app_record.to_csv(output_dir / f"{base_name}_application.csv", index=False)
        
        # Save Related Records
        for filename, df in extracted_data.items():
            if filename == 'application.csv': continue
            
            if filename == 'bureau_balance.csv':
                # Linked via SK_ID_BUREAU
                # Get this app's bureau IDs
                if 'bureau.csv' in extracted_data:
                    bureau_data = extracted_data['bureau.csv']
                    app_bureau_ids = bureau_data[bureau_data['SK_ID_CURR'] == app_id]['SK_ID_BUREAU']
                    
                    if not df.empty:
                        related_records = df[df['SK_ID_BUREAU'].isin(app_bureau_ids)]
                        if not related_records.empty:
                             related_records.to_csv(output_dir / f"{base_name}_{filename}", index=False)
            
            else:
                # Linked via SK_ID_CURR
                if not df.empty:
                    related_records = df[df['SK_ID_CURR'] == app_id]
                    if not related_records.empty:
                        related_records.to_csv(output_dir / f"{base_name}_{filename}", index=False)
                        
        print(f"Generated files for {base_name}")

if __name__ == "__main__":
    create_relational_test_files()
