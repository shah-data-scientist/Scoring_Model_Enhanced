
import pandas as pd
from pathlib import Path
import random

def anonymize_sk_id_curr():
    base_dir = Path(".")
    test_dir = base_dir / "data" / "end_user_tests"
    
    if not test_dir.exists():
        print(f"Error: Directory {test_dir} not found.")
        return

    csv_files = list(test_dir.glob("*.csv"))
    if not csv_files:
        print("No CSV files found in end_user_tests.")
        return

    print("Step 1: collecting unique SK_ID_CURR values...")
    unique_ids = set()
    
    for fp in csv_files:
        try:
            # Read only SK_ID_CURR to be fast
            # We assume the column exists; if not, we catch the error (some files might not have it)
            df = pd.read_csv(fp)
            if 'SK_ID_CURR' in df.columns:
                unique_ids.update(df['SK_ID_CURR'].dropna().unique())
        except Exception as e:
            print(f"  Warning reading {fp.name}: {e}")

    if not unique_ids:
        print("No SK_ID_CURR values found.")
        return

    sorted_ids = sorted(list(unique_ids))
    print(f"Found {len(sorted_ids)} unique SK_ID_CURR values.")

    # Step 2: Create Mapping
    # Map to new IDs starting from 500000
    new_ids = list(range(500000, 500000 + len(sorted_ids)))
    
    # Shuffle to ensure it's not just a shift
    # random.shuffle(new_ids) # Optional, but linear is fine too. Let's keep it simple.
    
    id_map = dict(zip(sorted_ids, new_ids))
    
    # Save mapping
    mapping_df = pd.DataFrame(list(id_map.items()), columns=['original_sk_id_curr', 'new_sk_id_curr'])
    mapping_path = test_dir / "id_mapping.csv"
    mapping_df.to_csv(mapping_path, index=False)
    print(f"Saved ID mapping to {mapping_path}")

    # Step 3: Apply Mapping
    print("Step 3: Replacing IDs in files...")
    
    for fp in csv_files:
        if fp.name == "id_mapping.csv":
            continue
            
        try:
            df = pd.read_csv(fp)
            modified = False
            
            if 'SK_ID_CURR' in df.columns:
                # Replace values
                # We use map().fillna(original) to be safe, though all should be in map
                df['SK_ID_CURR'] = df['SK_ID_CURR'].map(id_map).fillna(df['SK_ID_CURR'])
                # Ensure integer type
                df['SK_ID_CURR'] = df['SK_ID_CURR'].astype(int)
                modified = True
                print(f"  Updated SK_ID_CURR in {fp.name}")
            
            if modified:
                df.to_csv(fp, index=False)
                
        except Exception as e:
            print(f"  Error processing {fp.name}: {e}")

    print("Anonymization complete.")

if __name__ == "__main__":
    anonymize_sk_id_curr()
