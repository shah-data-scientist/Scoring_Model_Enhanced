"""
Script to anonymize SK_ID_CURR values in end_user_tests data files.
Maintains referential integrity across all related CSV files.
"""
import pandas as pd
import os
from pathlib import Path
import hashlib

def generate_anonymous_id(original_id: int, salt: str = "credit_scoring_2025") -> int:
    """Generate a deterministic anonymous ID from the original ID using hashing."""
    # Create a hash of the original ID + salt
    hash_input = f"{original_id}_{salt}".encode('utf-8')
    hash_output = hashlib.sha256(hash_input).hexdigest()
    # Convert first 8 hex characters to integer (ensures reasonable ID length)
    anonymous_id = int(hash_output[:8], 16) % 900000 + 100000  # 6-digit IDs
    return anonymous_id

def anonymize_end_user_tests():
    """Anonymize SK_ID_CURR in all end_user_tests CSV files."""
    
    # Define paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "end_user_tests"
    
    # Files that contain SK_ID_CURR
    files_with_sk_id = [
        "application.csv",
        "bureau.csv",
        "bureau_balance.csv",
        "credit_card_balance.csv",
        "installments_payments.csv",
        "POS_CASH_balance.csv",
        "previous_application.csv"
    ]
    
    # Step 1: Collect all unique SK_ID_CURR values across all files
    print("Collecting unique SK_ID_CURR values...")
    all_sk_ids = set()
    
    for filename in files_with_sk_id:
        filepath = data_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            if 'SK_ID_CURR' in df.columns:
                all_sk_ids.update(df['SK_ID_CURR'].dropna().unique())
                print(f"  {filename}: {df['SK_ID_CURR'].nunique()} unique IDs")
    
    print(f"\nTotal unique SK_ID_CURR values: {len(all_sk_ids)}")
    
    # Step 2: Create mapping from original to anonymous IDs
    print("\nCreating anonymous ID mapping...")
    id_mapping = {}
    for original_id in sorted(all_sk_ids):
        anonymous_id = generate_anonymous_id(int(original_id))
        id_mapping[original_id] = anonymous_id
    
    print(f"Sample mappings:")
    for i, (orig, anon) in enumerate(list(id_mapping.items())[:5]):
        print(f"  {orig} -> {anon}")
    
    # Step 3: Apply anonymization to each file
    print("\nAnonymizing files...")
    for filename in files_with_sk_id:
        filepath = data_dir / filename
        if filepath.exists():
            print(f"  Processing {filename}...")
            df = pd.read_csv(filepath)
            
            if 'SK_ID_CURR' in df.columns:
                # Apply mapping
                df['SK_ID_CURR'] = df['SK_ID_CURR'].map(id_mapping)
                
                # Save back to file
                df.to_csv(filepath, index=False)
                print(f"    ✓ Updated {len(df)} rows")
        else:
            print(f"  ⚠ File not found: {filename}")
    
    # Step 4: Save mapping for reference (optional)
    mapping_file = data_dir / "sk_id_mapping.csv"
    mapping_df = pd.DataFrame([
        {'original_sk_id': orig, 'anonymous_sk_id': anon}
        for orig, anon in id_mapping.items()
    ])
    mapping_df.to_csv(mapping_file, index=False)
    print(f"\n✓ Mapping saved to {mapping_file.name}")
    
    print("\n✓ Anonymization complete!")
    print(f"  {len(all_sk_ids)} unique IDs anonymized across {len(files_with_sk_id)} files")

if __name__ == "__main__":
    anonymize_end_user_tests()
