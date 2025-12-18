import pandas as pd
import sys
import os
from sqlalchemy import create_engine, text

# Add project root to path
sys.path.append(os.getcwd())

from backend.database import DATABASE_URL
from api.drift_detection import get_training_reference_data

def inspect_mismatch():
    # 1. Get latest batch
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        # Debug: list all batches
        print("Checking all batches in DB:")
        batches = conn.execute(text("SELECT id, status FROM prediction_batches ORDER BY id DESC LIMIT 5")).fetchall()
        for b in batches:
            print(f"  Batch {b[0]}: {b[1]}")

        result = conn.execute(text("SELECT id FROM prediction_batches WHERE status='completed' OR status='COMPLETED' ORDER BY id DESC LIMIT 1"))
        batch_id = result.scalar()
        
        if not batch_id:
            # Fallback to just taking the last one if available
            if batches:
                batch_id = batches[0][0]
                print(f"Fallback: Using Batch ID {batch_id} (Status: {batches[0][1]})")
            else:
                print("No batches found at all.")
                return

        print(f"Analyzing Batch ID: {batch_id}")
        
        # Get raw data samples
        data = conn.execute(text(f"SELECT raw_data FROM raw_applications WHERE batch_id={batch_id} LIMIT 1"))
        import json
        raw_json = data.scalar()
        if raw_json:
            # It's stored as JSON in DB (Text column) or via SQLAlchemy it might come out as dict if type is JSON
            # In crud.py it is stored as JSON.
            if isinstance(raw_json, str):
                row = json.loads(raw_json)
            else:
                row = raw_json
            
            batch_cols = list(row.keys())
            print(f"\nBatch (Raw) Columns ({len(batch_cols)}):")
            print(batch_cols[:20])
        else:
            print("No raw data in batch.")
            return

    # 2. Get Reference Data
    print("\nLoading Reference Data...")
    try:
        ref_df = get_training_reference_data()
        ref_cols = ref_df.columns.tolist()
        print(f"Reference (Processed) Columns ({len(ref_cols)}):")
        print(ref_cols[:20])
        
        # 3. Check Overlap
        overlap = set(batch_cols).intersection(ref_cols)
        print(f"\nOverlap Count: {len(overlap)}")
        if overlap:
            print("Overlapping Columns:", list(overlap)[:20])
        else:
            print("NO OVERLAP DETECTED.")
            
    except Exception as e:
        print(f"Error loading reference: {e}")

if __name__ == "__main__":
    inspect_mismatch()
