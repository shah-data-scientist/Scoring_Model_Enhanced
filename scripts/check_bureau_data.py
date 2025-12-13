
from pathlib import Path

import pandas as pd


def check_bureau():
    data_dir = Path("data")
    bureau_path = data_dir / "bureau.csv"

    # Check ID 123343
    target_id = 123343

    print(f"Checking bureau data for ID {target_id}...")

    chunks = []
    for chunk in pd.read_csv(bureau_path, chunksize=50000):
        filtered = chunk[chunk['SK_ID_CURR'] == target_id]
        if not filtered.empty:
            chunks.append(filtered)

    if chunks:
        df = pd.concat(chunks)
        print(df)
        print("\nValue Counts for CREDIT_ACTIVE:")
        print(df['CREDIT_ACTIVE'].value_counts())
    else:
        print("ID not found in bureau.csv")

if __name__ == "__main__":
    check_bureau()
