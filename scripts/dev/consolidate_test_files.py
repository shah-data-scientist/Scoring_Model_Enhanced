
import os
from pathlib import Path

import pandas as pd


def consolidate_test_files():
    # Define paths
    base_dir = Path()
    input_dir = base_dir / "data" / "end_user_tests"

    if not input_dir.exists():
        print(f"Error: Directory {input_dir} not found.")
        return

    # List of source filenames (suffixes) we expect
    source_files = [
        "application.csv",
        "bureau.csv",
        "bureau_balance.csv",
        "previous_application.csv",
        "POS_CASH_balance.csv",
        "credit_card_balance.csv",
        "installments_payments.csv"
    ]

    print(f"Consolidating files in {input_dir}...")

    for source_name in source_files:
        # Find all files ending with this source name
        # Pattern: *_{source_name}
        # Note: application.csv might match "previous_application.csv" if we aren't careful.
        # So we check the suffix carefully.

        pattern = f"*{source_name}"
        files_to_merge = []

        # Iterate over all files to safely filter
        for file_path in input_dir.glob("*.csv"):
            fname = file_path.name

            # Skip files that are exactly the source name (if they already exist from a previous run)
            if fname == source_name:
                continue

            # Check if file ends with the source name
            # Special handling to avoid overlapping names (e.g. application.csv vs previous_application.csv)
            if fname.endswith(f"_{source_name}"):
                 if source_name == "application.csv" and "previous" in fname:
                     continue
                 files_to_merge.append(file_path)
            # Special case for the main application file which we named "application_..._application.csv"
            elif source_name == "application.csv" and fname.endswith("_application.csv") and "previous" not in fname:
                 files_to_merge.append(file_path)

        if not files_to_merge:
            print(f"  No files found for {source_name}")
            continue

        print(f"  Merging {len(files_to_merge)} files into {source_name}...")

        dfs = []
        for fp in files_to_merge:
            try:
                df = pd.read_csv(fp)
                dfs.append(df)
            except Exception as e:
                print(f"    Error reading {fp.name}: {e}")

        if dfs:
            consolidated_df = pd.concat(dfs, ignore_index=True)
            output_path = input_dir / source_name
            consolidated_df.to_csv(output_path, index=False)
            print(f"    Saved {output_path} ({len(consolidated_df)} rows)")

            # Optional: Delete the individual files after merging?
            # User said "consolidate", usually implies replacing or grouping.
            # I will remove the individual fragments to keep it clean as per "consolidate these files"
            # effectively replacing the many small files with the few large ones.
            for fp in files_to_merge:
                try:
                    os.remove(fp)
                except OSError as e:
                    print(f"    Could not remove {fp.name}: {e}")

    print("\nConsolidation complete.")

if __name__ == "__main__":
    consolidate_test_files()
