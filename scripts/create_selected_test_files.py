
from pathlib import Path

import pandas as pd


def create_selected_test_files():
    # Define paths
    base_dir = Path()
    submission_path = base_dir / "results" / "submission.csv"
    test_data_path = base_dir / "data" / "application_test.csv"
    output_dir = base_dir / "data" / "end_user_tests"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading submission file from {submission_path}...")
    try:
        df_sub = pd.read_csv(submission_path)
    except FileNotFoundError:
        print(f"Error: Submission file not found at {submission_path}")
        return

    print(f"Reading application test data from {test_data_path}...")
    try:
        df_test = pd.read_csv(test_data_path)
    except FileNotFoundError:
        print(f"Error: Application test file not found at {test_data_path}")
        return

    # Sort by TARGET to identify risk levels
    df_sub_sorted = df_sub.sort_values(by="TARGET", ascending=False)

    # Select High Risk (Top 3)
    high_risk = df_sub_sorted.head(3).copy()
    high_risk['risk_category'] = 'High'

    # Select Low Risk (Bottom 4)
    low_risk = df_sub_sorted.tail(4).copy()
    low_risk['risk_category'] = 'Low'

    # Select Medium Risk (Middle 3)
    mid_idx = len(df_sub_sorted) // 2
    medium_risk = df_sub_sorted.iloc[mid_idx-1 : mid_idx+2].copy()
    medium_risk['risk_category'] = 'Medium'

    # Combine selected IDs
    selected_meta = pd.concat([high_risk, medium_risk, low_risk])
    selected_ids = selected_meta['SK_ID_CURR'].tolist()

    print(f"Selected {len(selected_ids)} applications:")
    print(selected_meta[['SK_ID_CURR', 'TARGET', 'risk_category']])

    # Filter application data
    df_selected = df_test[df_test['SK_ID_CURR'].isin(selected_ids)]

    # Merge with risk info for clarity in the output file (optional but helpful)
    # We won't merge it into the raw test data file to keep the schema identical to original
    # But we can print or save a metadata file.

    # Save combined CSV
    combined_csv_path = output_dir / "selected_10_applications.csv"
    df_selected.to_csv(combined_csv_path, index=False)
    print(f"Saved combined CSV to {combined_csv_path}")

    # Save individual JSON files and CSVs
    for _, row in df_selected.iterrows():
        app_id = row['SK_ID_CURR']

        # Get risk info
        risk_info = selected_meta[selected_meta['SK_ID_CURR'] == app_id].iloc[0]
        risk_cat = risk_info['risk_category']

        # Save as JSON (record orientation usually best for API payloads)
        # We'll save the raw features
        json_path = output_dir / f"application_{app_id}_{risk_cat.lower()}_risk.json"
        row.to_json(json_path)

        # Save as single-row CSV
        csv_path = output_dir / f"application_{app_id}_{risk_cat.lower()}_risk.csv"
        row.to_frame().T.to_csv(csv_path, index=False)

    print(f"Created individual JSON and CSV files in {output_dir}")

if __name__ == "__main__":
    create_selected_test_files()
