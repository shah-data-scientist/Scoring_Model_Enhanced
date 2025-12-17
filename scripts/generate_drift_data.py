import os
import pandas as pd
import numpy as np

def generate_drift_data(
    input_path: str = "data/samples",
    output_path: str = "data/drift_samples",
    num_samples: int = 1000,
    income_shift_pct: float = 0.20, # 20% increase in income
    ext_source_2_missing_pct: float = 0.15 # 15% additional missing values for EXT_SOURCE_2
):
    """
    Generates synthetic data with introduced data drift for testing purposes.

    Args:
        input_path (str): Path to the directory containing original sample data.
        output_path (str): Path to the directory where drift data will be saved.
        num_samples (int): Number of synthetic samples to generate.
        income_shift_pct (float): Percentage to increase AMT_INCOME_TOTAL by.
        ext_source_2_missing_pct (float): Percentage of additional missing values
                                          to introduce in EXT_SOURCE_2.
    """
    os.makedirs(output_path, exist_ok=True)
    print(f"Generating drift data in: {output_path}")

    # --- Process application.csv ---
    app_df_original = pd.read_csv(os.path.join(input_path, "application.csv"))

    # Sample data to create new batch
    # Use replacement to allow for more samples than original if num_samples > len(original)
    app_df_drift = app_df_original.sample(n=num_samples, replace=True, random_state=42).reset_index(drop=True)

    # Introduce Numerical Drift: Increase AMT_INCOME_TOTAL
    if 'AMT_INCOME_TOTAL' in app_df_drift.columns:
        app_df_drift['AMT_INCOME_TOTAL'] *= (1 + income_shift_pct)
        print(f"  - Introduced numerical drift: AMT_INCOME_TOTAL increased by {income_shift_pct*100:.0f}%")

    # Introduce Categorical Drift: Shift NAME_INCOME_TYPE distribution
    if 'NAME_INCOME_TYPE' in app_df_drift.columns:
        original_dist = app_df_original['NAME_INCOME_TYPE'].value_counts(normalize=True)
        
        # Example: Increase 'Working' proportion, decrease 'Pensioner'
        if 'Working' in original_dist.index and 'Pensioner' in original_dist.index:
            working_increase = 0.10 # 10% increase in working proportion
            pensioner_decrease = 0.05 # 5% decrease in pensioner proportion

            # Calculate new proportions - simplified for demonstration
            # In a real scenario, you'd want to ensure proportions sum to 1
            new_working_prop = min(original_dist['Working'] + working_increase, 1.0)
            new_pensioner_prop = max(original_dist['Pensioner'] - pensioner_decrease, 0.0)
            
            # Adjust other categories proportionally
            other_categories = original_dist.drop(['Working', 'Pensioner'], errors='ignore')
            other_sum = other_categories.sum()
            
            if other_sum > 0:
                scale_factor = (1.0 - new_working_prop - new_pensioner_prop) / (other_sum + 1e-9)
                new_other_props = other_categories * scale_factor
                
                new_dist = pd.Series(
                    {
                        'Working': new_working_prop,
                        'Pensioner': new_pensioner_prop,
                        **new_other_props.to_dict()
                    }
                ).clip(lower=0).fillna(0) # Ensure no negative proportions and fill NaN from drop

                # Renormalize to ensure sum is 1, handling potential floating point issues
                new_dist = new_dist / new_dist.sum()

                # Generate new categorical data based on new distribution
                app_df_drift['NAME_INCOME_TYPE'] = np.random.choice(
                    new_dist.index,
                    size=len(app_df_drift),
                    p=new_dist.values
                )
                print("  - Introduced categorical drift: NAME_INCOME_TYPE distribution shifted")
        else:
             print("  - Skipped categorical drift: 'Working' or 'Pensioner' not found in NAME_INCOME_TYPE")


    # Introduce Missing Value Drift: EXT_SOURCE_2
    if 'EXT_SOURCE_2' in app_df_drift.columns:
        num_to_make_nan = int(len(app_df_drift) * ext_source_2_missing_pct)
        # Ensure we don't try to make more NaNs than available non-NaN values
        non_nan_indices = app_df_drift[app_df_drift['EXT_SOURCE_2'].notna()].index
        
        if len(non_nan_indices) > num_to_make_nan:
            indices_to_nan = np.random.choice(non_nan_indices, num_to_make_nan, replace=False)
            app_df_drift.loc[indices_to_nan, 'EXT_SOURCE_2'] = np.nan
            print(f"  - Introduced missing value drift: {ext_source_2_missing_pct*100:.0f}% additional NaNs in EXT_SOURCE_2")
        else:
            print(f"  - Skipped missing value drift for EXT_SOURCE_2: Not enough non-NaN values to modify {ext_source_2_missing_pct*100:.0f}%")

    # Introduce Out-of-Range Drift: DAYS_BIRTH (e.g., extremely young)
    if 'DAYS_BIRTH' in app_df_drift.columns:
        # Assuming DAYS_BIRTH is negative (days since birth)
        # Make some values much closer to 0 (younger)
        num_to_shift = int(len(app_df_drift) * 0.05) # Shift 5% of values
        
        # Select random indices
        indices_to_shift = np.random.choice(app_df_drift.index, num_to_shift, replace=False)
        
        # Shift days birth to be very young (e.g., -5000 days = ~13 years old)
        app_df_drift.loc[indices_to_shift, 'DAYS_BIRTH'] = np.random.randint(-7000, -3650, num_to_shift)
        print(f"  - Introduced out-of-range drift: {num_to_shift} applications made unusually young in DAYS_BIRTH")

    app_df_drift.to_csv(os.path.join(output_path, "application.csv"), index=False)
    print(f"  - Saved application.csv to {output_path}")

    # --- Process bureau.csv (copy without drift for simplicity, or add drift if needed) ---
    bureau_df_original = pd.read_csv(os.path.join(input_path, "bureau.csv"))
    # For now, just copy bureau.csv, but ensure SK_ID_CURR matches the new application_df_drift
    
    # Filter bureau entries to only include SK_ID_CURR present in the new application_df_drift
    bureau_df_drift = bureau_df_original[
        bureau_df_original['SK_ID_CURR'].isin(app_df_drift['SK_ID_CURR'])
    ].copy()

    # To ensure there's enough bureau data for the new applications,
    # if sampling from original, we might need to duplicate bureau records
    # for the sampled SK_ID_CURR.
    # For simplicity, we'll just resample from the existing bureau data,
    # ensuring that SK_ID_CURR values align with the new application_df_drift.
    
    # Map new SK_ID_CURR to original ones for bureau data if there's a 1:1 relationship
    # If the original sampling for app_df_drift used replace=True, 
    # we need to ensure the bureau data corresponds.
    
    # A more robust approach would involve mapping original SK_ID_CURR to new ones
    # or generating bureau data completely from scratch based on the new app_df_drift's SK_ID_CURR.
    # For this task, a simple filtering or resampling on SK_ID_CURR will suffice for demonstration.

    # Option 1: Simple filter (if app_df_drift has a subset of original SK_ID_CURR)
    # bureau_df_drift = bureau_df_original[bureau_df_original['SK_ID_CURR'].isin(app_df_drift['SK_ID_CURR'])]

    # Option 2: Resample bureau data to match num_samples and align SK_ID_CURR
    # This is more complex and might not accurately reflect real relationships.
    # Let's use a simpler approach: for every SK_ID_CURR in app_df_drift,
    # try to find its bureau records from the original set. If not found, create a dummy one
    
    unique_sk_id_curr_in_app_drift = app_df_drift['SK_ID_CURR'].unique()
    bureau_records_for_new_apps = bureau_df_original[
        bureau_df_original['SK_ID_CURR'].isin(unique_sk_id_curr_in_app_drift)
    ].copy()

    # If some SK_ID_CURR from the new app_df_drift don't have bureau records in the original set
    # (due to sampling with replacement or just not being present), we might need to
    # synthesize some or ignore. For this demo, we'll just use the available ones.

    if not bureau_records_for_new_apps.empty:
        bureau_df_drift = bureau_records_for_new_apps.sample(n=num_samples, replace=True, random_state=42).reset_index(drop=True)
        # Now, replace SK_ID_CURR in bureau_df_drift to match new app_df_drift's SK_ID_CURR
        # This is a simplification and assumes each app_df_drift row needs a bureau entry.
        # In reality, a client can have multiple bureau records.
        bureau_df_drift['SK_ID_CURR'] = app_df_drift['SK_ID_CURR'].values
    else:
        # Fallback if no bureau data exists for the sampled SK_ID_CURR
        print("  - Warning: No matching bureau data for sampled SK_ID_CURR. Generating empty bureau.csv.")
        bureau_df_drift = pd.DataFrame(columns=bureau_df_original.columns)


    bureau_df_drift.to_csv(os.path.join(output_path, "bureau.csv"), index=False)
    print(f"  - Saved bureau.csv to {output_path}")

    # Copy other auxiliary files as-is
    aux_files = [
        "bureau_balance.csv",
        "credit_card_balance.csv",
        "installments_payments.csv",
        "POS_CASH_balance.csv",
        "previous_application.csv",
    ]
    for fname in aux_files:
        original_path = os.path.join(input_path, fname)
        new_path = os.path.join(output_path, fname)
        if os.path.exists(original_path):
            df_aux = pd.read_csv(original_path)
            # Filter auxiliary files to only include SK_ID_CURR present in the new application_df_drift
            if 'SK_ID_CURR' in df_aux.columns:
                df_aux = df_aux[df_aux['SK_ID_CURR'].isin(app_df_drift['SK_ID_CURR'])]
            df_aux.to_csv(new_path, index=False)
            print(f"  - Copied and filtered {fname} to {output_path}")
        else:
            print(f"  - Warning: {fname} not found in {input_path}, skipping.")

    print(f"\nSynthetic data with drift generated successfully in {output_path}")

if __name__ == "__main__":
    generate_drift_data()
