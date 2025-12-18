"""Preprocessing Pipeline for Batch Predictions

Wraps the existing preprocessing modules to transform raw CSV data
into the 189 features required by the production model.
"""

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import preprocessing modules
from src.domain_features import create_domain_features
from src.feature_aggregation import (
    aggregate_bureau,
    aggregate_credit_card,
    aggregate_installments,
    aggregate_pos_cash,
    aggregate_previous_applications,
)
from src.feature_engineering import clean_column_names

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class PreprocessingPipeline:
    """Pipeline to preprocess raw CSV files for batch predictions.

    Takes 7 raw CSV files and produces 189 model-ready features.
    """

    def __init__(self, feature_names_path: Path = None, use_precomputed: bool = True):
        """Initialize preprocessing pipeline.

        Args:
            feature_names_path: Path to feature_names.csv (189 features in order)
            use_precomputed: If True, use precomputed features for known applications

        """
        self.feature_names_path = feature_names_path or (
            PROJECT_ROOT / "data" / "processed" / "feature_names.csv"
        )

        # Path to scaler and medians
        self.scaler_path = PROJECT_ROOT / "data" / "processed" / "scaler.joblib"
        self.medians_path = PROJECT_ROOT / "data" / "processed" / "medians.json"

        self.use_precomputed = use_precomputed
        self.precomputed_features = None
        self.scaler = None
        self.medians = None

        # Load expected feature names and order
        if self.feature_names_path.exists():
            feature_df = pd.read_csv(self.feature_names_path)
            self.expected_features = feature_df['feature'].tolist()
        else:
            self.expected_features = None
            print(f"Warning: {self.feature_names_path} not found. Feature order may not match model.")

        # Load scaler
        if self.scaler_path.exists():
            try:
                self.scaler = joblib.load(self.scaler_path)
                print(f"[INFO] Loaded scaler from {self.scaler_path}")
            except Exception as e:
                print(f"[WARNING] Failed to load scaler: {e}")
        else:
            print(f"[WARNING] Scaler not found at {self.scaler_path}. Live features will be unscaled!")

        # Load medians
        if self.medians_path.exists():
            try:
                with open(self.medians_path) as f:
                    self.medians = json.load(f)
                print(f"[INFO] Loaded {len(self.medians)} medians from {self.medians_path}")
            except Exception as e:
                print(f"[WARNING] Failed to load medians: {e}")

        # Load precomputed features if available
        if self.use_precomputed:
            # Try Parquet first (10-100x faster), fallback to CSV
            parquet_path = PROJECT_ROOT / "data" / "processed" / "precomputed_features.parquet"
            csv_path = PROJECT_ROOT / "data" / "processed" / "X_train.csv"
            train_ids_path = PROJECT_ROOT / "data" / "processed" / "train_ids.csv"

            if parquet_path.exists():
                print("[INFO] Loading precomputed features from Parquet (fast!)...")
                X_train = pd.read_parquet(parquet_path)

                # SK_ID_CURR should already be in the parquet file
                if 'SK_ID_CURR' in X_train.columns:
                    self.precomputed_features = X_train.set_index('SK_ID_CURR')
                else:
                    # If not, load from train_ids.csv
                    train_ids = pd.read_csv(train_ids_path)
                    X_train['SK_ID_CURR'] = train_ids['SK_ID_CURR'].values
                    self.precomputed_features = X_train.set_index('SK_ID_CURR')

                print(f"[INFO] Loaded {len(self.precomputed_features)} precomputed feature sets from Parquet")

            elif csv_path.exists() and train_ids_path.exists():
                print("[INFO] Loading precomputed features from CSV (slow - consider converting to Parquet)...")
                X_train = pd.read_csv(csv_path)
                train_ids = pd.read_csv(train_ids_path)

                # Add SK_ID_CURR as index
                X_train['SK_ID_CURR'] = train_ids['SK_ID_CURR'].values
                self.precomputed_features = X_train.set_index('SK_ID_CURR')
                print(f"[INFO] Loaded {len(self.precomputed_features)} precomputed feature sets from CSV")
            else:
                print("[WARNING] Precomputed features not found. Will use full preprocessing for all applications.")
                self.precomputed_features = None

    def aggregate_data(
        self,
        application_df: pd.DataFrame,
        bureau_df: pd.DataFrame = None,
        bureau_balance_df: pd.DataFrame = None,
        previous_application_df: pd.DataFrame = None,
        pos_cash_df: pd.DataFrame = None,
        credit_card_df: pd.DataFrame = None,
        installments_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Aggregate auxiliary tables and merge with application data.

        Args:
            application_df: Main application DataFrame
            bureau_df: Bureau credit history
            bureau_balance_df: Bureau monthly balances
            previous_application_df: Previous applications
            pos_cash_df: POS/cash balances
            credit_card_df: Credit card balances
            installments_df: Payment installments

        Returns:
            Aggregated DataFrame

        """
        result_df = application_df.copy()

        # Aggregate bureau data
        if bureau_df is not None:
            print(f"  Aggregating bureau data ({len(bureau_df)} rows)...")
            bureau_agg = aggregate_bureau(bureau_df, bureau_balance_df)
            result_df = result_df.merge(bureau_agg, on='SK_ID_CURR', how='left')
            print(f"    Added {len(bureau_agg.columns)-1} bureau features")

        # Aggregate previous applications
        if previous_application_df is not None:
            print(f"  Aggregating previous applications ({len(previous_application_df)} rows)...")
            prev_agg = aggregate_previous_applications(previous_application_df)
            result_df = result_df.merge(prev_agg, on='SK_ID_CURR', how='left')
            print(f"    Added {len(prev_agg.columns)-1} previous application features")

        # Aggregate POS cash
        if pos_cash_df is not None:
            print(f"  Aggregating POS cash balance ({len(pos_cash_df)} rows)...")
            pos_agg = aggregate_pos_cash(pos_cash_df)
            result_df = result_df.merge(pos_agg, on='SK_ID_CURR', how='left')
            print(f"    Added {len(pos_agg.columns)-1} POS cash features")

        # Aggregate credit card
        if credit_card_df is not None:
            print(f"  Aggregating credit card balance ({len(credit_card_df)} rows)...")
            cc_agg = aggregate_credit_card(credit_card_df)
            result_df = result_df.merge(cc_agg, on='SK_ID_CURR', how='left')
            print(f"    Added {len(cc_agg.columns)-1} credit card features")

        # Aggregate installments
        if installments_df is not None:
            print(f"  Aggregating installments ({len(installments_df)} rows)...")
            inst_agg = aggregate_installments(installments_df)
            result_df = result_df.merge(inst_agg, on='SK_ID_CURR', how='left')
            print(f"    Added {len(inst_agg.columns)-1} installment features")

        print(f"  Total features after aggregation: {len(result_df.columns)}")
        return result_df

    def create_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific engineered features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with engineered features

        """
        print("  Creating domain features...")
        df = create_domain_features(df)
        print(f"    Features after domain engineering: {len(df.columns)}")
        return df

    def encode_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features and clean column names.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with encoded features and clean names

        """
        print("  Encoding categorical features...")

        # For prediction, we need to create dummy DataFrame for the encoder
        # to learn the categories, but we won't actually use it
        # Instead, we'll manually handle the encoding based on known categories

        # Save original columns
        original_cols = df.columns.tolist()

        # One-hot encode low-cardinality categoricals
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        # Exclude SK_ID_CURR from encoding
        if 'SK_ID_CURR' in categorical_cols:
            categorical_cols.remove('SK_ID_CURR')

        # For each categorical column with low cardinality (<= 10), one-hot encode
        for col in categorical_cols:
            if col in df.columns:
                unique_count = df[col].nunique()
                if unique_count <= 10:
                    # One-hot encode (drop_first=True to match training)
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
                    df = pd.concat([df, dummies], axis=1)
                    # Drop original column
                    df = df.drop(columns=[col])

        print(f"    Features after encoding: {len(df.columns)}")

        # Clean column names
        print("  Cleaning column names...")
        df = clean_column_names(df)

        return df

    def align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align features with expected model input (189 features).

        Handles:
        - Missing features (add as 0)
        - Extra features (drop)
        - Feature order (reorder to match training)

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with exactly 189 features in correct order

        """
        if self.expected_features is None:
            print("  Warning: Cannot align features - expected feature list not loaded")
            return df

        print("  Aligning features to model input (189 features)...")
        print(f"    Current features: {len(df.columns)}")

        # Drop SK_ID_CURR if present (not a feature)
        if 'SK_ID_CURR' in df.columns:
            sk_id_curr = df['SK_ID_CURR']
            df = df.drop(columns=['SK_ID_CURR'])
        else:
            sk_id_curr = None

        current_features = set(df.columns)
        expected_features_set = set(self.expected_features)

        # Missing features
        missing_features = expected_features_set - current_features
        if missing_features:
            print(f"    Adding {len(missing_features)} missing features (filled with 0)")
            for feat in missing_features:
                df[feat] = 0

        # Extra features
        extra_features = current_features - expected_features_set
        if extra_features:
            print(f"    Dropping {len(extra_features)} extra features")
            df = df.drop(columns=list(extra_features))

        # Reorder to match expected order
        df = df[self.expected_features]

        # Add back SK_ID_CURR if it existed
        if sk_id_curr is not None:
            df.insert(0, 'SK_ID_CURR', sk_id_curr)

        print(f"    Final features: {len(df.columns) - (1 if 'SK_ID_CURR' in df.columns else 0)}")
        return df

    def impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values using median for numerical features.
        Uses global training medians if available, otherwise batch median.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with imputed values

        """
        print("  Imputing missing values...")

        # Save SK_ID_CURR if present
        sk_id_curr = None
        if 'SK_ID_CURR' in df.columns:
            sk_id_curr = df['SK_ID_CURR']
            df = df.drop(columns=['SK_ID_CURR'])

        # Impute numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns

        # Use global medians if available
        if self.medians is not None:
            print("    Using global training medians for imputation")
            for col in numerical_cols:
                if col in self.medians:
                    df[col] = df[col].fillna(self.medians[col])
                else:
                    # Fallback to batch median
                    median_val = df[col].median()
                    if pd.isna(median_val):
                        median_val = 0
                    df[col] = df[col].fillna(median_val)
        else:
            # Fallback to batch median (legacy behavior)
            print("    Using batch medians for imputation (Warning: may be unstable for small batches)")
            missing_counts = df[numerical_cols].isnull().sum()
            cols_with_missing = missing_counts[missing_counts > 0]

            if len(cols_with_missing) > 0:
                print(f"    Imputing {len(cols_with_missing)} columns with missing values")
                for col in cols_with_missing.index:
                    median_val = df[col].median()
                    if pd.isna(median_val):
                        median_val = 0  # If all values are NaN, use 0
                    df[col] = df[col].fillna(median_val)

        # Add back SK_ID_CURR
        if sk_id_curr is not None:
            df.insert(0, 'SK_ID_CURR', sk_id_curr)

        return df

    def process(
        self,
        dataframes: dict[str, pd.DataFrame],
        keep_sk_id: bool = True
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Full preprocessing pipeline from raw CSVs to model-ready features.

        Uses precomputed features for known applications (from training data)
        and processes new applications through the full pipeline.

        Args:
            dataframes: Dictionary of {filename: DataFrame}
            keep_sk_id: Whether to keep SK_ID_CURR in output

        Returns:
            Tuple of (features_df, sk_id_curr_series)

        """
        print("\n" + "="*80)
        print("PREPROCESSING PIPELINE")
        print("="*80 + "\n")

        # Extract DataFrames
        application_df = dataframes.get('application.csv')
        bureau_df = dataframes.get('bureau.csv')
        bureau_balance_df = dataframes.get('bureau_balance.csv')
        previous_application_df = dataframes.get('previous_application.csv')
        pos_cash_df = dataframes.get('POS_CASH_balance.csv')
        credit_card_df = dataframes.get('credit_card_balance.csv')
        installments_df = dataframes.get('installments_payments.csv')

        if application_df is None:
            raise ValueError("application.csv is required")

        # Save SK_ID_CURR
        sk_id_curr = application_df['SK_ID_CURR'].copy()
        n_applications = len(application_df)

        print(f"Processing {n_applications} applications...\n")

        # Check if we can use precomputed features for any applications
        if self.precomputed_features is not None:
            app_ids = set(sk_id_curr.values)
            known_ids = app_ids & set(self.precomputed_features.index)

            if known_ids:
                print(f"[INFO] {len(known_ids)}/{n_applications} applications found in precomputed features")
                print("[INFO] Using precomputed features for known applications (100% accurate)\n")

                # Separate known and unknown applications
                known_mask = sk_id_curr.isin(known_ids)
                unknown_mask = ~known_mask

                # Get precomputed features for known applications (vectorized)
                known_app_ids = sk_id_curr[known_mask].values
                if len(known_app_ids) > 0:
                    # Use .loc with array for vectorized lookup - much faster than loop
                    known_features = self.precomputed_features.loc[known_app_ids].copy()
                    known_features['SK_ID_CURR'] = known_app_ids
                    known_features = known_features.reset_index(drop=True)
                else:
                    known_features = None

                # Process unknown applications through full pipeline
                if unknown_mask.sum() > 0:
                    print(f"[INFO] Processing {unknown_mask.sum()} new applications through full pipeline...\n")

                    # Filter all dataframes for unknown applications
                    unknown_ids = set(sk_id_curr[unknown_mask].values)
                    unknown_app_df = application_df[application_df['SK_ID_CURR'].isin(unknown_ids)]

                    # Filter auxiliary data for unknown applications
                    unknown_dataframes = {'application.csv': unknown_app_df}

                    if bureau_df is not None:
                        unknown_dataframes['bureau.csv'] = bureau_df[bureau_df['SK_ID_CURR'].isin(unknown_ids)]
                    if bureau_balance_df is not None:
                        # Get bureau IDs for unknown applications
                        if 'bureau.csv' in unknown_dataframes and len(unknown_dataframes['bureau.csv']) > 0:
                            unknown_bureau_ids = set(unknown_dataframes['bureau.csv']['SK_ID_BUREAU'])
                            unknown_dataframes['bureau_balance.csv'] = bureau_balance_df[
                                bureau_balance_df['SK_ID_BUREAU'].isin(unknown_bureau_ids)
                            ]
                    if previous_application_df is not None:
                        unknown_dataframes['previous_application.csv'] = previous_application_df[
                            previous_application_df['SK_ID_CURR'].isin(unknown_ids)
                        ]
                        # Get previous app IDs for unknown applications
                        if len(unknown_dataframes['previous_application.csv']) > 0:
                            unknown_prev_ids = set(unknown_dataframes['previous_application.csv']['SK_ID_PREV'])
                            if pos_cash_df is not None:
                                unknown_dataframes['POS_CASH_balance.csv'] = pos_cash_df[
                                    pos_cash_df['SK_ID_PREV'].isin(unknown_prev_ids)
                                ]
                            if credit_card_df is not None:
                                unknown_dataframes['credit_card_balance.csv'] = credit_card_df[
                                    credit_card_df['SK_ID_PREV'].isin(unknown_prev_ids)
                                ]
                            if installments_df is not None:
                                unknown_dataframes['installments_payments.csv'] = installments_df[
                                    installments_df['SK_ID_PREV'].isin(unknown_prev_ids)
                                ]

                    # Process unknown applications
                    unknown_features = self._process_full_pipeline(unknown_dataframes, keep_sk_id=True)
                else:
                    unknown_features = None

                # Combine known and unknown features
                if known_features is not None and unknown_features is not None:
                    # Ensure both have SK_ID_CURR
                    if 'SK_ID_CURR' not in known_features.columns:
                        known_features['SK_ID_CURR'] = sk_id_curr[known_mask].values
                    if 'SK_ID_CURR' not in unknown_features.columns:
                        unknown_features['SK_ID_CURR'] = sk_id_curr[unknown_mask].values

                    features_df = pd.concat([known_features, unknown_features], axis=0, ignore_index=True)

                    # Reorder to match original application order
                    id_to_idx = {app_id: idx for idx, app_id in enumerate(sk_id_curr)}
                    features_df['_order'] = features_df['SK_ID_CURR'].map(id_to_idx)
                    features_df = features_df.sort_values('_order').drop(columns=['_order']).reset_index(drop=True)
                elif known_features is not None:
                    features_df = known_features
                else:
                    features_df = unknown_features

                print("\n" + "="*80)
                print(f"PREPROCESSING COMPLETE: {len(features_df)} rows x {len([c for c in features_df.columns if c != 'SK_ID_CURR'])} features")
                print(f"  - {len(known_ids)} from precomputed (100% accurate)")
                print(f"  - {unknown_mask.sum()} from full pipeline")
                print("="*80 + "\n")

                if not keep_sk_id and 'SK_ID_CURR' in features_df.columns:
                    features_df = features_df.drop(columns=['SK_ID_CURR'])

                return features_df, sk_id_curr
            print("[INFO] No applications found in precomputed features. Processing all through full pipeline.\n")

        # If no precomputed features available or no matches, process all through full pipeline
        features_df = self._process_full_pipeline(dataframes, keep_sk_id)

        print("\n" + "="*80)
        print(f"PREPROCESSING COMPLETE: {len(features_df)} rows x {len([c for c in features_df.columns if c != 'SK_ID_CURR'])} features")
        print("="*80 + "\n")

        return features_df, sk_id_curr

    def _process_full_pipeline(
        self,
        dataframes: dict[str, pd.DataFrame],
        keep_sk_id: bool = True
    ) -> pd.DataFrame:
        """Process applications through the full preprocessing pipeline.

        Args:
            dataframes: Dictionary of {filename: DataFrame}
            keep_sk_id: Whether to keep SK_ID_CURR in output

        Returns:
            DataFrame with processed features

        """
        application_df = dataframes.get('application.csv')
        bureau_df = dataframes.get('bureau.csv')
        bureau_balance_df = dataframes.get('bureau_balance.csv')
        previous_application_df = dataframes.get('previous_application.csv')
        pos_cash_df = dataframes.get('POS_CASH_balance.csv')
        credit_card_df = dataframes.get('credit_card_balance.csv')
        installments_df = dataframes.get('installments_payments.csv')

        if application_df is None:
            raise ValueError("application.csv is required")

        sk_id_curr = application_df['SK_ID_CURR'].copy()

        # Step 1: Aggregate auxiliary tables
        print("Step 1: Aggregating auxiliary tables")
        df = self.aggregate_data(
            application_df=application_df,
            bureau_df=bureau_df,
            bureau_balance_df=bureau_balance_df,
            previous_application_df=previous_application_df,
            pos_cash_df=pos_cash_df,
            credit_card_df=credit_card_df,
            installments_df=installments_df
        )

        # Step 2: Create engineered features
        print("\nStep 2: Creating engineered features")
        df = self.create_engineered_features(df)

        # Step 3: Encode categoricals and clean names
        print("\nStep 3: Encoding categorical features")
        df = self.encode_and_clean(df)

        # Step 4: Impute missing values
        print("\nStep 4: Imputing missing values")
        df = self.impute_missing_values(df)

        # Step 5: Align features with model expectations
        print("\nStep 5: Aligning features with model")
        df = self.align_features(df)

        # Step 6: Scale features
        if self.scaler is not None:
            print("\nStep 6: Scaling features")
            # Save SK_ID_CURR if present
            current_sk_id = None
            if 'SK_ID_CURR' in df.columns:
                current_sk_id = df['SK_ID_CURR']
                df_to_scale = df.drop(columns=['SK_ID_CURR'])
            else:
                df_to_scale = df

            # Scale
            try:
                # Ensure columns match scaler expectations
                if list(df_to_scale.columns) == self.expected_features:
                    scaled_values = self.scaler.transform(df_to_scale)
                    df = pd.DataFrame(scaled_values, columns=df_to_scale.columns, index=df_to_scale.index)

                    # Add back SK_ID_CURR
                    if current_sk_id is not None:
                        df.insert(0, 'SK_ID_CURR', current_sk_id)
                    print("    Features scaled successfully")
                else:
                    print("    Warning: Column mismatch, skipping scaling")
            except Exception as e:
                print(f"    Warning: Scaling failed: {e}")

        # Extract final features
        if keep_sk_id:
            features_df = df  # Keep SK_ID_CURR
        elif 'SK_ID_CURR' in df.columns:
            features_df = df.drop(columns=['SK_ID_CURR'])
        else:
            features_df = df

        return features_df
