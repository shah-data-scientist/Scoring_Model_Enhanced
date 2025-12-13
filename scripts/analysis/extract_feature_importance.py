"""Extract Feature Importance from Production Model.

This script:
1. Loads the production model from MLflow
2. Extracts feature importance
3. Identifies critical features (85% cumulative importance or top 50)
4. Maps engineered features back to raw features
5. Saves configuration files for API validation
"""
import json
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MLFLOW_TRACKING_URI, REGISTERED_MODELS

# Set MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def load_production_model():
    """Load the production model from MLflow registry."""
    print("Loading production model from MLflow...")
    model_name = REGISTERED_MODELS['production']

    try:
        # Get latest version of production model
        client = mlflow.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")

        if not versions:
            raise ValueError(f"No versions found for model '{model_name}'")

        # Get the latest version
        latest_version = max(versions, key=lambda x: int(x.version))
        print(f"  Found model: {model_name}, version: {latest_version.version}")

        # Load model
        model_uri = f"models:/{model_name}/{latest_version.version}"
        model = mlflow.sklearn.load_model(model_uri)

        return model, latest_version
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting to load from latest run...")

        # Fallback: get from latest run
        experiment = client.get_experiment_by_name("credit_scoring_final_delivery")
        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["metrics.fbeta_score DESC"],
                max_results=1
            )
            if runs:
                run_id = runs[0].info.run_id
                model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
                return model, None

        raise ValueError("Could not load production model from MLflow")

def get_feature_importance(model, feature_names):
    """Extract feature importance from model."""
    print("\nExtracting feature importance...")

    # Check if model has feature_importances_ attribute
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_).flatten()
    else:
        raise ValueError("Model does not have feature importance")

    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })

    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False)

    # Calculate cumulative importance
    importance_df['cumulative_importance'] = importance_df['importance'].cumsum() / importance_df['importance'].sum()

    print(f"  Total features: {len(importance_df)}")
    print(f"  Top 10 features:\n{importance_df.head(10)[['feature', 'importance']]}")

    return importance_df

def identify_critical_features(importance_df, threshold=0.85, max_features=50):
    """Identify critical features based on cumulative importance threshold."""
    print(f"\nIdentifying critical features (threshold={threshold}, max={max_features})...")

    # Features explaining threshold% of importance
    critical_by_threshold = importance_df[importance_df['cumulative_importance'] <= threshold]

    # Top N features
    critical_by_count = importance_df.head(max_features)

    # Take whichever is fewer
    if len(critical_by_threshold) < len(critical_by_count):
        critical_features = critical_by_threshold['feature'].tolist()
        method = f"{threshold*100}% cumulative importance"
    else:
        critical_features = critical_by_count['feature'].tolist()
        method = f"top {max_features} features"

    print(f"  Selected {len(critical_features)} critical features using {method}")

    return critical_features, importance_df['feature'].tolist()

def map_engineered_to_raw_features(feature_names):
    """Map engineered features back to raw features.

    This analyzes feature names to identify which raw data files
    and columns they originate from.
    """
    print("\nMapping engineered features to raw features...")

    feature_mapping = {}
    raw_feature_sources = {}

    for feature in feature_names:
        # Determine source and raw features
        sources = []

        # Check for aggregation patterns
        if '_AGG_' in feature or '_SUM' in feature or '_MEAN' in feature or '_MAX' in feature or '_MIN' in feature:
            # This is an aggregated feature
            if 'BUREAU' in feature:
                sources.append(('bureau', extract_raw_columns(feature, 'bureau')))
            if 'PREV' in feature or 'PREVIOUS' in feature:
                sources.append(('previous_application', extract_raw_columns(feature, 'previous')))
            if 'INSTALL' in feature:
                sources.append(('installments_payments', extract_raw_columns(feature, 'installments')))
            if 'CC' in feature or 'CREDIT' in feature:
                sources.append(('credit_card_balance', extract_raw_columns(feature, 'credit_card')))
            if 'POS' in feature:
                sources.append(('POS_CASH_balance', extract_raw_columns(feature, 'pos')))

        # Domain features
        elif 'DOMAIN_' in feature:
            sources.append(('application', extract_domain_raw_features(feature)))

        # Direct application features
        else:
            sources.append(('application', [feature]))

        feature_mapping[feature] = sources

        # Track which raw files are needed
        for source_file, _ in sources:
            if source_file not in raw_feature_sources:
                raw_feature_sources[source_file] = set()

    print(f"  Mapped {len(feature_names)} features")
    print(f"  Raw data sources identified: {list(raw_feature_sources.keys())}")

    return feature_mapping, raw_feature_sources

def extract_raw_columns(feature_name, prefix):
    """Extract potential raw column names from engineered feature name."""
    # This is a simplified heuristic - you may need to adjust based on actual feature engineering
    parts = feature_name.split('_')
    potential_cols = []

    for i, part in enumerate(parts):
        if part in ['SUM', 'MEAN', 'MAX', 'MIN', 'STD', 'COUNT', 'AGG']:
            # The part before this is likely the column name
            if i > 0:
                potential_cols.append('_'.join(parts[:i]))

    if not potential_cols:
        potential_cols = [feature_name]

    return potential_cols

def extract_domain_raw_features(domain_feature):
    """Extract raw features used in domain feature calculation."""
    # Map domain features to raw features they use
    # This should match your actual domain feature engineering logic

    domain_mapping = {
        'DOMAIN_INCOME_CREDIT_RATIO': ['AMT_INCOME_TOTAL', 'AMT_CREDIT'],
        'DOMAIN_ANNUITY_INCOME_RATIO': ['AMT_ANNUITY', 'AMT_INCOME_TOTAL'],
        'DOMAIN_CREDIT_GOODS_RATIO': ['AMT_CREDIT', 'AMT_GOODS_PRICE'],
        'DOMAIN_DAYS_EMPLOYED_RATIO': ['DAYS_EMPLOYED', 'DAYS_BIRTH'],
        'DOMAIN_INCOME_PER_PERSON': ['AMT_INCOME_TOTAL', 'CNT_FAM_MEMBERS'],
    }

    for pattern, raw_features in domain_mapping.items():
        if pattern in domain_feature:
            return raw_features

    return [domain_feature]

def load_feature_names():
    """Load feature names from processed data."""
    feature_names_file = PROJECT_ROOT / "data" / "processed" / "feature_names.csv"

    if feature_names_file.exists():
        df = pd.read_csv(feature_names_file)
        if 'feature' in df.columns:
            return df['feature'].tolist()
        return df.iloc[:, 0].tolist()

    # Fallback: load from X_train columns
    x_train_file = PROJECT_ROOT / "data" / "processed" / "X_train.csv"
    if x_train_file.exists():
        df = pd.read_csv(x_train_file, nrows=1)
        return df.columns.tolist()

    raise FileNotFoundError("Could not find feature names")

def save_configuration(critical_features, all_features, feature_mapping, importance_df):
    """Save configuration files for API validation."""
    config_dir = PROJECT_ROOT / "config"
    config_dir.mkdir(exist_ok=True)

    print("\nSaving configuration files...")

    # 1. Critical features list
    critical_config = {
        "critical_features": critical_features,
        "num_critical": len(critical_features),
        "threshold": 0.85,
        "description": "Features that must be present in input data"
    }

    with open(config_dir / "critical_features.json", "w") as f:
        json.dump(critical_config, f, indent=2)
    print(f"  Saved: {config_dir / 'critical_features.json'}")

    # 2. All features list
    all_features_config = {
        "all_features": all_features,
        "num_features": len(all_features)
    }

    with open(config_dir / "all_features.json", "w") as f:
        json.dump(all_features_config, f, indent=2)
    print(f"  Saved: {config_dir / 'all_features.json'}")

    # 3. Feature importance
    importance_df.to_csv(config_dir / "feature_importance.csv", index=False)
    print(f"  Saved: {config_dir / 'feature_importance.csv'}")

    # 4. Required raw data files
    required_files = {
        "application.csv": {
            "required": True,
            "description": "Main application data (combines train/test)"
        },
        "bureau.csv": {
            "required": True,
            "description": "Bureau credit history"
        },
        "bureau_balance.csv": {
            "required": True,
            "description": "Bureau credit monthly balance"
        },
        "previous_application.csv": {
            "required": True,
            "description": "Previous applications at Home Credit"
        },
        "credit_card_balance.csv": {
            "required": True,
            "description": "Credit card monthly balance"
        },
        "installments_payments.csv": {
            "required": True,
            "description": "Payment installments history"
        },
        "POS_CASH_balance.csv": {
            "required": True,
            "description": "POS and cash loans monthly balance"
        }
    }

    with open(config_dir / "required_files.json", "w") as f:
        json.dump(required_files, f, indent=2)
    print(f"  Saved: {config_dir / 'required_files.json'}")

    # 5. Feature mapping (simplified for now)
    with open(config_dir / "feature_mapping.json", "w") as f:
        json.dump({"note": "Detailed mapping to be implemented"}, f, indent=2)
    print(f"  Saved: {config_dir / 'feature_mapping.json'}")

def main():
    """Main execution."""
    print("="*80)
    print("FEATURE IMPORTANCE EXTRACTION & ANALYSIS")
    print("="*80)

    try:
        # Load production model
        model, version = load_production_model()

        # Load feature names
        feature_names = load_feature_names()
        print(f"\nLoaded {len(feature_names)} feature names from processed data")

        # Extract feature importance
        importance_df = get_feature_importance(model, feature_names)

        # Identify critical features
        critical_features, all_features = identify_critical_features(importance_df)

        # Map to raw features
        feature_mapping, raw_sources = map_engineered_to_raw_features(all_features)

        # Save configuration
        save_configuration(critical_features, all_features, feature_mapping, importance_df)

        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Total features: {len(all_features)}")
        print(f"Critical features: {len(critical_features)}")
        print(f"Raw data sources: {list(raw_sources.keys())}")
        print("\nTop 20 critical features:")
        for i, feat in enumerate(critical_features[:20], 1):
            imp = importance_df[importance_df['feature'] == feat]['importance'].values[0]
            print(f"  {i:2d}. {feat:50s} {imp:.6f}")

        print("\n✓ Phase 1 completed successfully!")
        print(f"✓ Configuration files saved to: {PROJECT_ROOT / 'config'}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
