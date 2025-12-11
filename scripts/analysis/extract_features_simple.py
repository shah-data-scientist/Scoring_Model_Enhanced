"""
Simple Feature Importance Extraction.

Loads model directly from file and extracts feature importance.
"""
import joblib
import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def find_best_model():
    """Find the best model file."""
    print("Searching for best model...")

    # Use best XGBoost model (has feature importance)
    best_model = PROJECT_ROOT / "models" / "best_xgboost_model.pkl"
    if best_model.exists():
        print(f"  Found best model: {best_model}")
        return best_model

    # Fallback: check production model
    production_model = PROJECT_ROOT / "models" / "production_model.pkl"
    if production_model.exists():
        print(f"  Found production model: {production_model}")
        return production_model

    # Last resort: search in mlruns/5 (final delivery experiment)
    models_dir = PROJECT_ROOT / "mlruns" / "5" / "models"
    if models_dir.exists():
        model_files = list(models_dir.glob("*/artifacts/model.pkl"))
        if model_files:
            print(f"  Found {len(model_files)} model files in experiment 5")
            return model_files[0]

    raise FileNotFoundError("No model files found")

def load_model(model_path):
    """Load model from file."""
    print(f"\nLoading model from: {model_path}")
    model = joblib.load(model_path)
    print(f"  Model type: {type(model).__name__}")
    return model

def load_feature_names(model):
    """Load feature names from model or processed data."""
    # First, try to get feature names from the model itself
    if hasattr(model, 'feature_names_in_'):
        print("\nLoading feature names from model.feature_names_in_")
        return model.feature_names_in_.tolist()

    if hasattr(model, 'get_booster'):
        booster = model.get_booster()
        if booster.feature_names:
            print("\nLoading feature names from model booster")
            return booster.feature_names

    # Fallback: Try feature_names.csv
    feature_names_file = PROJECT_ROOT / "data" / "processed" / "feature_names.csv"

    if feature_names_file.exists():
        print("\nLoading feature names from feature_names.csv")
        df = pd.read_csv(feature_names_file)
        if 'feature' in df.columns:
            return df['feature'].tolist()
        else:
            return df.iloc[:, 0].tolist()

    # Last resort: load from X_train columns
    x_train_file = PROJECT_ROOT / "data" / "processed" / "X_train.csv"
    if x_train_file.exists():
        print("\nLoading feature names from X_train.csv columns")
        df = pd.read_csv(x_train_file, nrows=1)
        return df.columns.tolist()

    raise FileNotFoundError("Could not find feature names")

def get_feature_importance(model, feature_names):
    """Extract feature importance from model."""
    print("\nExtracting feature importance...")

    # Check if model has feature_importances_ attribute (tree-based models)
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        print("  Using feature_importances_ attribute")
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_).flatten()
        print("  Using coef_ attribute")
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
    total_importance = importance_df['importance'].sum()
    importance_df['cumulative_importance'] = importance_df['importance'].cumsum() / total_importance

    print(f"  Total features: {len(importance_df)}")
    print(f"\n  Top 10 features:")
    for idx, row in importance_df.head(10).iterrows():
        print(f"    {row['feature']:50s} {row['importance']:10.6f} ({row['cumulative_importance']*100:5.2f}%)")

    return importance_df

def identify_critical_features(importance_df, threshold=0.85, max_features=50):
    """Identify critical features."""
    print(f"\nIdentifying critical features (threshold={threshold*100}%, max={max_features})...")

    # Features explaining threshold% of importance
    critical_by_threshold = importance_df[importance_df['cumulative_importance'] <= threshold]

    # Top N features
    critical_by_count = importance_df.head(max_features)

    # Take whichever is fewer
    if len(critical_by_threshold) <= len(critical_by_count):
        critical_features = critical_by_threshold['feature'].tolist()
        method = f"{threshold*100}% cumulative importance"
        num_features = len(critical_by_threshold)
    else:
        critical_features = critical_by_count['feature'].tolist()
        method = f"top {max_features} features"
        num_features = max_features

    print(f"  Selected {num_features} critical features using {method}")

    return critical_features, importance_df['feature'].tolist()

def save_configuration(critical_features, all_features, importance_df):
    """Save configuration files."""
    config_dir = PROJECT_ROOT / "config"
    config_dir.mkdir(exist_ok=True)

    print("\nSaving configuration files...")

    # 1. Critical features
    critical_config = {
        "critical_features": critical_features,
        "num_critical": len(critical_features),
        "threshold": 0.85,
        "description": "Features that must be present for predictions"
    }

    critical_file = config_dir / "critical_features.json"
    with open(critical_file, "w") as f:
        json.dump(critical_config, f, indent=2)
    print(f"  Saved: {critical_file}")

    # 2. All features
    all_features_config = {
        "all_features": all_features,
        "num_features": len(all_features)
    }

    all_file = config_dir / "all_features.json"
    with open(all_file, "w") as f:
        json.dump(all_features_config, f, indent=2)
    print(f"  Saved: {all_file}")

    # 3. Feature importance CSV
    importance_file = config_dir / "feature_importance.csv"
    importance_df.to_csv(importance_file, index=False)
    print(f"  Saved: {importance_file}")

    # 4. Required raw data files
    required_files = {
        "application.csv": {
            "required": True,
            "description": "Main application data"
        },
        "bureau.csv": {
            "required": True,
            "description": "Bureau credit history"
        },
        "bureau_balance.csv": {
            "required": True,
            "description": "Bureau monthly balance"
        },
        "previous_application.csv": {
            "required": True,
            "description": "Previous applications"
        },
        "credit_card_balance.csv": {
            "required": True,
            "description": "Credit card monthly balance"
        },
        "installments_payments.csv": {
            "required": True,
            "description": "Payment installments"
        },
        "POS_CASH_balance.csv": {
            "required": True,
            "description": "POS and cash loans balance"
        }
    }

    files_config = config_dir / "required_files.json"
    with open(files_config, "w") as f:
        json.dump(required_files, f, indent=2)
    print(f"  Saved: {files_config}")

def main():
    """Main execution."""
    print("=" * 80)
    print("FEATURE IMPORTANCE EXTRACTION")
    print("=" * 80)

    try:
        # Find and load model
        model_path = find_best_model()
        model = load_model(model_path)

        # Load feature names (from model)
        feature_names = load_feature_names(model)
        print(f"Loaded {len(feature_names)} feature names")

        # Extract feature importance
        importance_df = get_feature_importance(model, feature_names)

        # Identify critical features
        critical_features, all_features = identify_critical_features(importance_df)

        # Save configuration
        save_configuration(critical_features, all_features, importance_df)

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total features: {len(all_features)}")
        print(f"Critical features: {len(critical_features)}")
        print(f"\nTop 20 most important features:")
        for i, feat in enumerate(critical_features[:20], 1):
            imp_row = importance_df[importance_df['feature'] == feat].iloc[0]
            print(f"  {i:2d}. {feat:50s} {imp_row['importance']:.6f} ({imp_row['cumulative_importance']*100:5.2f}%)")

        print("\nPhase 1 completed successfully!")
        print(f"Configuration files saved to: {PROJECT_ROOT / 'config'}")

        return 0

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
