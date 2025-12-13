"""Map Model Features to Raw Input Features.

This script:
1. Loads the best available model
2. Gets model feature importance
3. Analyzes preprocessing code to map engineered features → raw features
4. Identifies critical RAW features from original CSV files
5. Creates validation configuration for API
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import pandas as pd

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def load_model_and_importance():
    """Load model and extract feature importance."""
    print("Loading model...")

    # Load best available model
    model_path = PROJECT_ROOT / "models" / "best_xgboost_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)
    print(f"  Model type: {type(model).__name__}")

    # Get feature names and importance
    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_.tolist()
    else:
        raise ValueError("Model doesn't have feature_names_in_")

    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        raise ValueError("Model doesn't have feature_importances_")

    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'model_feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    importance_df['cumulative_importance'] = (
        importance_df['importance'].cumsum() / importance_df['importance'].sum()
    )

    print(f"  Loaded {len(feature_names)} model features")
    return model, importance_df

def analyze_feature_sources(model_features):
    """Analyze which raw CSV files and columns are needed for each model feature.

    This is based on understanding the preprocessing pipeline:
    - Direct features: come directly from application.csv
    - Aggregate features: computed from related tables (bureau, previous_application, etc.)
    - Domain features: derived from application.csv columns
    - Encoded categorical features: from application.csv categories
    """
    print("\nAnalyzing feature sources...")

    raw_feature_map = {}  # model_feature -> {file: [raw_columns]}

    for feat in model_features:
        sources = defaultdict(list)

        # 1. Check for aggregation keywords (from auxiliary tables)
        if any(x in feat for x in ['BUREAU', 'PREV', 'INSTALL', 'CC', 'CREDIT_CARD', 'POS']):
            if 'BUREAU' in feat:
                sources['bureau.csv'].append(extract_base_column(feat, 'BUREAU'))
                if 'BALANCE' in feat or 'STATUS' in feat:
                    sources['bureau_balance.csv'].append(extract_base_column(feat, 'BUREAU'))

            if 'PREV' in feat or 'PREVIOUS' in feat:
                sources['previous_application.csv'].append(extract_base_column(feat, 'PREV'))

            if 'INSTALL' in feat:
                sources['installments_payments.csv'].append(extract_base_column(feat, 'INSTALL'))

            if 'CC' in feat or 'CREDIT_CARD' in feat:
                sources['credit_card_balance.csv'].append(extract_base_column(feat, 'CC'))

            if 'POS' in feat:
                sources['POS_CASH_balance.csv'].append(extract_base_column(feat, 'POS'))

        # 2. Domain/derived features (from application.csv)
        elif any(x in feat for x in ['RATIO', 'PER_PERSON', 'MEAN', 'MIN', 'MAX']):
            # These are computed from application.csv columns
            raw_cols = extract_domain_raw_features(feat)
            if raw_cols:
                sources['application.csv'].extend(raw_cols)

        # 3. Encoded categorical features (NAME_, CODE_, FLAG_, etc.)
        elif any(feat.startswith(x) for x in ['NAME_', 'CODE_', 'FLAG_', 'WALLSMATERIAL_',
                                                'HOUSETYPE_', 'FONDKAPREMONT_', 'EMERGENCYSTATE_']):
            # Extract base column name (before the encoded value)
            base_col = extract_categorical_base(feat)
            sources['application.csv'].append(base_col)

        # 4. Direct numeric features (AMT_, DAYS_, CNT_, etc.)
        elif any(feat.startswith(x) for x in ['AMT_', 'DAYS_', 'CNT_', 'EXT_SOURCE_',
                                               'OWN_CAR_AGE', 'REGION_', 'HOUR_', 'WEEKDAY_',
                                               'REG_', 'LIVE_', 'DEF_', 'OBS_', 'FLOORSMAX_',
                                               'YEARS_', 'TOTALAREA_', 'ELEVATORS_',
                                               'ENTRANCES_', 'LIVINGAPARTMENTS_']):
            sources['application.csv'].append(feat)

        # 5. Boolean flags
        elif feat.startswith('HAS_'):
            # Derived features like HAS_CHILDREN
            raw_cols = extract_boolean_raw_features(feat)
            sources['application.csv'].extend(raw_cols)

        # 6. Other features - assume from application
        else:
            sources['application.csv'].append(feat)

        # Clean up and deduplicate
        raw_feature_map[feat] = {
            file: list(set(cols)) for file, cols in sources.items() if cols
        }

    print(f"  Mapped {len(model_features)} model features to raw sources")
    return raw_feature_map

def extract_base_column(feature_name, prefix):
    """Extract base column name from aggregated feature."""
    # Remove common aggregation suffixes
    for suffix in ['_SUM', '_MEAN', '_MAX', '_MIN', '_VAR', '_COUNT', '_AGG']:
        if suffix in feature_name:
            return feature_name.split(suffix)[0].replace(prefix + '_', '')
    return feature_name

def extract_categorical_base(feature_name):
    """Extract base column name from encoded categorical feature."""
    # Examples:
    # NAME_EDUCATION_TYPE_Higher_education -> NAME_EDUCATION_TYPE
    # CODE_GENDER_M -> CODE_GENDER
    # FLAG_OWN_CAR_Y -> FLAG_OWN_CAR

    parts = feature_name.split('_')

    # Find where the actual category value starts (usually last part or last 2 parts)
    # Common patterns: NAME_X_Y_value, CODE_X_value, FLAG_X_value

    if feature_name.startswith('NAME_'):
        # NAME_X_TYPE_value or NAME_X_Y_value
        # Find last part that looks like a value (starts with capital or is short)
        for i in range(len(parts) - 1, 1, -1):
            if parts[i][0].isupper() or len(parts[i]) <= 2:
                return '_'.join(parts[:i])
        return '_'.join(parts[:-1])

    if feature_name.startswith(('CODE_', 'FLAG_', 'WALLSMATERIAL_', 'HOUSETYPE_')):
        # Usually: PREFIX_COLUMN_value
        return '_'.join(parts[:-1])

    return feature_name

def extract_domain_raw_features(feature_name):
    """Extract raw features used in domain feature calculation."""
    domain_mappings = {
        'CREDIT_TO_GOODS_RATIO': ['AMT_CREDIT', 'AMT_GOODS_PRICE'],
        'CREDIT_TO_ANNUITY_RATIO': ['AMT_CREDIT', 'AMT_ANNUITY'],
        'CREDIT_TO_INCOME_RATIO': ['AMT_CREDIT', 'AMT_INCOME_TOTAL'],
        'ANNUITY_TO_INCOME_RATIO': ['AMT_ANNUITY', 'AMT_INCOME_TOTAL'],
        'INCOME_PER_PERSON': ['AMT_INCOME_TOTAL', 'CNT_FAM_MEMBERS'],
        'DEBT_TO_INCOME_RATIO': ['AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL'],
        'EXT_SOURCE_MEAN': ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'],
        'EXT_SOURCE_MIN': ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'],
        'EXT_SOURCE_MAX': ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'],
        'TOTAL_DOCUMENTS_PROVIDED': ['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4',
                                      'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7',
                                      'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10',
                                      'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13',
                                      'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16',
                                      'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19',
                                      'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21'],
    }

    for pattern, raw_cols in domain_mappings.items():
        if pattern in feature_name:
            return raw_cols

    return []

def extract_boolean_raw_features(feature_name):
    """Extract raw features for boolean derived features."""
    if 'HAS_CHILDREN' in feature_name:
        return ['CNT_CHILDREN']
    return []

def consolidate_raw_features(raw_feature_map, importance_df, threshold=0.85):
    """Consolidate all raw features needed and identify critical ones.
    """
    print("\nConsolidating raw features...")

    # Track all raw features by file
    raw_features_by_file = defaultdict(set)
    feature_importance_map = {}  # raw_feature -> cumulative importance from model features

    for model_feat, sources in raw_feature_map.items():
        # Get this model feature's importance
        imp_row = importance_df[importance_df['model_feature'] == model_feat]
        if imp_row.empty:
            continue

        importance = imp_row['importance'].values[0]
        cum_importance = imp_row['cumulative_importance'].values[0]

        # Add all raw features from this model feature
        for file, raw_cols in sources.items():
            for raw_col in raw_cols:
                raw_features_by_file[file].add(raw_col)

                # Track max importance for this raw feature
                key = f"{file}:{raw_col}"
                if key not in feature_importance_map:
                    feature_importance_map[key] = {
                        'importance': importance,
                        'cumulative': cum_importance,
                        'model_features': [model_feat]
                    }
                else:
                    # If this raw feature contributes to multiple model features,
                    # sum the importances
                    feature_importance_map[key]['importance'] += importance
                    feature_importance_map[key]['model_features'].append(model_feat)

    # Sort raw features by importance
    sorted_raw_features = sorted(
        feature_importance_map.items(),
        key=lambda x: x[1]['importance'],
        reverse=True
    )

    # Identify critical raw features (contributing to top model features)
    critical_model_features = importance_df[
        importance_df['cumulative_importance'] <= threshold
    ]['model_feature'].tolist()

    critical_raw_features = defaultdict(set)
    for model_feat in critical_model_features:
        if model_feat in raw_feature_map:
            for file, raw_cols in raw_feature_map[model_feat].items():
                critical_raw_features[file].update(raw_cols)

    print(f"  Total raw features needed: {len(feature_importance_map)}")
    print(f"  Critical raw features: {sum(len(v) for v in critical_raw_features.values())}")
    print("\n  Raw features by file:")
    for file in sorted(raw_features_by_file.keys()):
        total = len(raw_features_by_file[file])
        critical = len(critical_raw_features.get(file, set()))
        print(f"    {file:30s} Total: {total:3d}  Critical: {critical:3d}")

    return raw_features_by_file, critical_raw_features, feature_importance_map

def save_raw_feature_configuration(raw_features_by_file, critical_raw_features,
                                   feature_importance_map, importance_df):
    """Save configuration files for API validation."""
    config_dir = PROJECT_ROOT / "config"
    config_dir.mkdir(exist_ok=True)

    print("\nSaving configuration files...")

    # 1. Critical RAW features by file
    critical_config = {
        file: sorted(list(features))
        for file, features in critical_raw_features.items()
    }
    critical_config['_metadata'] = {
        'description': 'Critical RAW features that MUST be present in uploaded files',
        'total_critical_features': sum(len(v) for v in critical_raw_features.values()),
        'threshold': 0.85
    }

    with open(config_dir / "critical_raw_features.json", "w") as f:
        json.dump(critical_config, f, indent=2)
    print(f"  Saved: {config_dir / 'critical_raw_features.json'}")

    # 2. All RAW features by file
    all_config = {
        file: sorted(list(features))
        for file, features in raw_features_by_file.items()
    }
    all_config['_metadata'] = {
        'description': 'All RAW features used by the model',
        'total_features': sum(len(v) for v in raw_features_by_file.values())
    }

    with open(config_dir / "all_raw_features.json", "w") as f:
        json.dump(all_config, f, indent=2)
    print(f"  Saved: {config_dir / 'all_raw_features.json'}")

    # 3. Raw feature importance mapping
    importance_list = [
        {
            'file_column': key,
            'file': key.split(':')[0],
            'column': key.split(':')[1],
            'importance': float(data['importance']),  # Convert to Python float
            'contributing_model_features': data['model_features'][:5]  # Top 5
        }
        for key, data in sorted(
            feature_importance_map.items(),
            key=lambda x: x[1]['importance'],
            reverse=True
        )
    ]

    with open(config_dir / "raw_feature_importance.json", "w") as f:
        json.dump(importance_list[:100], f, indent=2)  # Save top 100
    print(f"  Saved: {config_dir / 'raw_feature_importance.json'}")

    # 4. Keep the model feature importance for reference
    importance_df.to_csv(config_dir / "model_feature_importance.csv", index=False)
    print(f"  Saved: {config_dir / 'model_feature_importance.csv'}")

def main():
    """Main execution."""
    print("=" * 80)
    print("RAW FEATURE MAPPING AND ANALYSIS")
    print("=" * 80)

    try:
        # Load model and get feature importance
        model, importance_df = load_model_and_importance()

        print("\nTop 10 model features:")
        for idx, row in importance_df.head(10).iterrows():
            print(f"  {row['model_feature']:50s} {row['importance']:.6f} ({row['cumulative_importance']*100:5.2f}%)")

        # Analyze feature sources
        model_features = importance_df['model_feature'].tolist()
        raw_feature_map = analyze_feature_sources(model_features)

        # Consolidate and identify critical raw features
        raw_features_by_file, critical_raw_features, feature_importance_map = consolidate_raw_features(
            raw_feature_map, importance_df
        )

        # Save configuration
        save_raw_feature_configuration(
            raw_features_by_file,
            critical_raw_features,
            feature_importance_map,
            importance_df
        )

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Model features analyzed: {len(model_features)}")
        print(f"Total raw features needed: {sum(len(v) for v in raw_features_by_file.values())}")
        print(f"Critical raw features: {sum(len(v) for v in critical_raw_features.values())}")
        print("\nCritical raw features by file:")
        for file in sorted(critical_raw_features.keys()):
            print(f"  {file:30s} {len(critical_raw_features[file]):3d} features")
            # Show first 10
            for feat in sorted(list(critical_raw_features[file]))[:10]:
                print(f"    - {feat}")
            if len(critical_raw_features[file]) > 10:
                print(f"    ... and {len(critical_raw_features[file]) - 10} more")

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
