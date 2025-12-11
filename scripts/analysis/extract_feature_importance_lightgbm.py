"""
Extract feature importance from the production LightGBM model (189 features)
and map to raw CSV features.

This script:
1. Loads the production LightGBM model with 189 features
2. Extracts feature importance scores
3. Maps model features back to raw CSV columns
4. Identifies critical raw features (85% cumulative importance or top 50)
5. Generates configuration files for API validation
"""

import pickle
import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from collections import defaultdict

# Setup path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MLFLOW_TRACKING_URI
import mlflow

# Paths
PRODUCTION_MODEL_PATH = PROJECT_ROOT / "models" / "production_model.pkl"
FEATURE_NAMES_PATH = PROJECT_ROOT / "data" / "processed" / "feature_names.csv"
CONFIG_DIR = PROJECT_ROOT / "config"
OUTPUT_DIR = PROJECT_ROOT / "scripts" / "analysis" / "output"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_production_model():
    """Load the production LightGBM model."""
    print(f"\n{'='*80}")
    print("STEP 1: Loading Production LightGBM Model")
    print(f"{'='*80}\n")

    # Try loading from MLflow first, fallback to file
    try:
        mlflow.set_tracking_uri(str(MLFLOW_TRACKING_URI))
        model = mlflow.lightgbm.load_model("models:/credit_scoring_production_model/Production")
        print(f"[OK] Loaded model from MLflow Registry")
    except Exception as e:
        print(f"[INFO] Could not load from MLflow ({e}), loading from file...")
        with open(PRODUCTION_MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print(f"[OK] Loaded model from: {PRODUCTION_MODEL_PATH.relative_to(PROJECT_ROOT)}")

    print(f"\nModel Information:")
    print(f"  Type: {type(model).__name__}")
    print(f"  Number of features: {model.n_features_in_}")
    print(f"  Number of classes: {model.n_classes_}")

    return model

def load_feature_names():
    """Load feature names from processed data."""
    print(f"\n{'='*80}")
    print("STEP 2: Loading Feature Names")
    print(f"{'='*80}\n")

    if FEATURE_NAMES_PATH.exists():
        df = pd.read_csv(FEATURE_NAMES_PATH)
        feature_names = df['feature'].tolist()
        print(f"[OK] Loaded {len(feature_names)} feature names from: {FEATURE_NAMES_PATH.relative_to(PROJECT_ROOT)}")
    else:
        print(f"[WARNING] Feature names file not found, using model's feature_names_in_")
        if hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_.tolist()
        else:
            raise ValueError("Could not load feature names")

    return feature_names

def extract_feature_importance(model, feature_names):
    """Extract feature importance from the model."""
    print(f"\n{'='*80}")
    print("STEP 3: Extracting Feature Importance")
    print(f"{'='*80}\n")

    # Get feature importance
    importance = model.feature_importances_

    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })

    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)

    # Add cumulative importance
    importance_df['cumulative_importance'] = importance_df['importance'].cumsum()

    # Normalize cumulative importance to 0-1
    total_importance = importance_df['importance'].sum()
    importance_df['cumulative_importance'] = importance_df['cumulative_importance'] / total_importance

    print(f"[OK] Extracted importance for {len(importance_df)} features")
    print(f"\nTop 10 Most Important Features:")
    for i, row in importance_df.head(10).iterrows():
        print(f"  {i+1:2d}. {row['feature']:50s} {row['importance']:8.4f} ({row['cumulative_importance']*100:5.1f}%)")

    # Save full feature importance
    output_path = OUTPUT_DIR / "model_feature_importance_189.csv"
    importance_df.to_csv(output_path, index=False)
    print(f"\n[OK] Saved to: {output_path.relative_to(PROJECT_ROOT)}")

    return importance_df

def identify_critical_features(importance_df, threshold=0.85):
    """Identify critical features using cumulative importance threshold."""
    print(f"\n{'='*80}")
    print(f"STEP 4: Identifying Critical Features ({threshold*100}% threshold)")
    print(f"{'='*80}\n")

    # Method 1: By cumulative importance threshold
    critical_by_threshold = importance_df[importance_df['cumulative_importance'] <= threshold]

    # Method 2: Top 50 features
    critical_by_count = importance_df.head(50)

    # Use the larger set
    if len(critical_by_threshold) > len(critical_by_count):
        critical_features = critical_by_threshold
        method = f"cumulative importance <= {threshold}"
    else:
        critical_features = critical_by_count
        method = "top 50 features"

    print(f"[OK] Identified {len(critical_features)} critical features using: {method}")
    print(f"  Cumulative importance coverage: {critical_features['cumulative_importance'].iloc[-1]*100:.2f}%")

    return critical_features

def map_to_raw_features(model_feature):
    """
    Map a single model feature to its source raw CSV file(s) and column(s).

    Returns: dict with structure {file: [columns]}
    """
    sources = defaultdict(list)

    # ========================================
    # BUREAU aggregations
    # ========================================
    if any(x in model_feature for x in ['BUREAU', 'BURO']):
        # Bureau-related features
        if 'AMT_REQ_CREDIT_BUREAU' in model_feature:
            # Direct bureau features
            for suffix in ['DAY', 'MON', 'QRT', 'WEEK', 'YEAR']:
                if suffix in model_feature:
                    sources['bureau.csv'].append(f'AMT_REQ_CREDIT_BUREAU_{suffix}')
        else:
            # Aggregated bureau features - mark as requiring bureau.csv
            sources['bureau.csv'].append('*')  # Wildcard: needs bureau data

    # ========================================
    # PREVIOUS APPLICATION aggregations
    # ========================================
    elif 'PREV' in model_feature or 'PREVIOUS' in model_feature:
        sources['previous_application.csv'].append('*')

    # ========================================
    # CREDIT CARD aggregations
    # ========================================
    elif 'CC' in model_feature or 'CREDIT_CARD' in model_feature:
        sources['credit_card_balance.csv'].append('*')

    # ========================================
    # INSTALLMENTS aggregations
    # ========================================
    elif 'INSTAL' in model_feature or 'INSTALLMENT' in model_feature:
        sources['installments_payments.csv'].append('*')

    # ========================================
    # POS CASH aggregations
    # ========================================
    elif 'POS' in model_feature:
        sources['POS_CASH_balance.csv'].append('*')

    # ========================================
    # Domain/Engineered features
    # ========================================
    elif 'EXT_SOURCE_MEAN' in model_feature or 'EXT_SOURCE_WEIGHTED' in model_feature:
        sources['application.csv'].extend(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'])

    elif 'EXT_SOURCE_MIN' in model_feature:
        sources['application.csv'].extend(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'])

    elif 'EXT_SOURCE_MAX' in model_feature:
        sources['application.csv'].extend(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'])

    elif 'CREDIT_TO_GOODS_RATIO' in model_feature or 'CREDIT_GOODS_RATIO' in model_feature:
        sources['application.csv'].extend(['AMT_CREDIT', 'AMT_GOODS_PRICE'])

    elif 'CREDIT_TO_ANNUITY_RATIO' in model_feature or 'ANNUITY_CREDIT_RATIO' in model_feature:
        sources['application.csv'].extend(['AMT_CREDIT', 'AMT_ANNUITY'])

    elif 'INCOME_TO_CREDIT_RATIO' in model_feature:
        sources['application.csv'].extend(['AMT_INCOME_TOTAL', 'AMT_CREDIT'])

    elif 'DEBT_TO_INCOME_RATIO' in model_feature:
        sources['application.csv'].extend(['AMT_ANNUITY', 'AMT_INCOME_TOTAL'])

    elif 'INCOME_PER_PERSON' in model_feature or 'INCOME_PER_FAMILY' in model_feature:
        sources['application.csv'].extend(['AMT_INCOME_TOTAL', 'CNT_FAM_MEMBERS'])

    elif 'ANNUITY_TO_INCOME_RATIO' in model_feature:
        sources['application.csv'].extend(['AMT_ANNUITY', 'AMT_INCOME_TOTAL'])

    elif 'TOTAL_DOCUMENTS_PROVIDED' in model_feature:
        # Sum of FLAG_DOCUMENT_X columns
        doc_cols = [f'FLAG_DOCUMENT_{i}' for i in range(2, 22)]
        sources['application.csv'].extend(doc_cols)

    elif 'HAS_CHILDREN' in model_feature:
        sources['application.csv'].append('CNT_CHILDREN')

    # ========================================
    # One-hot encoded categorical features
    # ========================================
    elif model_feature.startswith('NAME_'):
        # Extract base column name (before the last underscore which is the category value)
        parts = model_feature.split('_')
        # Find where the actual column name ends
        for i in range(len(parts), 1, -1):
            potential_col = '_'.join(parts[:i])
            sources['application.csv'].append(potential_col)
            break

    elif model_feature.startswith('CODE_GENDER'):
        sources['application.csv'].append('CODE_GENDER')

    elif model_feature.startswith('FLAG_OWN_CAR'):
        sources['application.csv'].append('FLAG_OWN_CAR')

    elif model_feature.startswith('FLAG_OWN_REALTY'):
        sources['application.csv'].append('FLAG_OWN_REALTY')

    elif model_feature.startswith('WALLSMATERIAL_MODE'):
        sources['application.csv'].append('WALLSMATERIAL_MODE')

    elif model_feature.startswith('FONDKAPREMONT_MODE'):
        sources['application.csv'].append('FONDKAPREMONT_MODE')

    elif model_feature.startswith('HOUSETYPE_MODE'):
        sources['application.csv'].append('HOUSETYPE_MODE')

    elif model_feature.startswith('EMERGENCYSTATE_MODE'):
        sources['application.csv'].append('EMERGENCYSTATE_MODE')

    elif model_feature.startswith('WEEKDAY_APPR_PROCESS_START'):
        sources['application.csv'].append('WEEKDAY_APPR_PROCESS_START')

    elif model_feature.startswith('FLAG_DOCUMENT'):
        sources['application.csv'].append(model_feature)

    elif model_feature.startswith('FLAG_'):
        # Generic FLAG features
        sources['application.csv'].append(model_feature)

    # ========================================
    # Direct numeric features from application
    # ========================================
    elif model_feature.startswith(('AMT_', 'DAYS_', 'CNT_', 'EXT_SOURCE_', 'HOUR_', 'OBS_', 'DEF_', 'REGION_', 'REG_', 'LIVE_', 'OWN_CAR_', 'ELEVATORS_', 'FLOORSMAX_', 'FLOORSMIN_', 'TOTALAREA_')):
        sources['application.csv'].append(model_feature)

    # ========================================
    # Default: assume application.csv
    # ========================================
    else:
        sources['application.csv'].append(model_feature)

    return {file: list(set(cols)) for file, cols in sources.items() if cols}

def analyze_raw_feature_sources(critical_features):
    """Analyze which raw CSV files and columns are needed for critical features."""
    print(f"\n{'='*80}")
    print("STEP 5: Mapping to Raw CSV Features")
    print(f"{'='*80}\n")

    raw_feature_map = {}
    raw_feature_importance = defaultdict(lambda: {'importance': 0.0, 'contributing_features': []})

    for _, row in critical_features.iterrows():
        model_feat = row['feature']
        importance = row['importance']

        # Map to raw features
        sources = map_to_raw_features(model_feat)
        raw_feature_map[model_feat] = sources

        # Aggregate importance by raw features
        for file, columns in sources.items():
            for col in columns:
                key = f"{file}:{col}"
                raw_feature_importance[key]['importance'] += importance
                raw_feature_importance[key]['contributing_features'].append(model_feat)

    # Convert to sorted list
    raw_importance_list = []
    for key, data in raw_feature_importance.items():
        file, col = key.split(':', 1)
        raw_importance_list.append({
            'file_column': key,
            'file': file,
            'column': col,
            'importance': float(data['importance']),
            'contributing_model_features': data['contributing_features']
        })

    raw_importance_list.sort(key=lambda x: x['importance'], reverse=True)

    print(f"[OK] Mapped {len(critical_features)} model features to raw features")
    print(f"  Identified {len(raw_importance_list)} unique raw features")
    print(f"\nTop 15 Most Important Raw Features:")
    for i, feat in enumerate(raw_importance_list[:15], 1):
        print(f"  {i:2d}. {feat['file_column']:50s} {feat['importance']:8.4f}")

    return raw_feature_map, raw_importance_list

def identify_critical_raw_features(raw_importance_list, threshold=0.85):
    """Identify critical raw features that must be present."""
    print(f"\n{'='*80}")
    print("STEP 6: Identifying Critical Raw Features")
    print(f"{'='*80}\n")

    # Calculate cumulative importance
    total_importance = sum(f['importance'] for f in raw_importance_list)
    cumulative = 0.0
    critical_raw = []

    for feat in raw_importance_list:
        cumulative += feat['importance']
        feat['cumulative_importance'] = cumulative / total_importance

        if feat['cumulative_importance'] <= threshold or len(critical_raw) < 50:
            critical_raw.append(feat)

    # Group by file
    by_file = defaultdict(list)
    for feat in critical_raw:
        # Skip wildcard features for now
        if feat['column'] != '*':
            by_file[feat['file']].append(feat['column'])

    print(f"[OK] Identified {len(critical_raw)} critical raw features")
    print(f"  Coverage: {critical_raw[-1]['cumulative_importance']*100:.2f}% of importance")
    print(f"\nBreakdown by file:")
    for file, columns in sorted(by_file.items()):
        unique_cols = list(set(columns))
        print(f"  {file:30s} {len(unique_cols):3d} features")

    return critical_raw, by_file

def generate_config_files(all_raw_importance, critical_raw, critical_by_file):
    """Generate configuration JSON files for API validation."""
    print(f"\n{'='*80}")
    print("STEP 7: Generating Configuration Files")
    print(f"{'='*80}\n")

    # 1. All raw features
    all_by_file = defaultdict(list)
    for feat in all_raw_importance:
        if feat['column'] != '*':
            all_by_file[feat['file']].append(feat['column'])

    # Remove duplicates and sort
    for file in all_by_file:
        all_by_file[file] = sorted(list(set(all_by_file[file])))

    all_raw_config = dict(all_by_file)
    all_raw_config['_metadata'] = {
        'description': 'All RAW features used by the model',
        'total_features': sum(len(cols) for cols in all_by_file.values())
    }

    all_raw_path = CONFIG_DIR / "all_raw_features.json"
    with open(all_raw_path, 'w') as f:
        json.dump(all_raw_config, f, indent=2)
    print(f"[OK] Saved all raw features: {all_raw_path.relative_to(PROJECT_ROOT)}")

    # 2. Critical raw features
    critical_config = {}
    for file, columns in critical_by_file.items():
        critical_config[file] = sorted(list(set(columns)))

    critical_config['_metadata'] = {
        'description': 'Critical RAW features that MUST be present in uploaded files',
        'total_critical_features': sum(len(cols) for cols in critical_by_file.values()),
        'threshold': 0.85
    }

    critical_path = CONFIG_DIR / "critical_raw_features.json"
    with open(critical_path, 'w') as f:
        json.dump(critical_config, f, indent=2)
    print(f"[OK] Saved critical raw features: {critical_path.relative_to(PROJECT_ROOT)}")

    # 3. Raw feature importance (detailed mapping)
    importance_path = CONFIG_DIR / "raw_feature_importance.json"
    with open(importance_path, 'w') as f:
        json.dump(all_raw_importance, f, indent=2)
    print(f"[OK] Saved raw feature importance: {importance_path.relative_to(PROJECT_ROOT)}")

    return all_raw_config, critical_config

def main():
    print(f"\n{'#'*80}")
    print("# EXTRACTING FEATURE IMPORTANCE FROM LIGHTGBM MODEL (189 FEATURES)")
    print(f"{'#'*80}\n")

    try:
        # Step 1: Load model
        model = load_production_model()

        # Step 2: Load feature names
        feature_names = load_feature_names()

        # Step 3: Extract importance
        importance_df = extract_feature_importance(model, feature_names)

        # Step 4: Identify critical model features
        critical_features = identify_critical_features(importance_df, threshold=0.85)

        # Step 5: Map to raw features
        raw_feature_map, raw_importance_list = analyze_raw_feature_sources(critical_features)

        # Step 6: Identify critical raw features
        critical_raw, critical_by_file = identify_critical_raw_features(raw_importance_list, threshold=0.85)

        # Step 7: Generate config files
        all_raw_config, critical_config = generate_config_files(raw_importance_list, critical_raw, critical_by_file)

        print(f"\n{'='*80}")
        print("SUCCESS: Feature Importance Analysis Complete")
        print(f"{'='*80}\n")
        print("Generated configuration files:")
        print(f"  - {CONFIG_DIR.relative_to(PROJECT_ROOT)}/all_raw_features.json")
        print(f"  - {CONFIG_DIR.relative_to(PROJECT_ROOT)}/critical_raw_features.json")
        print(f"  - {CONFIG_DIR.relative_to(PROJECT_ROOT)}/raw_feature_importance.json")
        print(f"\nFiles Required for Predictions:")
        for file in sorted(critical_by_file.keys()):
            print(f"  - {file}")
        print(f"\nReady to proceed with Phase 2: API Reconstruction")

    except Exception as e:
        print(f"\n{'='*80}")
        print("ERROR: Feature Importance Analysis Failed")
        print(f"{'='*80}\n")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
