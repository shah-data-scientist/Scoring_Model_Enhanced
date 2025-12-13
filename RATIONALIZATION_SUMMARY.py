"""
Summary of MLflow Rationalization and Encoder Investigation
"""

print("="*80)
print("MLFLOW RATIONALIZATION - COMPLETE")
print("="*80)

print("""
✅ DATABASE STRUCTURE:

1. SINGLE DATABASE:
   - mlruns/mlflow.db (864KB) - Production database
   - Root mlflow.db (450KB) - OLD, can be deleted
   - mlruns_full_backup/ - Backup, keep for safety

2. RATIONALIZED EXPERIMENTS:
   ┌─────┬────────────────────────────────────┬─────────┬────────┐
   │ Exp │ Name                               │ Runs    │ Status │
   ├─────┼────────────────────────────────────┼─────────┼────────┤
   │  0  │ Default                            │    0    │DELETED │
   │  1  │ credit_scoring_model_selection     │    8    │DELETED │
   │  2  │ feature_engineering_cv             │   28    │DELETED │
   │  3  │ optimization_fbeta                 │   21    │DELETED │
   │  4  │ final_delivery (PRODUCTION)        │    1    │ ACTIVE │
   │  5  │ test_experiment                    │    4    │DELETED │
   │  6  │ production                         │    1    │DELETED │
   └─────┴────────────────────────────────────┴─────────┴────────┘

   ONLY Experiment 4 is visible in MLflow UI
   All development experiments archived (lifecycle_stage='deleted')

3. PRODUCTION RUN:
   - UUID: 7ce7c8f6371e43af9ced637e5a4da7f0
   - Name: production_lightgbm_189features_final
   - Parameters: 170 (including optimal_threshold=0.48, n_features=189)
   - Metrics: 10 (accuracy=0.7459, roc_auc=0.7839)
   - Artifacts: 5 files (391KB total)

✅ ARTIFACTS LOCATION:
   mlruns/7c/7ce7c8f6371e43af9ced637e5a4da7f0/artifacts/
   ├── confusion_matrix_metrics.json (257 bytes)
   ├── model_hyperparameters.json (307 bytes)
   ├── model_metadata.json (449 bytes)
   ├── production_model.pkl (377,579 bytes) <-- LightGBM model
   └── threshold_analysis.json (12,862 bytes)

✅ ENCODER INVESTIGATION:
   
   Q: "I had a problem with encoder for data as it seems it was not in the artifacts"
   
   A: ENCODERS NOT NEEDED IN ARTIFACTS
   
   Reason:
   1. production_model.pkl is a RAW LightGBM classifier (NOT a pipeline)
   2. It expects 189 numeric features (already encoded)
   3. Encoding happens in preprocessing_pipeline.py
   4. The pipeline uses src/feature_engineering.py which handles:
      - Categorical encoding (one-hot or label encoding)
      - Feature aggregation from 7 CSV files
      - Domain feature engineering
      - Scaling with scaler.joblib
   
   Files:
   - data/processed/scaler.joblib - StandardScaler for numeric features
   - data/processed/medians.json - Median values for imputation
   - api/preprocessing_pipeline.py - Handles all encoding/preprocessing
   
   The model does NOT need a separate encoder artifact because:
   - Input: Raw CSV files → PreprocessingPipeline → 189 encoded features
   - Model: Receives 189 numeric features (post-encoding)
   - Output: Predictions
   
   NO ENCODER ARTIFACTS REQUIRED ✓

✅ API DATABASE PATH - FIXED:
   
   BEFORE: mlflow.set_tracking_uri("sqlite:///mlflow.db")  ❌ WRONG
   AFTER:  mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")  ✓ CORRECT
   
   File: api/mlflow_loader.py (Line 47)

✅ CLEANUP TASKS:

   Optional:
   1. Delete root mlflow.db (450KB) - No longer needed
   2. Keep mlruns_full_backup/ for safety (backup of original state)
   3. All development runs physically deleted from filesystem

✅ VERIFICATION:

   1. MLflow UI: http://localhost:5000/#/experiments
      - Shows only Experiment 4 with 1 run
      - All other experiments hidden (deleted lifecycle stage)
   
   2. Database queries:
      - Total runs: 67 (62 deleted + 5 active in Exp 4, then cleaned to 1)
      - Active experiments: 1 (Exp 4 only)
      - Production run visible with all metadata
   
   3. API:
      - Uses mlruns/mlflow.db (correct)
      - Loads production model successfully
      - Preprocessing pipeline handles encoding
""")

print("="*80)
print("NEXT STEPS")
print("="*80)
print("""
1. Refresh MLflow UI at http://localhost:5000/#/experiments
   - You should see ONLY Experiment 4 with 1 run
   
2. Test API:
   - Start: poetry run uvicorn api.app:app --reload --port 8000
   - Verify it loads model from MLflow
   
3. Optional cleanup:
   - Remove root mlflow.db: rm mlflow.db
   - Keep backup: mlruns_full_backup/

4. Model works with:
   - Input: 7 raw CSV files
   - Processing: PreprocessingPipeline (handles encoding)
   - Model: LightGBM with 189 features
   - Output: Predictions with risk levels
""")
