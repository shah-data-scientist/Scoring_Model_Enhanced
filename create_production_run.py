"""
MLflow Rationalization Script
Creates a clean production run with all artifacts and metrics
"""

import mlflow
import mlflow.lightgbm
import pickle
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import shutil

# Configure MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
PRODUCTION_EXPERIMENT = "credit_scoring_final_delivery"
PRODUCTION_EXPERIMENT_ID = 4

print("=" * 80)
print("MLFLOW RATIONALIZATION - Creating Clean Production Run")
print("=" * 80)

# Set experiment
try:
    mlflow.set_experiment(PRODUCTION_EXPERIMENT)
    print(f"✓ Set experiment: {PRODUCTION_EXPERIMENT}")
except Exception as e:
    print(f"✗ Error setting experiment: {e}")
    exit(1)

# Load model
print("\n" + "=" * 80)
print("LOADING PRODUCTION MODEL")
print("=" * 80)

model_path = Path("models/production_model.pkl")
if not model_path.exists():
    print(f"✗ Model not found: {model_path}")
    exit(1)

with open(model_path, 'rb') as f:
    model = pickle.load(f)

print(f"✓ Loaded model: {type(model).__name__}")
if hasattr(model, 'n_features_'):
    print(f"  - Features: {model.n_features_}")
if hasattr(model, 'n_estimators'):
    print(f"  - N Estimators: {model.n_estimators}")

# Load predictions for metric calculation
print("\n" + "=" * 80)
print("LOADING PREDICTIONS & CALCULATING METRICS")
print("=" * 80)

pred_path = Path("results/static_model_predictions.parquet")
if not pred_path.exists():
    print(f"✗ Predictions not found: {pred_path}")
    exit(1)

df_pred = pd.read_parquet(pred_path)
print(f"✓ Loaded predictions: {len(df_pred):,} samples")

# Business costs
cost_fn = 10
cost_fp = 1
optimal_threshold = 0.48

# Calculate metrics at optimal threshold
y_true = df_pred['TARGET'].values
y_proba = df_pred['PROBABILITY'].values
y_pred = (y_proba >= optimal_threshold).astype(int)

from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix)

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_proba)
business_cost = fn * cost_fn + fp * cost_fp

print(f"✓ Calculated metrics at threshold {optimal_threshold}:")
print(f"  - Accuracy: {accuracy:.4f}")
print(f"  - Precision: {precision:.4f}")
print(f"  - Recall: {recall:.4f}")
print(f"  - F1-Score: {f1:.4f}")
print(f"  - ROC-AUC: {roc_auc:.4f}")
print(f"  - Business Cost: {business_cost:,}")
print(f"  - True Positives: {tp:,}")
print(f"  - True Negatives: {tn:,}")
print(f"  - False Positives: {fp:,}")
print(f"  - False Negatives: {fn:,}")

# Get model hyperparameters
model_params = {}
if hasattr(model, 'get_params'):
    model_params = model.get_params()
    # Filter to keep only important ones
    important_params = [
        'n_estimators', 'max_depth', 'learning_rate', 'min_child_samples',
        'num_leaves', 'colsample_bytree', 'subsample', 'reg_alpha', 'reg_lambda',
        'class_weight', 'random_state', 'verbose'
    ]
    model_params = {k: v for k, v in model_params.items() if k in important_params}

print(f"\n✓ Extracted {len(model_params)} model parameters")

# Create MLflow run
print("\n" + "=" * 80)
print("CREATING MLFLOW RUN")
print("=" * 80)

run_name = "production_lightgbm_189features_final"

with mlflow.start_run(run_name=run_name) as run:
    run_id = run.info.run_id
    print(f"✓ Started run: {run_name}")
    print(f"  Run ID: {run_id}")
    
    # Log parameters
    print(f"\nLogging parameters...")
    for param_name, param_value in model_params.items():
        mlflow.log_param(param_name, param_value)
    
    # Log model-specific parameters
    mlflow.log_param("optimal_threshold", optimal_threshold)
    mlflow.log_param("n_features", 189)
    mlflow.log_param("model_type", "LightGBM")
    mlflow.log_param("cost_fn", cost_fn)
    mlflow.log_param("cost_fp", cost_fp)
    print(f"✓ Logged {len(model_params) + 5} parameters")
    
    # Log metrics
    print(f"\nLogging metrics...")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("business_cost", business_cost)
    mlflow.log_metric("tp", tp)
    mlflow.log_metric("tn", tn)
    mlflow.log_metric("fp", fp)
    mlflow.log_metric("fn", fn)
    print(f"✓ Logged 10 metrics")
    
    # Log tags
    print(f"\nLogging tags...")
    mlflow.set_tag("stage", "production")
    mlflow.set_tag("status", "deployed")
    mlflow.set_tag("model_type", "LightGBM")
    mlflow.set_tag("features", "189")
    mlflow.set_tag("description", f"Production LightGBM classifier with 189 features. Optimal threshold: {optimal_threshold}")
    mlflow.set_tag("created_at", datetime.now().isoformat())
    mlflow.set_tag("dataset_size", f"{len(df_pred):,} samples")
    print(f"✓ Logged 7 tags")
    
    # Log model
    print(f"\nLogging model...")
    mlflow.lightgbm.log_model(model, "model", registered_model_name=None)
    print(f"✓ Logged model artifact")
    
    # Log additional artifacts
    print(f"\nLogging additional artifacts...")
    
    # Create temporary directory for artifacts
    artifacts_dir = Path("temp_artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    
    # 1. Model metadata
    metadata = {
        "model_type": "LightGBM",
        "n_features": 189,
        "n_estimators": model_params.get("n_estimators", "unknown"),
        "optimal_threshold": optimal_threshold,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "business_cost": int(business_cost),
        "cost_fn": cost_fn,
        "cost_fp": cost_fp,
        "training_samples": int(len(df_pred)),
        "created_at": datetime.now().isoformat(),
        "status": "production"
    }
    
    with open(artifacts_dir / "model_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    mlflow.log_artifact(str(artifacts_dir / "model_metadata.json"))
    print(f"  ✓ Logged model_metadata.json")
    
    # 2. Confusion matrix metrics
    cm_data = {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }
    
    with open(artifacts_dir / "confusion_matrix_metrics.json", 'w') as f:
        json.dump(cm_data, f, indent=2)
    mlflow.log_artifact(str(artifacts_dir / "confusion_matrix_metrics.json"))
    print(f"  ✓ Logged confusion_matrix_metrics.json")
    
    # 3. Threshold analysis
    threshold_analysis = []
    for threshold in np.arange(0.01, 1.0, 0.01):
        y_pred_t = (y_proba >= threshold).astype(int)
        fn_t = ((y_true == 1) & (y_pred_t == 0)).sum()
        fp_t = ((y_true == 0) & (y_pred_t == 1)).sum()
        tp_t = ((y_true == 1) & (y_pred_t == 1)).sum()
        tn_t = ((y_true == 0) & (y_pred_t == 0)).sum()
        cost_t = fn_t * cost_fn + fp_t * cost_fp
        
        threshold_analysis.append({
            "threshold": float(threshold),
            "cost": int(cost_t),
            "fn": int(fn_t),
            "fp": int(fp_t),
            "tp": int(tp_t),
            "tn": int(tn_t)
        })
    
    with open(artifacts_dir / "threshold_analysis.json", 'w') as f:
        json.dump(threshold_analysis, f, indent=2)
    mlflow.log_artifact(str(artifacts_dir / "threshold_analysis.json"))
    print(f"  ✓ Logged threshold_analysis.json (99 thresholds)")
    
    # 4. Model hyperparameters
    with open(artifacts_dir / "model_hyperparameters.json", 'w') as f:
        json.dump({k: str(v) for k, v in model_params.items()}, f, indent=2)
    mlflow.log_artifact(str(artifacts_dir / "model_hyperparameters.json"))
    print(f"  ✓ Logged model_hyperparameters.json")
    
    # 5. Copy model pickle to artifacts
    shutil.copy(model_path, artifacts_dir / "production_model.pkl")
    mlflow.log_artifact(str(artifacts_dir / "production_model.pkl"))
    print(f"  ✓ Logged production_model.pkl")
    
    # Clean up
    shutil.rmtree(artifacts_dir)
    
    print(f"\n✓ Run completed successfully!")
    print(f"  Run URI: mlflow://credit_scoring_final_delivery/{run_id}")

print("\n" + "=" * 80)
print("RATIONALIZATION COMPLETE")
print("=" * 80)
print(f"""
Next steps:
1. Start MLflow UI: mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
2. Navigate to experiment: credit_scoring_final_delivery
3. Verify run: {run_name}
4. Check artifacts are visible in UI
5. Update API to reference this run (optional)
""")
