"""
Register the correct LightGBM production model in MLflow.

This script:
1. Loads the LightGBM model with 189 features
2. Creates a new MLflow run with the model
3. Registers it in MLflow Model Registry as production model
"""

import pickle
import shutil
from pathlib import Path
import sys
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

# Setup path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MLFLOW_TRACKING_URI

# Model and run information
SOURCE_MODEL_PATH = PROJECT_ROOT / "mlruns" / "1" / "models" / "m-31445abb63de460ca9684e54d447fc7c" / "artifacts" / "model.pkl"
PRODUCTION_MODEL_NAME = "credit_scoring_production_model"
EXPERIMENT_NAME = "credit_scoring_production"

def create_production_run():
    """Create a new MLflow run and register the production model."""
    print(f"\n{'='*80}")
    print("STEP 1: Creating Production MLflow Run")
    print(f"{'='*80}\n")

    mlflow.set_tracking_uri(str(MLFLOW_TRACKING_URI))
    client = MlflowClient()

    # Create or get experiment
    try:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
        print(f"[OK] Created new experiment: {EXPERIMENT_NAME} (ID: {experiment_id})")
    except:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        experiment_id = experiment.experiment_id
        print(f"[OK] Using existing experiment: {EXPERIMENT_NAME} (ID: {experiment_id})")

    # Load the model to verify it works
    with open(SOURCE_MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    print(f"\nModel Information:")
    print(f"  Type: {type(model).__name__}")
    print(f"  Features: {model.n_features_in_}")
    print(f"  Source: {SOURCE_MODEL_PATH.relative_to(PROJECT_ROOT)}")

    # Create a new run
    with mlflow.start_run(experiment_id=experiment_id, run_name="production_lightgbm_189features") as run:
        run_id = run.info.run_id

        # Log model parameters
        mlflow.log_param("model_type", "LightGBM")
        mlflow.log_param("n_features", model.n_features_in_)
        mlflow.log_param("source", "m-31445abb63de460ca9684e54d447fc7c")

        # Log tags
        mlflow.set_tag("stage", "production")
        mlflow.set_tag("description", "Production LightGBM model with 189 features")

        # Load feature names
        feature_names_path = PROJECT_ROOT / "data" / "processed" / "feature_names.csv"
        if feature_names_path.exists():
            feature_names_df = pd.read_csv(feature_names_path)
            feature_names = feature_names_df['feature'].tolist()
            mlflow.log_param("n_features_actual", len(feature_names))

        # Log the model with MLflow
        model_info = mlflow.lightgbm.log_model(
            lgb_model=model,
            artifact_path="model",
            registered_model_name=PRODUCTION_MODEL_NAME
        )

        print(f"\n[OK] Model logged to MLflow")
        print(f"  Run ID: {run_id}")
        print(f"  Artifact path: model")
        print(f"  Model URI: {model_info.model_uri}")

    return run_id, model_info.model_uri

def transition_to_production():
    """Transition the registered model to Production stage."""
    print(f"\n{'='*80}")
    print("STEP 2: Transitioning Model to Production Stage")
    print(f"{'='*80}\n")

    mlflow.set_tracking_uri(str(MLFLOW_TRACKING_URI))
    client = MlflowClient()

    # Get the latest version
    latest_versions = client.get_latest_versions(PRODUCTION_MODEL_NAME)
    if latest_versions:
        latest_version = latest_versions[0]
        version_number = latest_version.version

        # Transition to Production stage
        client.transition_model_version_stage(
            name=PRODUCTION_MODEL_NAME,
            version=version_number,
            stage="Production",
            archive_existing_versions=True
        )

        print(f"[OK] Model '{PRODUCTION_MODEL_NAME}' version {version_number}")
        print(f"  Transitioned to Production stage")
        print(f"  Previous versions archived")

        return version_number
    else:
        print("[WARNING] Could not find registered model version")
        return None

def verify_model_access(run_id, version_number):
    """Verify the model can be accessed from different locations."""
    print(f"\n{'='*80}")
    print("STEP 3: Verifying Model Access")
    print(f"{'='*80}\n")

    mlflow.set_tracking_uri(str(MLFLOW_TRACKING_URI))

    # Method 1: Load from registry (Production stage)
    try:
        model_uri = f"models:/{PRODUCTION_MODEL_NAME}/Production"
        model = mlflow.lightgbm.load_model(model_uri)
        print(f"[OK] Model accessible via registry (Production): {model_uri}")
        print(f"  Type: {type(model).__name__}")
        print(f"  Features: {model.n_features_in_}")
    except Exception as e:
        print(f"[ERROR] Error loading from registry (Production): {e}")

    # Method 2: Load from registry (by version)
    if version_number:
        try:
            model_uri = f"models:/{PRODUCTION_MODEL_NAME}/{version_number}"
            model = mlflow.lightgbm.load_model(model_uri)
            print(f"[OK] Model accessible via registry (version {version_number}): {model_uri}")
            print(f"  Type: {type(model).__name__}")
            print(f"  Features: {model.n_features_in_}")
        except Exception as e:
            print(f"[ERROR] Error loading from registry (version): {e}")

    # Method 3: Load from run
    try:
        run_uri = f"runs:/{run_id}/model"
        model = mlflow.lightgbm.load_model(run_uri)
        print(f"[OK] Model accessible via run: {run_uri}")
        print(f"  Type: {type(model).__name__}")
        print(f"  Features: {model.n_features_in_}")
    except Exception as e:
        print(f"[ERROR] Error loading from run: {e}")

    # Method 4: Direct file access
    try:
        with open(SOURCE_MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print(f"[OK] Model accessible via direct file")
        print(f"  Path: {SOURCE_MODEL_PATH.relative_to(PROJECT_ROOT)}")
        print(f"  Type: {type(model).__name__}")
        print(f"  Features: {model.n_features_in_}")
    except Exception as e:
        print(f"[ERROR] Error loading from file: {e}")

def update_production_model_reference():
    """Update the production_model.pkl to point to the correct model."""
    print(f"\n{'='*80}")
    print("STEP 4: Updating Production Model Reference")
    print(f"{'='*80}\n")

    production_model_path = PROJECT_ROOT / "models" / "production_model.pkl"

    # Copy the correct LightGBM model to production_model.pkl
    shutil.copy2(SOURCE_MODEL_PATH, production_model_path)

    print(f"[OK] Updated production model at: {production_model_path}")

    # Verify
    with open(production_model_path, 'rb') as f:
        model = pickle.load(f)

    print(f"  Type: {type(model).__name__}")
    print(f"  Features: {model.n_features_in_}")

def main():
    print(f"\n{'#'*80}")
    print("# REGISTERING LIGHTGBM PRODUCTION MODEL")
    print(f"{'#'*80}\n")

    try:
        # Step 1: Create production run and log model
        run_id, model_uri = create_production_run()

        # Step 2: Transition to Production stage
        version_number = transition_to_production()

        # Step 3: Verify access
        verify_model_access(run_id, version_number)

        # Step 4: Update production_model.pkl
        update_production_model_reference()

        print(f"\n{'='*80}")
        print("SUCCESS: Model Registration Complete")
        print(f"{'='*80}\n")
        print("The LightGBM model (189 features) is now accessible via:")
        print(f"  1. MLflow Registry: models:/{PRODUCTION_MODEL_NAME}/Production")
        print(f"  2. MLflow Registry: models:/{PRODUCTION_MODEL_NAME}/{version_number}")
        print(f"  3. MLflow Run: runs:/{run_id}/model")
        print(f"  4. Direct file: models/production_model.pkl")
        print(f"\nNext Steps:")
        print(f"  - Re-run feature importance analysis with 189-feature model")
        print(f"  - Update configuration files with correct raw feature mappings")
        print(f"  - View model in MLflow UI: http://localhost:5000")

    except Exception as e:
        print(f"\n{'='*80}")
        print("ERROR: Model Registration Failed")
        print(f"{'='*80}\n")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
