
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
import sys

# Setup path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MLFLOW_TRACKING_URI

def log_artifacts_to_production():
    print("Logging preprocessing artifacts to Production run...")
    
    mlflow.set_tracking_uri(str(MLFLOW_TRACKING_URI))
    client = MlflowClient()
    
    model_name = "credit_scoring_production_model"
    
    # 1. Find Production Model Version
    try:
        latest_versions = client.get_latest_versions(model_name, stages=["Production"])
        if not latest_versions:
            # Fallback to None stage (latest) if no Production
            latest_versions = client.get_latest_versions(model_name)
            
        if not latest_versions:
            print("Error: No registered model found.")
            return
            
        version = latest_versions[0]
        run_id = version.run_id
        current_stage = version.current_stage
        
        print(f"Target Run ID: {run_id}")
        print(f"Model Version: {version.version} (Stage: {current_stage})")
        
    except Exception as e:
        print(f"Error finding production model: {e}")
        return

    # 2. Log Artifacts
    artifacts_to_log = [
        PROJECT_ROOT / "data" / "processed" / "scaler.joblib",
        PROJECT_ROOT / "data" / "processed" / "medians.json"
    ]
    
    with mlflow.start_run(run_id=run_id):
        for artifact_path in artifacts_to_log:
            if artifact_path.exists():
                print(f"Logging {artifact_path.name}...")
                mlflow.log_artifact(str(artifact_path), artifact_path="preprocessing")
            else:
                print(f"Warning: Artifact {artifact_path} not found")
                
    print("Artifacts logged successfully.")
    
    # Verify
    print("\nVerifying artifacts in run:")
    artifacts = client.list_artifacts(run_id, "preprocessing")
    for art in artifacts:
        print(f"  - {art.path}")

if __name__ == "__main__":
    log_artifacts_to_production()
