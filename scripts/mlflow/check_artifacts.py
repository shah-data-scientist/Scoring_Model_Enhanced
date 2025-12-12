
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
import sys

# Setup path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MLFLOW_TRACKING_URI

def check_artifacts():
    run_id = "e009fa2f6e0c4792bb6bbfe981f8d4e6"
    print(f"Checking artifacts for Run ID: {run_id}")
    print(f"Tracking URI: {MLFLOW_TRACKING_URI}")
    
    mlflow.set_tracking_uri(str(MLFLOW_TRACKING_URI))
    client = MlflowClient()
    
    try:
        artifacts = client.list_artifacts(run_id)
        print("\nArtifacts found:")
        for art in artifacts:
            print(f"- {art.path} (Size: {art.file_size} bytes)")
            if art.is_dir:
                # List contents of directory
                sub_artifacts = client.list_artifacts(run_id, art.path)
                for sub in sub_artifacts:
                    print(f"  - {sub.path} (Size: {sub.file_size} bytes)")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_artifacts()
