
import sys
import time

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()

log("Starting debug_imports.py")

log("Importing json, datetime, pathlib, numpy, pandas...")
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
log("Basic imports done.")

log("Importing fastapi...")
from fastapi import FastAPI
log("FastAPI imported.")

log("Importing backend.database...")
try:
    from backend.database import engine, get_db_info
    log("backend.database imported.")
except Exception as e:
    log(f"Failed to import backend.database: {e}")

log("Importing backend.models...")
try:
    from backend.models import Base
    log("backend.models imported.")
except Exception as e:
    log(f"Failed to import backend.models: {e}")

log("Importing api.batch_predictions...")
try:
    from api.batch_predictions import router as batch_router
    log("api.batch_predictions imported.")
except Exception as e:
    log(f"Failed to import api.batch_predictions: {e}")

log("Importing api.drift_api...")
try:
    from api.drift_api import router as drift_router
    log("api.drift_api imported.")
except Exception as e:
    log(f"Failed to import api.drift_api: {e}")

log("Importing api.metrics...")
try:
    from api.metrics import router as metrics_router, precompute_all_metrics
    log("api.metrics imported.")
except Exception as e:
    log(f"Failed to import api.metrics: {e}")

log("Importing api.file_validation...")
try:
    from api.file_validation import validate_input_data
    log("api.file_validation imported.")
except Exception as e:
    log(f"Failed to import api.file_validation: {e}")

log("Importing api.mlflow_loader...")
try:
    from api.mlflow_loader import load_model_from_mlflow
    log("api.mlflow_loader imported.")
except Exception as e:
    log(f"Failed to import api.mlflow_loader: {e}")

log("Importing api.app...")
try:
    import api.app
    log("api.app imported.")
except Exception as e:
    log(f"Failed to import api.app: {e}")

log("Done.")
