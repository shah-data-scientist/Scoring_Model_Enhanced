# MLflow & API Commands

## Quick Start

### Start All Services
```bash
# Terminal 1: MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# Terminal 2: Credit Scoring API
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload

# Terminal 3: Streamlit Dashboard (optional)
streamlit run streamlit_app/app.py
```

## MLflow UI Access
```
http://localhost:5000
```

### Navigate To
1. **Experiments** → `credit_scoring_final_delivery`
2. **Run** → `production_lightgbm_189features_final`
3. **View** → Parameters, Metrics, Artifacts

## API Endpoints

### Health Checks
```bash
# General health
curl http://localhost:8000/health

# MLflow connection status
curl http://localhost:8000/health/mlflow

# Database status
curl http://localhost:8000/health/database
```

### API Documentation
```
http://localhost:8000/docs           # Interactive Swagger UI
http://localhost:8000/redoc          # ReDoc documentation
```

### Single Predictions
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.1, 0.2, 0.3, ... 189 features total ...],
    "client_id": "CLIENT_123"
  }'
```

### Batch Predictions
```bash
# Upload CSV file for predictions
curl -X POST http://localhost:8000/batch/predict \
  -F "file=@input_data.csv" \
  -F "threshold=0.48"
```

### Model Metrics
```bash
# Get metrics at specific threshold
curl http://localhost:8000/metrics/threshold/0.48

# Get threshold analysis
curl http://localhost:8000/metrics/thresholds

# Get all precomputed metrics
curl http://localhost:8000/metrics/all
```

## MLflow Commands

### View Experiments
```bash
# List all experiments
mlflow experiments list

# Get specific experiment details
mlflow experiments describe --experiment-name credit_scoring_final_delivery
```

### View Runs
```bash
# List all runs in experiment
mlflow runs list --experiment-name credit_scoring_final_delivery

# Get run details
mlflow runs describe <run_id>
```

### Download Artifacts
```bash
# Download model from run
mlflow artifacts download \
  --run-id <run_id> \
  --artifact-path model \
  --dst-path ./downloaded_model
```

## Python API Usage

### Load Model from MLflow
```python
from api.mlflow_loader import load_model_from_mlflow
from pathlib import Path

# Load with fallback
model, metadata = load_model_from_mlflow(
    experiment_name="credit_scoring_final_delivery",
    fallback_path=Path("models/production_model.pkl")
)

# Check metadata
print(f"Model type: {metadata['type']}")
print(f"Optimal threshold: {metadata['parameters']['optimal_threshold']}")
print(f"Metrics: {metadata['metrics']}")
```

### Get Run Information
```python
from api.mlflow_loader import get_mlflow_run_info

run_info = get_mlflow_run_info(
    experiment_name="credit_scoring_final_delivery"
)

print(run_info['run_name'])
print(run_info['parameters'])
print(run_info['metrics'])
```

### List Experiments
```python
from api.mlflow_loader import list_mlflow_experiments

experiments = list_mlflow_experiments()
for exp in experiments:
    print(f"{exp['id']}: {exp['name']} ({exp['lifecycle_stage']})")
```

## Create New Production Run

### Using Python Script
```python
import mlflow
import mlflow.lightgbm
import pickle
from pathlib import Path

# Set tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("credit_scoring_final_delivery")

# Start run
with mlflow.start_run(run_name="production_v2") as run:
    # Load and log model
    with open("models/production_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    mlflow.lightgbm.log_model(model, "model")
    
    # Log parameters
    mlflow.log_param("optimal_threshold", 0.48)
    mlflow.log_param("n_features", 189)
    
    # Log metrics
    mlflow.log_metric("accuracy", 0.7459)
    mlflow.log_metric("roc_auc", 0.7839)
    
    # Log tags
    mlflow.set_tag("stage", "production")
    mlflow.set_tag("status", "deployed")
    
    print(f"Created run: {run.info.run_id}")
```

## Database Commands

### Check API Database
```bash
# View database file
ls -lh data/credit_scoring.db

# Connect with sqlite3
sqlite3 data/credit_scoring.db

# Query users
sqlite3 data/credit_scoring.db "SELECT * FROM users;"

# Query recent predictions
sqlite3 data/credit_scoring.db "SELECT * FROM predictions ORDER BY created_at DESC LIMIT 10;"
```

## Troubleshooting

### API Won't Start
```bash
# Check if port 8000 is in use
netstat -ano | findstr :8000

# Kill process using port
taskkill /PID <PID> /F

# Restart API
python -m uvicorn api.app:app --port 8000
```

### MLflow UI Won't Load
```bash
# Check if port 5000 is in use
netstat -ano | findstr :5000

# Kill process using port
taskkill /PID <PID> /F

# Restart MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

### Model Loading Fails
```bash
# Test API import
python -c "from api import app; print('OK')"

# Check model file exists
ls -lh models/production_model.pkl

# Test MLflow connection
python -c "import mlflow; mlflow.set_tracking_uri('sqlite:///mlflow.db'); print(mlflow.search_experiments())"
```

### Database Errors
```bash
# Check database file
ls -lh data/credit_scoring.db

# Reset database (WARNING: loses data)
rm data/credit_scoring.db
python backend/init_db.py
```

## Performance Tuning

### Cache Metrics
```bash
# Metrics are cached at startup
# To refresh cache, restart API
python -m uvicorn api.app:app --reload
```

### Batch Predictions
```bash
# Optimal batch size depends on memory
# Try: 1000, 5000, 10000 rows per request
curl -X POST http://localhost:8000/batch/predict \
  -F "file=@large_file.csv" \
  -F "batch_size=5000"
```

### Connection Pooling
```bash
# API automatically handles connection pooling
# For high concurrency, increase workers:
python -m uvicorn api.app:app \
  --workers 4 \
  --host 127.0.0.1 \
  --port 8000
```

## Monitoring

### API Logs
```bash
# Logs printed to console by default
# For file logging, modify app.py logging configuration
python -m uvicorn api.app:app --log-level info
```

### MLflow Logs
```bash
# MLflow logs to stderr
# View in terminal where mlflow ui was started
```

### Database Logs
```bash
# Check API request logs table
sqlite3 data/credit_scoring.db "SELECT timestamp, method, endpoint, status FROM api_request_logs ORDER BY timestamp DESC LIMIT 20;"
```

## Useful Links

- **API Docs:** http://localhost:8000/docs
- **MLflow UI:** http://localhost:5000
- **Health Check:** http://localhost:8000/health
- **MLflow Health:** http://localhost:8000/health/mlflow

## Key Files

```
Models:
  - models/production_model.pkl (production LightGBM model)

Predictions:
  - results/static_model_predictions.parquet (307,511 test predictions)

Configuration:
  - config/model_feature_importance.csv
  - config/critical_features.json
  - config/all_features.json

Database:
  - data/credit_scoring.db (SQLite auth & batch tracking)
  - mlruns/mlflow.db (MLflow tracking)

API Code:
  - api/app.py (main API)
  - api/mlflow_loader.py (MLflow integration)
  - api/batch_predictions.py (batch processing)
  - api/metrics.py (performance metrics)
```

---

**Last Updated:** 2025-12-13
**API Version:** 2.0.0
**Model:** LightGBM with 189 features
**Optimal Threshold:** 0.48
