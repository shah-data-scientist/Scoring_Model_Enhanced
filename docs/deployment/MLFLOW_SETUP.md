# MLflow Setup

## Configuration

MLflow is used for model versioning and loading in production.

## Local MLflow Server

### Start Server
```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000
```

### Access UI
http://localhost:5000

## Model Registration

### 1. Train Model
See notebooks in `notebooks/` directory.

### 2. Log Model
```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("credit_scoring")

with mlflow.start_run():
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
```

### 3. Register Model
In MLflow UI:
1. Select run
2. Click "Register Model"
3. Name: `credit_scoring_model`
4. Version: Auto-incremented

### 4. Promote to Production
1. Navigate to Models
2. Select `credit_scoring_model`
3. Select version
4. Stage → Production

## API Integration

API loads model from MLflow on startup:

```python
model = mlflow.pyfunc.load_model(
    model_uri="models:/credit_scoring_model/Production"
)
```

## Environment Variables

Set in `docker-compose.yml`:
```yaml
MLFLOW_TRACKING_URI: http://mlflow:5000
```

## Troubleshooting

### Model Not Found
Verify model is registered and in Production stage:
```bash
docker logs api | grep mlflow
```

### Connection Refused
Check MLflow server is running:
```bash
curl http://localhost:5000/health
```
