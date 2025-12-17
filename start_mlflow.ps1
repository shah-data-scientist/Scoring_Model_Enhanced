# Start MLflow UI
# Run this in PowerShell: .\start_mlflow.ps1

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Starting MLflow UI" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check if mlruns/mlflow.db exists
if (-Not (Test-Path "mlruns/mlflow.db")) {
    Write-Host "ERROR: mlruns/mlflow.db not found!" -ForegroundColor Red
    Write-Host "MLflow database must exist to launch UI" -ForegroundColor Yellow
    exit 1
}

# Activate virtual environment and start MLflow
# & .\.venv\Scripts\Activate.ps1
# Use poetry run to ensure we are in the environment
Write-Host "Starting MLflow UI on http://127.0.0.1:5000" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop MLflow UI" -ForegroundColor Yellow
Write-Host ""

poetry run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
