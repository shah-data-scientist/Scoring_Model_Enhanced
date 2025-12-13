# Start FastAPI Server
# Run this in PowerShell: .\start_api.ps1

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Starting Credit Scoring API Server" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment and start API
& .\.venv\Scripts\Activate.ps1
Write-Host "Starting API on http://localhost:8000" -ForegroundColor Green
Write-Host "API Docs: http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

python -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
