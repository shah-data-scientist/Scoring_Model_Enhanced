# Start Streamlit Dashboard
# Run this in PowerShell: .\start_streamlit.ps1

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Starting Credit Scoring Dashboard" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment and start Streamlit
& .\.venv\Scripts\Activate.ps1
Write-Host "Starting Dashboard..." -ForegroundColor Green
Write-Host ""
Write-Host "Login credentials:" -ForegroundColor Yellow
Write-Host "  Admin:   admin / admin123" -ForegroundColor White
Write-Host "  Analyst: analyst / analyst123" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop the dashboard" -ForegroundColor Yellow
Write-Host ""

streamlit run streamlit_app/app.py
