# ============================================================================
# CREDIT SCORING SYSTEM - STARTUP COMMANDS
# ============================================================================
# All commands to run the Credit Scoring API, Dashboard, and MLflow UI
# Run commands in PowerShell from project root directory
# ============================================================================

# ----------------------------------------------------------------------------
# STEP 1: START API SERVER (Required - Port 8000)
# ----------------------------------------------------------------------------
# Open PowerShell Terminal 1 and run:

.\.venv\Scripts\Activate.ps1
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload

# Expected output:
# ✓ Model loaded successfully from models/production_model.pkl
# INFO: Uvicorn running on http://127.0.0.1:8000

# API Documentation: http://localhost:8000/docs


# ----------------------------------------------------------------------------
# STEP 2: START STREAMLIT DASHBOARD (Port 8501)
# ----------------------------------------------------------------------------
# Open PowerShell Terminal 2 and run:

.\.venv\Scripts\Activate.ps1
streamlit run streamlit_app/app.py

# Browser will open automatically to http://localhost:8501

# LOGIN CREDENTIALS:
# Admin:   admin / admin123
# Analyst: analyst / analyst123


# ----------------------------------------------------------------------------
# STEP 3: START MLFLOW UI (Optional - Port 5000)
# ----------------------------------------------------------------------------
# Open PowerShell Terminal 3 and run:

.\.venv\Scripts\Activate.ps1
cd mlruns
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# Access MLflow UI: http://localhost:5000


# ============================================================================
# QUICK STARTUP SCRIPTS (Alternative Method)
# ============================================================================
# Use these PowerShell scripts for one-command startup:

# Terminal 1: .\start_api.ps1
# Terminal 2: .\start_streamlit.ps1
# Terminal 3: .\start_mlflow.ps1


# ============================================================================
# TROUBLESHOOTING
# ============================================================================

# Check if ports are in use:
Get-NetTCPConnection -LocalPort 8000  # API
Get-NetTCPConnection -LocalPort 8501  # Streamlit
Get-NetTCPConnection -LocalPort 5000  # MLflow

# Kill process on port:
# Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process -Force

# Initialize database if login fails:
python backend/init_db.py

# Check Python environment:
.\.venv\Scripts\python.exe --version

# Test API import:
.\.venv\Scripts\python.exe -c "from api import app; print('OK')"


# ============================================================================
# SYSTEM INFORMATION
# ============================================================================
# Python: 3.13.9 (virtual environment: .venv)
# API: FastAPI + Uvicorn
# Dashboard: Streamlit
# Database: SQLite (data/credit_scoring.db)
# Model: LightGBM (models/production_model.pkl)
# Optimal Threshold: 0.48 (cost_fn=10, cost_fp=1)
# ============================================================================
