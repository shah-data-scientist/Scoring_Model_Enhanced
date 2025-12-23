# Quick Start Guide

## ✅ All Issues Fixed!

The codebase had **syntax errors** in the commented-out API endpoints. These have been fixed.

---

## � All Commands in One File

**See [START_COMMANDS.ps1](START_COMMANDS.ps1) for complete reference**

---

## 🚀 How to Start Everything

### Option 1: Simple Scripts (Recommended)

Open 2 PowerShell terminals in this folder:

**Terminal 1 - API Server:**
```powershell
.\start_api.ps1
```
Wait until you see: `✓ Model loaded successfully`

**Terminal 2 - Streamlit Dashboard:**
```powershell
.\start_streamlit.ps1
```
Your browser will open automatically.

### Option 2: Manual Commands

See [START_COMMANDS.ps1](START_COMMANDS.ps1) for copy-paste commands.

---

## 🔐 Login Credentials

- **Admin:** `admin` / `admin123`
- **Analyst:** `analyst` / `analyst123`

---

## 📊 What You Can Do

1. **Model Performance Tab** - View confusion matrix, metrics, optimal threshold (0.48)
2. **Batch Predictions Tab** - Upload CSVs and get predictions
3. **Monitoring Tab** (Admin only) - View system health and batch history

---

## 🔧 Optional: MLflow UI

To view experiment tracking (optional):

**Terminal 3:**
```powershell
.\start_mlflow.ps1
```
Then visit: http://localhost:5000

---

## ❓ Troubleshooting

### API won't start?
- Check if port 8000 is free: `Get-NetTCPConnection -LocalPort 8000`
- Kill process if needed: `Stop-Process -Id <PID> -Force`

### Streamlit won't start?
- Make sure API is running first
- Check if port 8501 is free

### Login fails?
- Database should be at: `data/credit_scoring.db`
- If missing, run: `python backend/init_db.py`

---

## 📝 Key Changes Made

1. ✅ Fixed syntax errors in `api/batch_predictions.py` and `api/metrics.py`
2. ✅ Created easy startup scripts (`.ps1` files)
3. ✅ Removed MLflow dependencies from API (faster startup)
4. ✅ Verified database and users exist
5. ✅ Optimal threshold calculation fixed (now shows 0.48)

---

## 🎯 Everything Works Now!

Just run the two scripts and you're good to go! 🚀
