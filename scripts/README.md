# Project Scripts

This directory contains scripts organized by purpose. **Production scripts** are at the root and in organized subdirectories. **Development/testing scripts** are in the `dev/` folder.

---

## 🚀 Production Scripts (Root Level)

### Core Scripts
- **`convert_to_onnx.py`**: Convert trained LightGBM model to ONNX format for optimized inference
- **`generate_drift_data.py`**: Generate synthetic production data for drift detection testing
- **`mlflow_verify.py`**: Verify MLflow setup and model registry
- **`precompute_features.py`**: Precompute features for production data

---

## 📂 Organized Subdirectories

### `pipeline/`
**ML Pipeline Workflow** - Run these scripts in order to reproduce model training:

1. **`select_best_model.py`**: Compare baseline models (Dummy, LR, RF, XGBoost, LightGBM)
2. **`run_feature_experiments.py`**: Test feature engineering strategies (Domain, Polynomial) + sampling strategies
3. **`optimize_model.py`**: Hyperparameter tuning with Optuna (maximize F3.2 score)
4. **`apply_best_model.py`**: Final model training and artifact generation

### `deployment/`
**Deployment & Startup Scripts**

- **`start_all.py`**: Start all services (API + MLflow + Streamlit)
- **`start_api.ps1`/`.py`**: Start FastAPI server
- **`start_mlflow.ps1`**: Start MLflow UI
- **`start_streamlit.ps1`**: Start Streamlit dashboard
- **`START_COMMANDS.ps1`**: PowerShell commands reference

### `monitoring/`
**Production Monitoring**

- **`dashboard.py`**: Real-time monitoring dashboard
- **`detect_drift.py`**: Run drift detection on production data
- **`profile_performance.py`**: Performance profiling and benchmarking

### `setup/`
**Environment Setup**

- **`create_processed_data.py`**: Generate `data/processed/` from raw CSVs
- **`create_notebooks.py`**: Generate Jupyter notebooks for analysis
- **`create_all_notebooks.py`**: Regenerate all notebook files

### `utils/`
**MLflow & Utility Scripts**

- **`check_mlflow_status.py`**: Verify MLflow server status
- **`cleanup_mlflow_runs.py`**: Clean up old MLflow experiments
- **`migrate_mlruns.py`**: Migrate MLflow data between backends
- **`start_mlflow_ui.py`**: Launch MLflow UI programmatically

### `analysis/`
**Exploratory Analysis**

- **`analyze_overfitting.py`**: Investigate train/validation performance gaps
- **`investigate_smote.py`**: Analyze SMOTE resampling effects

### `legacy/`
**Deprecated Scripts** - Old or superseded code (kept for reference)

---

## 🛠️ Development Scripts (`dev/`)

**All development, testing, and debugging scripts have been moved to `dev/` to keep the production codebase clean.**

Includes:
- `test_*.py` - Testing scripts
- `check_*.py` - Validation scripts
- `debug_*.py` - Debugging utilities
- `diagnose_*.py` - Diagnostic tools
- `verify_*.py` - Verification scripts
- `anonymize_*.py` - Data anonymization
- `benchmark_*.py` - Performance benchmarks
- `compare_*.py` - Comparison utilities
- `create_test*.py` - Test data generation
- `inspect_*.py` - Model inspection tools
- And many more...

**Usage**: Only use dev scripts during development. Production deployments should not depend on these.

---

## Quick Start

### Run Complete ML Pipeline
```bash
# 1. Prepare data
poetry run python scripts/setup/create_processed_data.py

# 2. Model selection
poetry run python scripts/pipeline/select_best_model.py

# 3. Feature experiments
poetry run python scripts/pipeline/run_feature_experiments.py

# 4. Hyperparameter optimization
poetry run python scripts/pipeline/optimize_model.py

# 5. Final model
poetry run python scripts/pipeline/apply_best_model.py
```

### Start Production Services
```bash
# Option 1: Start all at once
poetry run python scripts/deployment/start_all.py

# Option 2: Start individually
poetry run python scripts/deployment/start_mlflow.ps1
poetry run python scripts/deployment/start_api.ps1
poetry run python scripts/deployment/start_streamlit.ps1
```

### Run Monitoring
```bash
# Detect drift
poetry run python scripts/monitoring/detect_drift.py

# Launch dashboard
poetry run python scripts/monitoring/dashboard.py
```

---

## Notes

- **Production scripts** (root + subdirectories): Safe to use in production
- **Development scripts** (`dev/`): For development/testing only
- All scripts assume you're in the project root directory
- Use Poetry to manage dependencies: `poetry install`
