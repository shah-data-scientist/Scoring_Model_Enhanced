# Credit Scoring Model - Production ML System

[![Tests](https://img.shields.io/badge/tests-67%2F67%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.12-blue)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)]()
[![MLflow](https://img.shields.io/badge/MLflow-3.6-orange)]()
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)]()
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-success)]()

A **production-ready MLOps system** for credit default prediction with:
- 🚀 **FastAPI REST API** with batch processing
- 🐳 **Docker containerization** for easy deployment
- ⚙️ **CI/CD pipeline** with automated testing
- 📊 **Real-time monitoring** with data drift detection
- 🔍 **Performance profiling** and optimization
- 📈 **MLflow tracking** and model registry

---

## 🎯 Quick Start

**📋 All commands:** [START_COMMANDS.ps1](START_COMMANDS.ps1)

### Launch Services (Windows PowerShell)

**Terminal 1 - API Server:**
```powershell
.\start_api.ps1
```

**Terminal 2 - Dashboard:**
```powershell
.\start_streamlit.ps1
```

**Terminal 3 - MLflow (Optional):**
```powershell
.\start_mlflow.ps1
```

This opens:
- **API Docs**: http://localhost:8000/docs
- **Dashboard**: http://localhost:8501 (Login: `admin`/`admin123`)
- **MLflow UI**: http://localhost:5000

### 🐳 Docker Support
```bash
docker-compose up --build -d
docker-compose ps
docker-compose down
```
poetry run python scripts/deployment/start_mlflow_ui.py

# Dashboard only
poetry run streamlit run scripts/deployment/dashboard.py

# API Server only
poetry run python scripts/deployment/start_api.py
```

---

## 📊 Project Overview

### Business Problem
Predict credit default risk for loan applications to minimize financial losses while maintaining customer approval rates.

### Solution
Machine learning system that:
- **Predicts** default probability for each application
- **Optimizes** decision threshold for business cost (FN=€10, FP=€1)
- **Achieves** 0.7761 ROC-AUC with domain-engineered features
- **Serves** predictions via REST API (<50ms latency)

### Key Metrics
| Metric | Value | Target |
|--------|-------|--------|
| **ROC-AUC** | 0.7761 | > 0.75 |
| **Precision** | 0.52 | > 0.50 |
| **Recall** | 0.68 | > 0.60 |
| **Business Cost** | €2.45/client | Minimize |
| **API Latency** | <50ms | <100ms |
| **Optimal Threshold** | 0.48 | 0.48 |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACES                          │
├──────────────┬──────────────┬──────────────┬────────────────┤
│  Web Apps    │  Mobile Apps │  Dashboards  │  Notebooks     │
└──────┬───────┴──────┬───────┴──────┬───────┴────────┬───────┘
       │              │              │                │
       └──────────────┴──────────────┴────────────────┘
                          │
                  ┌───────▼────────┐
                  │   REST API     │
                  │  (FastAPI)     │
                  │  Port 8000     │
                  └───────┬────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
  ┌───────▼────────┐ ┌───▼────┐ ┌────────▼────────┐
  │   ML Model     │ │MLflow  │ │   Monitoring    │
  │   (LightGBM)   │ │Registry│ │   (Drift, Perf) │
  │   189 Features │ │Port    │ │                 │
  └────────────────┘ │5000    │ └─────────────────┘
                     └────────┘
```

---

## 📁 Repository Structure

```
Scoring_Model/
├── README.md                          # This file
├── launch_services.bat/sh             # Quick service launcher
├── pyproject.toml                     # Poetry dependencies
│
├── src/                               # Production source code
│   ├── config.py                      # Configuration management
│   ├── validation.py                  # Data validation utilities
│   ├── data_preprocessing.py          # Data loading & preprocessing
│   ├── feature_engineering.py         # Feature creation
│   ├── domain_features.py             # Business domain features
│   ├── model_training.py              # Model training utilities
│   ├── evaluation.py                  # Evaluation metrics
│   └── mlflow_utils.py                # MLflow integration
│
├── api/                               # REST API
│   └── app.py                         # FastAPI application
│
├── scripts/                           # Utility scripts
│   ├── deployment/                    # Service launchers
│   ├── mlflow/                        # MLflow management
│   ├── experiments/                   # ML experiments
│   └── data/                          # Data utilities
│
├── tests/                             # Test suite (67 tests)
│   ├── test_api.py                   # API endpoint tests
│   ├── test_validation.py            # Validation tests
│   └── test_config.py                # Configuration tests
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_models.ipynb
│   └── 04_hyperparameter_optimization.ipynb
│
├── docs/                              # Documentation
│   ├── API_TESTING_GUIDE.md          # How to test API
│   ├── MODEL_MONITORING.md           # Monitoring guide
│   ├── DEPLOYMENT_GUIDE.md           # Deployment instructions
│   └── presentations/                # Presentations
│
├── data/processed/                    # Processed features
├── mlruns/                            # MLflow tracking
├── models/                            # Saved models
└── results/                           # Generated outputs
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.13+
- Poetry (dependency management)
- 8GB RAM minimum
- 10GB disk space

### Installation

```bash
# 1. Clone repository
git clone <repository-url>
cd Scoring_Model

# 2. Install dependencies
poetry install

# 3. Run tests to verify installation
poetry run pytest tests/ -v

# 4. Launch services
./launch_services.bat  # Windows
# or
./launch_services.sh   # Linux/Mac
```

---

## 🧪 Testing

```bash
# Run all tests
poetry run pytest tests/ -v

# Run with coverage
poetry run pytest tests/ --cov=src --cov=api --cov-report=html
```

**Results**: 67/67 tests passing ✅

---

## 📡 API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Single Prediction
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "features": [0.5, 0.3, ...],  # 189 features
        "client_id": "100002"
    }
)

result = response.json()
print(f"Risk: {result['risk_level']}")  # LOW, MEDIUM, HIGH, CRITICAL
print(f"Probability: {result['probability']:.4f}")
```

**Full API Guide**: [docs/API_TESTING_GUIDE.md](docs/API_TESTING_GUIDE.md)

---

## 📊 Model Details

### Best Model
- **Algorithm**: LightGBM Classifier
- **Features**: 189 (baseline + domain-engineered)
- **Cross-Validation**: 5-fold StratifiedKFold
- **ROC-AUC**: 0.7761 ± 0.0064
- **Optimal Threshold**: 0.48

### Feature Categories
1. **Baseline** (184): Original application data
2. **Domain** (5): Business logic features
   - DEBT_TO_INCOME_RATIO
   - EMPLOYMENT_YEARS
   - INCOME_PER_PERSON
   - AGE_YEARS
   - CREDIT_UTILIZATION

---

## 🔍 Monitoring & Compliance

We have implemented a comprehensive monitoring strategy that fully meets production requirements, ensuring model reliability and performance transparency.

### Strategy & Adequacy
Our monitoring architecture is clearly defined and documented, covering:
- **Data Drift**: Automated detection of feature, prediction, and target drift using statistical tests (KS, Chi-Square, PSI).
- **Performance**: Real-time tracking of ROC-AUC, precision, recall, and business metrics (cost/default rate).
- **System Health**: Latency, throughput, and error rate monitoring with automated alerting.

**Documentation**:
- [Monitoring Strategy & Architecture](docs/MODEL_MONITORING.md)
- [Drift Detection Methodology](docs/DRIFT_DETECTION.md)
- [Analysis & Results](docs/MONITORING_RESULTS.md)

### Metrics Tracked
- **Business**: Default rate stability, financial impact.
- **Model**: ROC-AUC (>0.75), Precision (>0.50), Recall (>0.60).
- **Data Quality**: Missing values, schema validation, range checks.
- **Infrastructure**: API latency (<50ms P95), error rates (<1%).

---

## 📚 Documentation

For a complete list of all documentation, see the **[Master Index](docs/INDEX.md)**.

### 🚀 Essentials
- **[Quick Start](QUICK_START.md)**: Deployment instructions in 3 steps.
- **[User Guide](docs/USER_GUIDE.md)**: Detailed instructions for using the system.
- **[Technical Guide](docs/TECHNICAL_GUIDE.md)**: Deep dive into architecture and implementation.

### 🛠️ Operations
- **[API Documentation](docs/API.md)**: Endpoint reference and testing.
- **[Monitoring Strategy](docs/MODEL_MONITORING.md)**: Production monitoring and drift detection.
- **[Deployment Setup](docs/deployment/DOCKER_SETUP.md)**: Docker and environment configuration.

### 💼 Business
- **[Business Presentation](docs/presentations/BUSINESS_PRESENTATION.md)**: ROI and impact analysis.
- **[Technical Presentation](docs/presentations/TECHNICAL_PRESENTATION.md)**: System architecture overview.

---


## 🐳 Docker Deployment

### Full Stack Deployment (Recommended)

The complete system includes:
- **FastAPI** - REST API (port 8000)
- **Streamlit** - Interactive dashboard (port 8501)
- **PostgreSQL** - Production database (port 5432)

#### 1. Setup Environment Variables
```bash
# Copy example environment file
cp .env.example .env

# Edit .env and update:
# - POSTGRES_PASSWORD (REQUIRED)
# - SECRET_KEY (REQUIRED - generate with: openssl rand -hex 32)
# - Other settings as needed
```

#### 2. Launch All Services
```bash
# Build and start all services
docker-compose up --build -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api
docker-compose logs -f streamlit
docker-compose logs -f postgres
```

#### 3. Access Services
- **API Documentation**: http://localhost:8000/docs
- **Streamlit Dashboard**: http://localhost:8501
- **Health Check**: http://localhost:8000/health

### Individual Service Deployment

#### API Only
```bash
docker build -t credit-scoring-api -f Dockerfile .
docker run -d -p 8000:8000 \
  -e DATABASE_URL=your_db_url \
  -e SECRET_KEY=your_secret_key \
  --name credit-api \
  credit-scoring-api
```

#### Streamlit Only
```bash
docker build -t credit-scoring-streamlit -f Dockerfile.streamlit .
docker run -d -p 8501:8501 \
  -e API_BASE_URL=http://api:8000 \
  --name credit-dashboard \
  credit-scoring-streamlit
```

### Docker Compose Services

The `docker-compose.yml` defines three services:

| Service | Port | Description |
|---------|------|-------------|
| `postgres` | 5432 | PostgreSQL 15 database |
| `api` | 8000 | FastAPI REST API |
| `streamlit` | 8501 | Streamlit dashboard |

**Volumes**:
- `postgres_data` - Database persistence
- `./logs` - Application logs
- `./data` - Model and feature data
- `./models` - ML model files

**Health Checks**:
- All services include health checks
- Dependencies ensure proper startup order

### Environment Variables

See [.env.example](.env.example) for all available settings. Key variables:

```bash
# Database
POSTGRES_USER=postgres
POSTGRES_PASSWORD=changeme_secure_password
POSTGRES_DB=credit_scoring_db

# Security
SECRET_KEY=your-secret-key-here-change-in-production

# API
API_PORT=8000
API_WORKERS=4

# Streamlit
STREAMLIT_PORT=8501
API_BASE_URL=http://api:8000
```

### Production Deployment

For production deployment to cloud platforms, see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md).

Supported platforms:
- **Heroku** - Container deployment
- **Google Cloud Run** - Serverless containers
- **AWS ECS** - Elastic Container Service
- **Azure Container Instances** - Managed containers

---

## ⚙️ CI/CD Pipeline

Automated workflow on push to main:
1. Run tests (pytest)
2. Build Docker image
3. Push to GitHub Container Registry

Location: `.github/workflows/ci-cd.yml`

---

## 📊 Monitoring & Logging

### Production Logs
```bash
# View monitoring dashboard
poetry run python scripts/monitoring/dashboard.py

# Check data drift
poetry run python scripts/monitoring/detect_drift.py

# Profile performance
poetry run python scripts/monitoring/profile_performance.py
```

**Logs**: `logs/predictions.jsonl` (JSON format)

### Metrics Tracked
- Predictions count & distribution
- Processing time & throughput  
- Error rates
- Data drift detection
- Performance profiling

---

**Last Updated**: December 9, 2025
**Version**: 1.0.0
**Status**: Production Ready ✅

---

## Docker Deployment

### Quick Start
```bash
docker-compose up --build -d
docker-compose logs -f api
```

### Manual Build
```bash
docker build -t credit-scoring-api .
docker run -d -p 8000:8000 credit-scoring-api
```

---

## CI/CD Pipeline

Automated workflow on push to main:
1. Run tests (pytest)
2. Build Docker image  
3. Push to GitHub Container Registry

Location: `.github/workflows/ci-cd.yml`

---

## Monitoring & Logging

### Production Logs
```bash
# View monitoring dashboard
poetry run python scripts/monitoring/dashboard.py

# Check data drift
poetry run python scripts/monitoring/detect_drift.py

# Profile performance
poetry run python scripts/monitoring/profile_performance.py
```

**Logs**: `logs/predictions.jsonl` (JSON format)

### Metrics Tracked
- Predictions count & distribution
- Processing time & throughput
- Error rates
- Data drift detection
- Performance profiling

---

