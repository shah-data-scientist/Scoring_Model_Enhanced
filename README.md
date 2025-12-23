# Credit Scoring Model - Production ML System

[![Tests](https://img.shields.io/badge/tests-67%2F67%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.12-blue)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)]()
[![MLflow](https://img.shields.io/badge/MLflow-3.6-orange)]()
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)]()
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-success)]()

A **production-ready MLOps system** for credit default prediction with real-time API, monitoring, and CI/CD automation.

---

## 🎯 Quick Start

### 1. Install Dependencies
```bash
poetry install
poetry run pytest tests/ -v  # Verify installation
```

### 2. Launch Services (Windows)

Open 3 PowerShell terminals:

**Terminal 1 - API Server:**
```powershell
.\start_api.ps1
```
Wait for: `✓ Model loaded successfully`

**Terminal 2 - Dashboard:**
```powershell
.\start_streamlit.ps1
```
Browser opens automatically at http://localhost:8501

**Terminal 3 - MLflow (Optional):**
```powershell
.\start_mlflow.ps1
```
Visit http://localhost:5000

### 3. Login & Test

**Credentials:**
- Admin: `admin` / `admin123`
- Analyst: `analyst` / `analyst123`

**Access:**
- **API Docs**: http://localhost:8000/docs
- **Dashboard**: http://localhost:8501
- **MLflow UI**: http://localhost:5000

### Troubleshooting
- **Port conflicts**: Kill processes on ports 8000, 8501, or 5000
- **Database missing**: Run `python backend/init_db.py`
- **Model not found**: Check `models/` directory exists

---

## 🐳 Docker Deployment

### Full Stack (Recommended)
```bash
# 1. Configure environment
cp .env.example .env
# Edit .env: Set POSTGRES_PASSWORD and SECRET_KEY

# 2. Launch all services
docker-compose up --build -d

# 3. Check status
docker-compose ps
docker-compose logs -f api
```

**Services:**
- API: http://localhost:8000
- Dashboard: http://localhost:8501
- Database: PostgreSQL on port 5432

### Individual Services
```bash
# API only
docker build -t credit-api .
docker run -d -p 8000:8000 credit-api

# Streamlit only
docker build -t credit-dashboard -f Dockerfile.streamlit .
docker run -d -p 8501:8501 credit-dashboard
```

---

## 📊 Project Overview

### Business Problem
Predict credit default risk to minimize financial losses while maintaining customer approval rates.

### Solution
Production ML system delivering:
- **Real-time predictions** via REST API (<50ms latency)
- **Automated monitoring** with drift detection
- **CI/CD pipeline** with 67 automated tests
- **Business optimization** (€2.45/client vs €3.62 baseline)

### Key Metrics
| Metric | Value | Target |
|--------|-------|--------|
| **ROC-AUC** | 0.7761 | > 0.75 ✅ |
| **Precision** | 0.52 | > 0.50 ✅ |
| **Recall** | 0.68 | > 0.60 ✅ |
| **API Latency (P95)** | 42ms | < 50ms ✅ |
| **Business Cost** | €2.45/client | Minimized ✅ |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│           USER INTERFACES               │
│   Web Apps │ Dashboards │ Notebooks     │
└──────────────────┬──────────────────────┘
                   │
            ┌──────▼───────┐
            │  FastAPI     │ ← REST API (Port 8000)
            │  (Async)     │
            └──────┬───────┘
                   │
       ┌───────────┼────────────┐
       │           │            │
  ┌────▼────┐ ┌───▼───┐ ┌─────▼─────┐
  │LightGBM │ │MLflow │ │Monitoring │
  │189 Feat.│ │Port   │ │Drift/Perf │
  └─────────┘ │5000   │ └───────────┘
              └───────┘
```

---

## 📁 Repository Structure

```
Scoring_Model_Enhanced/
├── README.md                   # ← You are here
├── LICENSE
│
├── api/                        # FastAPI application
│   ├── app.py                 # Main endpoints
│   ├── drift_detection.py     # Monitoring
│   └── preprocessing_pipeline.py
│
├── backend/                    # Database & auth
│   ├── database.py
│   ├── models.py
│   └── init_db.py
│
├── src/                        # ML pipeline
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── mlflow_utils.py
│
├── scripts/                    # Production scripts
│   ├── pipeline/              # ML workflow
│   ├── deployment/            # Start scripts
│   ├── monitoring/            # Drift detection
│   └── dev/                   # Dev tools (archived)
│
├── tests/                      # 67 tests, >80% coverage
│   ├── test_api.py
│   ├── test_preprocessing.py
│   └── test_drift_detection.py
│
├── docs/                       # Documentation
│   ├── README.md              # Documentation index
│   ├── API.md                 # API reference
│   ├── MODEL_MONITORING.md    # Monitoring guide
│   ├── DRIFT_DETECTION.md     # Drift detection
│   ├── SETUP.md               # Setup guide
│   ├── USER_GUIDE.md          # User manual
│   ├── presentations/         # Oral defense slides
│   ├── architecture/          # System design
│   ├── deployment/            # Deployment guides
│   └── archive/               # Historical docs
│
├── .github/workflows/          # CI/CD pipelines
│   └── test.yml               # Automated testing
│
├── Dockerfile                  # API container
├── Dockerfile.streamlit        # Dashboard container
├── docker-compose.yml          # Multi-service deployment
├── pyproject.toml              # Dependencies (Poetry)
└── .env.example                # Environment template
```

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
        "SK_ID_CURR": 100001,
        "features": [0.12, 0.45, ...],  # 189 features
    }
)

result = response.json()
print(f"Risk: {result['risk_level']}")      # LOW/MEDIUM/HIGH/CRITICAL
print(f"Probability: {result['probability']:.4f}")
print(f"Business Cost: €{result['business_cost']:.2f}")
```

### Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d @batch_requests.json
```

**Full API Reference**: [docs/API.md](docs/API.md)

---

## 🧪 Testing

```bash
# Run all tests
poetry run pytest tests/ -v

# With coverage report
poetry run pytest --cov=src --cov=api --cov-report=html

# Run specific test file
poetry run pytest tests/test_api.py -v
```

**Results**: 67/67 tests passing ✅
**Coverage**: >80% across all modules

---

## 📊 Model Details

### Best Model Configuration
- **Algorithm**: LightGBM Classifier
- **Features**: 189 (184 baseline + 5 domain-engineered)
- **Validation**: 5-fold StratifiedKFold cross-validation
- **Performance**: ROC-AUC 0.7761 ± 0.0064
- **Optimal Threshold**: 0.48 (business cost optimized)

### Domain Features (Top 5)
1. `DEBT_TO_INCOME_RATIO` - Total debt / Income
2. `EMPLOYMENT_YEARS` - Days employed / 365
3. `INCOME_PER_PERSON` - Income / Family size
4. `AGE_YEARS` - Days birth / 365
5. `CREDIT_UTILIZATION` - Credit amount / Credit limit

### Business Optimization
- **False Negative Cost**: €10 (loan default)
- **False Positive Cost**: €1 (lost opportunity)
- **Optimized For**: Minimum total business cost
- **Result**: 32% cost reduction vs baseline

---

## 🔍 Monitoring & Drift Detection

### Automated Monitoring
- **Data Drift**: Weekly KS tests on all 189 features
- **Performance**: ROC-AUC tracking on production data
- **System Health**: API latency, throughput, error rates
- **Alerting**: Email notifications when drift > 10%

### Current Status
- **Drift**: 5.8% features drifting (✅ Healthy)
- **Performance**: ROC-AUC stable at 0.776
- **Latency**: P95 = 42ms (✅ <50ms SLA)

### View Monitoring
```bash
# Drift detection
poetry run python scripts/monitoring/detect_drift.py

# Performance dashboard
poetry run streamlit run streamlit_app/Home.py
```

**Documentation**: [docs/MODEL_MONITORING.md](docs/MODEL_MONITORING.md)

---

## ⚙️ CI/CD Pipeline

Automated workflow on every push:

```yaml
1. Install dependencies (Poetry)
2. Run linting (Ruff, MyPy)
3. Run 67 tests (Pytest)
4. Check coverage (>80% required)
5. Build Docker image
6. Deploy to staging (auto)
7. Deploy to production (manual)
```

**Configuration**: [.github/workflows/test.yml](.github/workflows/test.yml)
**Duration**: ~3-4 minutes from commit to deployment

---

## 📚 Documentation

### Getting Started
- **[Setup Guide](docs/SETUP.md)** - Installation & configuration
- **[User Guide](docs/USER_GUIDE.md)** - How to use the system
- **[API Documentation](docs/API.md)** - Endpoint reference

### Operations
- **[Model Monitoring](docs/MODEL_MONITORING.md)** - Production monitoring
- **[Drift Detection](docs/DRIFT_DETECTION.md)** - Drift detection methodology
- **[Docker Setup](docs/deployment/DOCKER_SETUP.md)** - Container deployment
- **[MLflow Setup](docs/deployment/MLFLOW_SETUP.md)** - Experiment tracking

### Architecture
- **[System Design](docs/architecture/SYSTEM_DESIGN.md)** - Technical architecture
- **[Database Schema](docs/architecture/DATABASE_SCHEMA.md)** - Data model

### Presentations
- **[Business Presentation](docs/presentations/BUSINESS_PRESENTATION.md)** - Oral defense (30 min)
- **[Technical Presentation](docs/presentations/TECHNICAL_PRESENTATION.md)** - Technical deep dive

### Community
- **[Contributing Guide](docs/CONTRIBUTING.md)** - How to contribute
- **[Code of Conduct](docs/CODE_OF_CONDUCT.md)** - Community guidelines

**Full Documentation Index**: [docs/README.md](docs/README.md)

---

## 🚀 Production Deployment

### Cloud Platforms Supported
- **Heroku** - Container deployment
- **Google Cloud Run** - Serverless containers
- **AWS ECS/Fargate** - Elastic Container Service
- **Azure Container Instances** - Managed containers

### Environment Variables
See [.env.example](.env.example) for all configuration options.

**Critical Settings**:
```bash
POSTGRES_PASSWORD=change_me_in_production
SECRET_KEY=generate_with_openssl_rand_hex_32
DATABASE_URL=postgresql://user:pass@host:5432/db
```

---

## 📈 Performance Optimization

### Optimizations Implemented
1. **ONNX Runtime**: 73% faster inference (45ms → 12ms)
2. **Feature Caching**: 77% faster preprocessing (150ms → 35ms)
3. **Batch Endpoints**: 275% higher throughput (120 → 450 req/s)

### Results
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Cold Start | 2000ms | 500ms | -75% |
| P50 Latency | 95ms | 10ms | -89% |
| P95 Latency | 200ms | 42ms | -79% |
| Throughput | 120/s | 450/s | +275% |

---

## 📞 Support & Contact

**Issues**: [GitHub Issues](https://github.com/your-org/Scoring_Model_Enhanced/issues)
**Documentation**: [docs/README.md](docs/README.md)
**Presentations**: [docs/presentations/](docs/presentations/)

---

**Version**: 1.0.0
**Status**: ✅ Production Ready
**Last Updated**: December 2025
**License**: MIT
