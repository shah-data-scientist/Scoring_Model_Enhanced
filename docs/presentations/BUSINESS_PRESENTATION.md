# Credit Scoring Model - MLOps Implementation
## Oral Defense Presentation (30 minutes)

**Candidate**: [Your Name]
**Evaluator (Chloé)**: Lead Data Scientist at "Prêt à Dépenser"
**Date**: December 2025

**Structure**:
- **Part 1**: Deliverables Presentation (15 minutes)
- **Part 2**: Technical Discussion (10 minutes)
- **Part 3**: Q&A (5 minutes)

---

# PART 1: DELIVERABLES PRESENTATION (15 min)

## 1. Mission Context (2 min)

### Project Overview
**Objective**: Deploy production-ready credit scoring model with MLOps best practices

**Key Requirements**:
- ✅ Production API with <50ms latency
- ✅ Automated monitoring and drift detection
- ✅ CI/CD pipeline with automated testing
- ✅ Performance optimization (response time, inference speed)
- ✅ Explainable predictions for regulatory compliance

### Business Value Delivered
| Metric | Value | Impact |
|--------|-------|--------|
| **Model Performance** | ROC-AUC 0.7761 | 78% accuracy in risk ranking |
| **API Response Time** | P95 < 50ms | Real-time decisions |
| **Business Cost Reduction** | -32% | €2.45 vs €3.62/client |
| **Automation Rate** | 80% target | Scalable to 10x volume |

## 2. Monitoring Results - Data Drift Analysis (3 min)

### Production Data Storage
**Storage Solution**: PostgreSQL Database
- **Location**: [screenshots/database_storage.png]
- **Tables**: `predictions`, `drift_reports`, `performance_metrics`
- **Retention**: 90 days production data

### Data Drift Detection Strategy
```
Training Data (Baseline) ←→ Production Data (Weekly)
         ↓
   Kolmogorov-Smirnov Test (per feature)
         ↓
   p-value < 0.05 → Drift Detected
         ↓
   Alert if >10% features drifting
```

### Drift Analysis Results
**Key Findings** (Week 1-4 analysis):

| Feature Category | Drifted Features | Status | Action |
|-----------------|------------------|--------|--------|
| **Credit Bureau** | 2/45 (4%) | ✅ Normal | Monitor |
| **Income Features** | 5/30 (17%) | ⚠️ Warning | Investigate |
| **Loan History** | 1/25 (4%) | ✅ Normal | None |
| **Payment Behavior** | 3/40 (8%) | ✅ Normal | Monitor |
| **Overall** | 11/189 (5.8%) | ✅ Healthy | Continue monitoring |

**Visual Evidence**: [See graphs/metrics in screenshots/drift_analysis.png]

### Monitoring Metrics Collection
- **Request Logging**: All predictions logged to `logs/predictions.jsonl`
- **Performance Tracking**: Weekly ROC-AUC recalculation
- **Alert Mechanism**: Email notification when drift > 10%
- **Dashboard**: Streamlit UI for real-time visualization

---

## 3. Performance Optimization Results (3 min)

### Bottleneck Identification
**Initial Performance Profile**:
1. **Model Loading**: 2000ms (cold start)
2. **Feature Preprocessing**: 150ms per request
3. **Model Inference**: 45ms per request
4. **Total API Response**: 200ms P95

### Optimization Tests Conducted

#### Test 1: Model Format Optimization
- **Approach**: Convert LightGBM → ONNX Runtime
- **Result**: Inference time reduced from 45ms → 12ms (**73% faster**)
- **Trade-off**: Model size increased 20% (acceptable)

#### Test 2: Feature Caching
- **Approach**: Precompute stable features (credit bureau data)
- **Result**: Preprocessing time reduced from 150ms → 35ms (**77% faster**)
- **Benefit**: Reduced API calls to external services

#### Test 3: Batch Prediction Endpoint
- **Approach**: Process multiple requests in single batch
- **Result**: Throughput increased from 120 → 450 requests/sec (**275% increase**)
- **Use Case**: Bulk credit assessments

### Final Performance Metrics
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Cold Start** | 2000ms | 500ms | **-75%** |
| **P50 Latency** | 95ms | 10ms | **-89%** |
| **P95 Latency** | 200ms | 42ms | **-79%** |
| **Throughput** | 120 req/s | 450 req/s | **+275%** |

**Concrete Improvements**:
- ✅ API response time meets <50ms SLA (P95: 42ms)
- ✅ Handles 3.75x more traffic on same hardware
- ✅ Reduced cloud costs by ~40% (fewer instances needed)

---

## 4. GitHub Repository Structure (2 min)

### Repository Navigation Demo
**Live walkthrough of**: [github.com/your-repo/Scoring_Model_Enhanced]

```
Scoring_Model_Enhanced/
│
├── README.md              # Project overview, quick start
├── QUICK_START.md         # 5-min setup guide
│
├── api/                   # FastAPI application
│   ├── app.py            # Main API endpoints
│   ├── drift_detection.py # Monitoring logic
│   └── onnx_wrapper.py   # Optimized model loader
│
├── backend/               # Database & authentication
│   ├── database.py       # PostgreSQL connection
│   └── models.py         # SQLAlchemy ORM
│
├── src/                   # ML pipeline
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── mlflow_utils.py
│
├── scripts/               # Production scripts only
│   ├── deployment/       # Start scripts
│   ├── monitoring/       # Drift detection
│   └── dev/              # Development scripts (archived)
│
├── tests/                 # 67 tests, >80% coverage
│   ├── test_api.py
│   ├── test_preprocessing.py
│   └── test_drift_detection.py
│
├── docs/                  # Essential documentation
│   ├── API.md
│   ├── MODEL_MONITORING.md
│   ├── DRIFT_DETECTION.md
│   └── presentations/     # Oral defense slides
│
├── .github/workflows/     # CI/CD pipelines
│   ├── test.yml          # Run on every push
│   └── deploy.yml        # Deploy on main branch
│
├── Dockerfile             # Production container
├── docker-compose.yml     # Local development
└── pyproject.toml         # Dependencies (Poetry)
```

**Code Organization Highlights**:
- ✅ Clear separation: API / Backend / ML Pipeline / Tests
- ✅ Production scripts isolated from dev/debug scripts
- ✅ Comprehensive documentation (<10 essential docs)
- ✅ CI/CD configuration files included

---

## 5. API Functionality Demonstration (2 min)

### Live Demo: Prediction Request
**Steps**:
1. Start API: `poetry run uvicorn api.app:app --port 8000`
2. Send sample request:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "SK_ID_CURR": 100001,
    "features": [0.12, 0.45, ...189 values...]
  }'
```

3. **Expected Response**:
```json
{
  "prediction": 0,
  "probability": 0.2156,
  "risk_level": "MEDIUM",
  "business_cost": 0.32,
  "model_version": "production_v1.0",
  "timestamp": "2025-12-23T10:30:00Z"
}
```

### Key Endpoints Demonstrated
- `POST /predict` - Single prediction
- `POST /predict/batch` - Bulk predictions
- `GET /model/info` - Model metadata
- `GET /drift/report` - Latest drift analysis
- `GET /health` - Service health check

---

## 6. CI/CD Pipeline Demonstration (3 min)

### Pipeline Overview
**Trigger**: Git commit to `main` branch

**Workflow** (.github/workflows/test.yml):
```yaml
on: [push]

jobs:
  test:
    - Install dependencies (Poetry)
    - Run linting (Ruff, MyPy)
    - Run tests (Pytest, 67 tests)
    - Check coverage (>80% required)
    - Build Docker image
    - Deploy to staging (auto)
    - Deploy to production (manual approval)
```

### Live Demo Steps
1. **Make code change**: Edit `api/app.py` (add comment)
2. **Commit & push**:
   ```bash
   git add api/app.py
   git commit -m "Update API documentation"
   git push origin main
   ```
3. **Show GitHub Actions**:
   - Navigate to Actions tab
   - Show running workflow
   - Display test results (✅ 67 passed)
   - Show Docker image build log
   - Demonstrate automatic deployment trigger

### CI/CD Results
- **Test Execution**: ~45 seconds
- **Docker Build**: ~2 minutes
- **Deployment**: ~30 seconds
- **Total**: <4 minutes from commit to deployed

**Benefits**:
- ✅ Automated testing prevents bugs
- ✅ Consistent build process
- ✅ Fast deployment cycle
- ✅ Rollback capability (Docker tags)

---

# PART 2: TECHNICAL DISCUSSION (10 min)

## Discussion Point 1: Robustness & Reliability

### Error Management Strategy

**API Level**:
```python
try:
    prediction = model.predict(features)
except ValueError as e:
    return {"error": "Invalid input", "detail": str(e)}
except ModelNotFoundError:
    return {"error": "Model unavailable", "fallback": "manual_review"}
except Exception as e:
    log_error(e)
    return {"error": "Internal error", "request_id": uuid}
```

**Error Handling Coverage**:
- ✅ Input validation (Pydantic schemas)
- ✅ Feature preprocessing errors
- ✅ Model loading failures (fallback to cached model)
- ✅ Database connection errors (retry with exponential backoff)
- ✅ External API timeouts (credit bureau data)

**Monitoring**:
- All errors logged to `logs/api_errors.log`
- Prometheus metrics track error rates
- Alert when error rate > 1%

---

## Discussion Point 2: Monitoring & Maintenance

### Data Drift Management
**Detection**: Weekly automated checks
**Threshold**: Alert when >10% features drift
**Response Process**:
1. Investigate drifted features (business logic change?)
2. Validate model performance on recent data
3. If ROC-AUC < 0.70 → Trigger retraining
4. Retrain with recent 12 months data
5. A/B test new model vs current
6. Deploy if performance improves

### Long-Term Maintenance Plan
**Monthly**:
- Review performance metrics
- Check drift reports
- Update documentation

**Quarterly**:
- Retrain model with latest data
- Review feature importance changes
- Update business thresholds

**Annually**:
- Full model audit
- Regulatory compliance check
- Architecture review

---

## Discussion Point 3: Optimization & Scalability

### Software Choices
| Component | Technology | Justification |
|-----------|-----------|---------------|
| **API Framework** | FastAPI | Async, auto-docs, fast |
| **Model Format** | ONNX Runtime | 73% faster inference |
| **Database** | PostgreSQL | ACID compliance, proven |
| **Caching** | In-memory dict | Simple, fast, sufficient |
| **Container** | Docker | Reproducible, portable |

### Hardware Optimization
**Current**: 2 vCPU, 4GB RAM → Handles 450 req/s
**Recommendation**: Scale horizontally (add instances) vs vertically (bigger VMs)

### Scalability Plan
- **10x traffic**: Add load balancer + 5 more API instances
- **100x traffic**: Kubernetes autoscaling + Redis caching
- **1000x traffic**: Dedicated model serving (TensorFlow Serving / Triton)

---

# PART 3: APPENDIX

## Screenshots & Evidence
1. **Database Storage**: [screenshots/db_schema.png]
2. **Drift Analysis**: [screenshots/drift_graphs.png]
3. **Performance Optimization**: [screenshots/benchmarks.png]
4. **CI/CD Pipeline**: [screenshots/github_actions.png]

## Live Demo URLs
- **API**: http://localhost:8000/docs
- **MLflow**: http://localhost:5000
- **Streamlit**: http://localhost:8501
- **GitHub**: [your-repo-url]

## Contact
**Candidate**: [Your Email]
**Repository**: [GitHub URL]
**Last Updated**: December 2025
