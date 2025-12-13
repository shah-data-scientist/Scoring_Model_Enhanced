# Data Drift Detection & Quality Monitoring

## Overview

Real-time data drift detection and quality monitoring for the credit scoring model. Detects distribution changes and data quality issues automatically.

## Features Implemented

### 1. Statistical Drift Detection

**Kolmogorov-Smirnov (KS) Test** - For numeric features
- Compares two distributions
- Returns KS statistic and p-value
- p-value < 0.05 indicates significant drift

**Chi-Square Test** - For categorical features
- Tests for category distribution changes
- Useful for detecting shifts in proportions

**Population Stability Index (PSI)**
- Measures population shift over time
- PSI < 0.1: No drift
- PSI 0.1-0.25: Small drift
- PSI > 0.25: Significant drift

### 2. Data Quality Checks

**Missing Value Detection**
- Monitors percentage of missing values per feature
- Alert threshold: > 20% missing

**Out-of-Range Detection**
- Compares current values against training data bounds
- Detects sudden value shifts
- Identifies impossible values

**Schema Validation**
- Ensures all expected columns are present
- Validates column count
- Reports missing/extra columns

## API Endpoints

### POST `/monitoring/drift`
Detect drift in a single feature.

**Request:**
```json
{
  "feature_name": "EXT_SOURCE_1",
  "feature_type": "numeric",
  "reference_data": [0.5, 0.6, 0.55, ...],
  "current_data": [0.48, 0.61, 0.52, ...],
  "alert_threshold": 0.05
}
```

**Response:**
```json
{
  "feature_name": "EXT_SOURCE_1",
  "feature_type": "numeric",
  "drift_test": "KS",
  "is_drifted": true,
  "interpretation": "⚠️ Small drift detected",
  "statistics": {
    "ks_statistic": 0.152,
    "p_value": 0.023,
    "psi": 0.089,
    "reference_mean": 0.502,
    "current_mean": 0.498
  }
}
```

### POST `/monitoring/drift/batch/{batch_id}`
Detect drift for all features in a batch.

**Query Parameters:**
- `reference_batch_id` (optional): Batch to use as reference

**Response:**
```json
{
  "batch_id": 1,
  "features_checked": 45,
  "features_drifted": 3,
  "results": {
    "EXT_SOURCE_1": {...},
    "EXT_SOURCE_2": {...}
  }
}
```

### POST `/monitoring/quality`
Check data quality for a dataset.

**Request:**
```json
{
  "dataframe_dict": {
    "AMT_CREDIT": [500000, 450000, 550000],
    "AMT_INCOME": [180000, 160000, 200000]
  },
  "check_missing": true,
  "check_range": true,
  "check_schema": true,
  "expected_columns": ["AMT_CREDIT", "AMT_INCOME"]
}
```

**Response:**
```json
{
  "valid": true,
  "missing_values": {
    "AMT_CREDIT": 0.5,
    "AMT_INCOME": 0.2
  },
  "out_of_range": {
    "AMT_CREDIT": {
      "min": 450000,
      "max": 550000,
      "expected_min": 400000,
      "expected_max": 600000,
      "out_of_range_count": 0,
      "out_of_range_pct": 0.0,
      "status": "OK"
    }
  },
  "schema_validation": {
    "valid": true,
    "match_percentage": 100
  },
  "summary": "✅ All checks passed"
}
```

### GET `/monitoring/drift/history/{feature_name}`
Get drift detection history for a feature.

**Query Parameters:**
- `limit` (1-100, default 30): Number of records to return

**Response:**
```json
{
  "feature_name": "EXT_SOURCE_1",
  "records": [
    {
      "recorded_at": "2025-12-13T10:30:00",
      "drift_score": 0.089,
      "drift_type": "PSI",
      "is_drifted": false,
      "reference_mean": 0.502,
      "current_mean": 0.498
    }
  ],
  "count": 5
}
```

### GET `/monitoring/stats/summary`
Get overall data quality and drift statistics.

**Response:**
```json
{
  "data_drift": {
    "total_features_checked": 189,
    "features_with_drift": 12,
    "drift_percentage": 6.35
  },
  "predictions": {
    "total": 5420
  }
}
```

## Streamlit Dashboard

### 📊 Feature Drift Tab
- Select batch ID and reference batch
- Adjust alert threshold
- View feature-level drift results
- See KS statistics and PSI scores

### ✔️ Data Quality Tab
- Run quality checks on any batch
- View missing value rates
- Identify out-of-range values
- Validate schema

### 📈 Drift History Tab
- Select feature to analyze
- View drift scores over time
- Chart visualization
- Historical data table

## Database Models

### DataDrift Table
```sql
CREATE TABLE data_drift (
  id INTEGER PRIMARY KEY,
  feature_name VARCHAR(100) NOT NULL,
  drift_score FLOAT NOT NULL,
  drift_type VARCHAR(50) NOT NULL,
  is_drifted BOOLEAN,
  reference_mean FLOAT,
  current_mean FLOAT,
  reference_std FLOAT,
  current_std FLOAT,
  batch_id INTEGER,
  n_samples INTEGER,
  recorded_at TIMESTAMP
);
```

## Example Usage

### Python Client
```python
import requests
import numpy as np

API_URL = "http://localhost:8000"

# Generate sample data
reference = np.random.normal(0.5, 0.1, 100)
current = np.random.normal(0.52, 0.12, 100)

# Detect drift
response = requests.post(
    f"{API_URL}/monitoring/drift",
    json={
        "feature_name": "EXT_SOURCE_1",
        "feature_type": "numeric",
        "reference_data": reference.tolist(),
        "current_data": current.tolist(),
        "alert_threshold": 0.05
    }
)

result = response.json()
print(f"Drifted: {result['is_drifted']}")
print(f"KS Statistic: {result['statistics']['ks_statistic']:.4f}")
print(f"PSI: {result['statistics']['psi']:.4f}")
```

### cURL
```bash
# Detect drift
curl -X POST http://localhost:8000/monitoring/drift \
  -H "Content-Type: application/json" \
  -d '{
    "feature_name": "AMT_CREDIT",
    "feature_type": "numeric",
    "reference_data": [500000, 450000, 550000],
    "current_data": [510000, 460000, 560000],
    "alert_threshold": 0.05
  }'

# Get drift history
curl http://localhost:8000/monitoring/drift/history/EXT_SOURCE_1?limit=30
```

## Configuration

### Alert Thresholds

Default values can be customized:

```python
# In api/drift_detection.py
KS_ALERT_THRESHOLD = 0.05  # p-value
PSI_WARNING = 0.1           # Small drift
PSI_CRITICAL = 0.25         # Significant drift
MISSING_VALUE_ALERT = 0.20  # 20% threshold
OUT_OF_RANGE_ALERT = 0.05   # 5% threshold
```

## Interpreting Results

### KS Test Results
- **p-value > 0.05**: No drift detected ✅
- **p-value < 0.05**: Significant drift detected 🔴

### PSI Interpretation
- **< 0.1**: No population shift ✅
- **0.1 - 0.25**: Small shift ⚠️
- **> 0.25**: Significant shift 🔴

### Data Quality Status
- **Green**: All checks passed
- **Yellow**: Minor warnings (5-20% issues)
- **Red**: Critical issues (>20% or schema mismatch)

## Performance Considerations

### Large Datasets
- KS test is O(n log n)
- PSI calculation uses binning (adjustable bin count)
- For 1M+ samples, consider sampling

### Real-Time Monitoring
- Cache reference distributions
- Use incremental statistics
- Store only recent history

## Future Enhancements

- [ ] Real-time streaming drift detection
- [ ] Automatic anomaly detection (Isolation Forest)
- [ ] Feature importance for drift
- [ ] Custom drift thresholds per feature
- [ ] Email/Slack alerts on drift detection
- [ ] Multivariate drift detection (Mahalanobis distance)
