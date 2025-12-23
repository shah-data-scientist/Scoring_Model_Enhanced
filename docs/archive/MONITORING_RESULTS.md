# Monitoring Results & Analysis

## 1. Data Drift Analysis

We monitor data stability using statistical tests to detect distribution shifts between training data (reference) and production data (current).

### Methodology
Drift detection logic is implemented in `api/drift_detection.py` and uses the following metrics:

| Feature Type | Metric Used | Threshold | Interpretation |
| :--- | :--- | :--- | :--- |
| **Numerical** | **Kolmogorov-Smirnov (KS)** | p-value < 0.05 | Detects changes in distribution shape |
| **Numerical** | **Population Stability Index (PSI)** | PSI > 0.1 | Measures magnitude of shift (0.1=Low, 0.25=High) |
| **Categorical** | **Chi-Square Test** | p-value < 0.05 | Detects changes in category frequencies |

### Drift Testing
We generated synthetic drift data in `data/drift_samples/` to validate our detection mechanisms:
*   **Numerical Drift**: `AMT_INCOME_TOTAL` was shifted by +20%, triggering KS and PSI alerts.
*   **Missing Values**: Artificial `NaN` injection in `EXT_SOURCE_2` triggers data quality alerts.
*   **Out-of-Range**: Values for `DAYS_BIRTH` were modified to simulate invalid ages (<18), triggering range checks.

The results are accessible via the `/monitoring/drift/batch/{batch_id}` endpoint, which returns a detailed report of drifted features.

## 2. Log Collection Strategy

All model activities are logged to local JSONL files in the `logs/` directory for auditability and analysis.

### Prediction Logs (`logs/predictions.jsonl`)
Every inference request is logged with structural metadata.
**Format:**
```json
{
  "timestamp": "2025-12-17T14:30:00.123",
  "event_type": "prediction",
  "sk_id_curr": 100001,
  "probability": 0.1234,
  "prediction": 0,
  "risk_level": "LOW",
  "processing_time_ms": 42.5,
  "source": "api"
}
```

### Batch Summaries
Batch operations log aggregated statistics to track throughput and risk distribution over time:
```json
{
  "event_type": "batch_prediction",
  "num_applications": 100,
  "avg_time_per_app_ms": 35.2,
  "risk_distribution": {"LOW": 80, "HIGH": 20},
  "probability_stats": {"min": 0.01, "max": 0.89, "avg": 0.15}
}
```

### Error Logs (`logs/errors.jsonl`)
Captures runtime exceptions with stack trace information and endpoint context for debugging.

## 3. Performance & Metrics Analysis

### Latency Monitoring
Performance is tracked at two levels:
1.  **Single Inference**: The `processing_time_ms` field in prediction logs captures end-to-end latency (excluding network).
2.  **Batch Processing**: The `avg_time_per_app_ms` metric monitors the efficiency of the bulk processing pipeline.

### Dashboard Integration
The Streamlit dashboard (`pages/monitoring.py`) consumes these logs and database records to visualize:
*   **Drift Status**: Percentage of features with significant drift.
*   **Throughput**: Daily prediction counts.
*   **Latency Trends**: Evolution of response times to detect performance regression.
