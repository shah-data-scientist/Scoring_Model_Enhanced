# Model Card: Credit Scoring

## Overview
- Task: Binary classification (default risk)
- Algorithm: LightGBM / XGBoost (see models/)
- Data: Home Credit dataset (+ engineered features)

## Intended Use
- Risk assessment for loan applications
- Not a replacement for human decision-making

## Metrics
- ROC AUC (target ≥0.75)
- Precision/Recall @ threshold
- Calibration

## Limitations
- Bias risks due to historical data
- Concept drift over time

## Ethics & Fairness
- Monitor drift and performance ([docs/DRIFT_DETECTION.md](DRIFT_DETECTION.md))
- Review subgroup metrics

## Data
- Source: Kaggle (Home Credit)
- Anonymization: `SK_ID_CURR` mapping applied

## Maintenance
- Monitoring: Streamlit monitoring page
- Retraining policy: see [docs/MODEL_MONITORING.md](MODEL_MONITORING.md)
