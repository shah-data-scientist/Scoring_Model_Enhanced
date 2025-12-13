#!/usr/bin/env python3
# Data Drift Detection using Evidently AI
# Run: poetry run python scripts/monitoring/detect_drift.py

import json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
except ImportError:
    print("Install evidently: poetry add evidently")
    exit(1)

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Load training reference
X_train = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "X_train.csv")
X_val = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "X_val.csv")
reference = pd.concat([X_train, X_val], axis=0, ignore_index=True)

print(f"Reference data: {len(reference):,} rows")

# Create drift report
report = Report(metrics=[DataDriftPreset()])

# For now, use test set as "current" (in production, load from logs)
X_test = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "X_test.csv")
current = X_test.sample(min(1000, len(X_test)))

print(f"Current data: {len(current):,} rows")

# Run drift analysis
report.run(reference_data=reference.sample(min(10000, len(reference))), 
           current_data=current)

# Save report
output_dir = PROJECT_ROOT / "reports" / "drift"
output_dir.mkdir(parents=True, exist_ok=True)
report_path = output_dir / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
report.save_html(str(report_path))

print(f"Drift report saved: {report_path}")
