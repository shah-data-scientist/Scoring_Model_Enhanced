#!/usr/bin/env python3
# Simple Monitoring Dashboard
# Run: poetry run python scripts/monitoring/dashboard.py

import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"

print("="*80)
print("CREDIT SCORING API - MONITORING DASHBOARD")
print("="*80)

# Read predictions log
pred_log = LOGS_DIR / "predictions.jsonl"
if not pred_log.exists():
    print("
No prediction logs found. API needs to run first.")
    exit(0)

# Parse logs
predictions = []
batches = []
errors = []

with open(pred_log, "r") as f:
    for line in f:
        try:
            entry = json.loads(line)
            event_type = entry.get("event_type")
            if event_type == "prediction":
                predictions.append(entry)
            elif event_type == "batch_prediction":
                batches.append(entry)
            elif event_type == "error":
                errors.append(entry)
        except:
            continue

print(f"
Logs Summary:")
print(f"  Total predictions logged: {len(predictions)}")
print(f"  Total batches: {len(batches)}")
print(f"  Total errors: {len(errors)}")

if batches:
    print("
" + "="*80)
    print("BATCH PREDICTIONS SUMMARY")
    print("="*80)
    
    total_apps = sum(b.get("num_applications", 0) for b in batches)
    avg_time = sum(b.get("total_time_ms", 0) for b in batches) / len(batches)
    
    print(f"  Total applications processed: {total_apps:,}")
    print(f"  Average batch processing time: {avg_time:.2f} ms")
    print(f"  Average time per application: {avg_time/max(total_apps/len(batches), 1):.2f} ms")
    
    # Risk distribution
    all_risk_dist = Counter()
    for b in batches:
        for level, count in b.get("risk_distribution", {}).items():
            all_risk_dist[level] += count
    
    print(f"
  Risk Distribution:")
    total = sum(all_risk_dist.values())
    for level in ["LOW", "MEDIUM", "HIGH"]:
        count = all_risk_dist.get(level, 0)
        pct = (count / total * 100) if total > 0 else 0
        print(f"    {level:8s}: {count:6,} ({pct:5.1f}%)")

    # Probability statistics
    all_probs = []
    for b in batches:
        stats = b.get("probability_stats", {})
        if stats:
            all_probs.append(stats.get("avg", 0))
    
    if all_probs:
        print(f"
  Probability Statistics:")
        print(f"    Min avg: {min(all_probs):.4f}")
        print(f"    Max avg: {max(all_probs):.4f}")
        print(f"    Overall avg: {sum(all_probs)/len(all_probs):.4f}")

if errors:
    print("
" + "="*80)
    print("ERRORS SUMMARY")
    print("="*80)
    
    error_types = Counter(e.get("error_type") for e in errors)
    print(f"  Total errors: {len(errors)}")
    print(f"  Error types:")
    for error_type, count in error_types.most_common(5):
        print(f"    {error_type}: {count}")

print("
" + "="*80)
