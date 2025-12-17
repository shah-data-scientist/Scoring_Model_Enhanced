#!/usr/bin/env python3
# Performance Profiling Script
# Run: poetry run python scripts/monitoring/profile_performance.py

import cProfile
import pstats
import time
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from io import StringIO

PROJECT_ROOT = Path(__file__).parent.parent.parent

print("="*80)
print("API PERFORMANCE PROFILING")
print("="*80)

# Load model
print("Loading model...")
with open(PROJECT_ROOT / "models" / "production_model.pkl", 'rb') as f:
    model = pickle.load(f)

# Load sample data
print("Loading test data...")
X_test = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "X_test.csv")
sample = X_test.sample(min(100, len(X_test)))

print(f"Test sample: {len(sample)} applications")

# Profile prediction time
print("="*80)
print("PREDICTION PERFORMANCE")
print("="*80)

def benchmark_predictions(X, n_iterations=10):
    times = []
    for _ in range(n_iterations):
        start = time.time()
        predictions = model.predict(X.values)
        probabilities = model.predict_proba(X.values)[:, 1]
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
    
    return times

print(f"Benchmarking {len(sample)} predictions (10 iterations)...")
times = benchmark_predictions(sample)

print(f"Results:")
print(f"  Min time: {min(times):.2f} ms")
print(f"  Max time: {max(times):.2f} ms")
print(f"  Avg time: {np.mean(times):.2f} ms")
print(f"  Median time: {np.median(times):.2f} ms")
print(f"  Std dev: {np.std(times):.2f} ms")
print(f"  Time per application: {np.mean(times)/len(sample):.2f} ms")
print(f"  Throughput: {len(sample)/(np.mean(times)/1000):.1f} predictions/second")

# Profile with cProfile
print("="*80)
print("DETAILED PROFILING (cProfile)")
print("="*80)

profiler = cProfile.Profile()
profiler.enable()

# Run predictions
for _ in range(5):
    predictions = model.predict(sample.values)
    probabilities = model.predict_proba(sample.values)[:, 1]

profiler.disable()

# Print stats
s = StringIO()
ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
ps.print_stats(20)
print(s.getvalue())

# Save profile
profile_path = PROJECT_ROOT / "reports" / "performance_profile.txt"
profile_path.parent.mkdir(parents=True, exist_ok=True)
with open(profile_path, 'w') as f:
    ps = pstats.Stats(profiler, stream=f).sort_stats('cumulative')
    ps.print_stats()

print(f"Full profile saved to: {profile_path}")

# Recommendations
print("="*80)
print("OPTIMIZATION RECOMMENDATIONS")
print("="*80)

avg_ms = np.mean(times)
if avg_ms < 100:
    print("  [EXCELLENT] Prediction latency < 100ms")
elif avg_ms < 500:
    print("  [GOOD] Prediction latency < 500ms")
else:
    print("  [WARNING] Prediction latency > 500ms")
    print("  Consider:")
    print("    - Model quantization")
    print("    - Reduce number of features")
    print("    - Use faster hardware")

print(f"  Current: {avg_ms:.2f} ms average")
print(f"  Target: < 100 ms for good user experience")
