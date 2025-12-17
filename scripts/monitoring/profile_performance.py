#!/usr/bin/env python3
# Performance Profiling Script
# Run: poetry run python scripts/monitoring/profile_performance.py

import cProfile
import pstats
import time
import os
import psutil
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from io import StringIO

PROJECT_ROOT = Path(__file__).parent.parent.parent

def get_resource_usage():
    """Get current process memory and CPU usage."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    cpu_pct = process.cpu_percent(interval=0.1)
    return {
        'memory_mb': mem_info.rss / (1024 * 1024),
        'cpu_percent': cpu_pct
    }

print("="*80)
print("API PERFORMANCE PROFILING")
print("="*80)

# Initial usage
initial_resources = get_resource_usage()
print(f"Initial State: Memory={initial_resources['memory_mb']:.2f} MB, CPU={initial_resources['cpu_percent']}%")

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
    mem_start = get_resource_usage()['memory_mb']
    
    for _ in range(n_iterations):
        start = time.time()
        predictions = model.predict(X.values)
        probabilities = model.predict_proba(X.values)[:, 1]
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
    
    mem_end = get_resource_usage()['memory_mb']
    return times, mem_end - mem_start

print(f"Benchmarking {len(sample)} predictions (10 iterations)...")
times, mem_delta = benchmark_predictions(sample)

print(f"Results:")
print(f"  Min time: {min(times):.2f} ms")
print(f"  Max time: {max(times):.2f} ms")
print(f"  Avg time: {np.mean(times):.2f} ms")
print(f"  Median time: {np.median(times):.2f} ms")
print(f"  Std dev: {np.std(times):.2f} ms")
print(f"  Time per application: {np.mean(times)/len(sample):.2f} ms")
print(f"  Throughput: {len(sample)/(np.mean(times)/1000):.1f} predictions/second")

# Resource Usage section
print("="*80)
print("RESOURCE USAGE")
print("="*80)
current_resources = get_resource_usage()
print(f"  Current Memory: {current_resources['memory_mb']:.2f} MB")
print(f"  Memory Delta (during benchmark): {mem_delta:+.2f} MB")
print(f"  Peak CPU Load (sampled): {current_resources['cpu_percent']}%")

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
