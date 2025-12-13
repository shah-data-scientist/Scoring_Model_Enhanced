import sqlite3
import json
from pathlib import Path

# Check experiments
print("=" * 80)
print("MLflow Experiments")
print("=" * 80)
conn = sqlite3.connect('mlruns/mlflow.db')
cursor = conn.cursor()
cursor.execute('SELECT experiment_id, name, lifecycle_stage FROM experiments')
experiments = cursor.fetchall()
for exp in experiments:
    print(f"ID: {exp[0]}, Name: {exp[1]}, Stage: {exp[2]}")

# Check runs for each experiment
print("\n" + "=" * 80)
print("MLflow Runs")
print("=" * 80)
cursor.execute('''
    SELECT run_uuid, experiment_id, name, status, start_time, end_time 
    FROM runs 
    ORDER BY start_time DESC
''')
runs = cursor.fetchall()
for run in runs:
    print(f"\nRun ID: {run[0]}")
    print(f"  Experiment: {run[1]}")
    print(f"  Name: {run[2]}")
    print(f"  Status: {run[3]}")

# Check for specific runs
print("\n" + "=" * 80)
print("Searching for specific runs")
print("=" * 80)
cursor.execute('''
    SELECT run_uuid, name FROM runs 
    WHERE name LIKE '%credit_scoring_final_delivery%' OR name LIKE '%credit_scoring_production%'
''')
specific_runs = cursor.fetchall()
for run in specific_runs:
    print(f"Run ID: {run[0]}, Name: {run[1]}")

conn.close()

# Check what parameters/metrics are stored for these runs
print("\n" + "=" * 80)
print("Run Parameters and Metrics")
print("=" * 80)
conn = sqlite3.connect('mlruns/mlflow.db')
cursor = conn.cursor()

for run in specific_runs:
    print(f"\n{'='*60}")
    print(f"Run: {run[1]} ({run[0]})")
    print(f"{'='*60}")
    
    # Get params
    cursor.execute('SELECT key, value FROM params WHERE run_uuid = ?', (run[0],))
    params = cursor.fetchall()
    if params:
        print("\nParameters:")
        for p in params:
            print(f"  {p[0]}: {p[1]}")
    
    # Get metrics
    cursor.execute('SELECT key, value FROM metrics WHERE run_uuid = ?', (run[0],))
    metrics = cursor.fetchall()
    if metrics:
        print("\nMetrics:")
        for m in metrics:
            print(f"  {m[0]}: {m[1]}")
    
    # Get tags
    cursor.execute('SELECT key, value FROM tags WHERE run_uuid = ?', (run[0],))
    tags = cursor.fetchall()
    if tags:
        print("\nTags:")
        for t in tags:
            print(f"  {t[0]}: {t[1]}")

conn.close()

# Check artifacts in directory
print("\n" + "=" * 80)
print("Checking artifacts in mlruns directory")
print("=" * 80)
mlruns_path = Path('mlruns')
for run in specific_runs:
    run_id = run[0]
    # Find run directory (first 2 chars are subfolder)
    for exp_dir in mlruns_path.iterdir():
        if exp_dir.is_dir() and exp_dir.name not in ['.trash', 'models']:
            run_dir = exp_dir / run_id
            if run_dir.exists():
                print(f"\nRun: {run[1]}")
                print(f"  Directory: {run_dir}")
                artifacts_dir = run_dir / 'artifacts'
                if artifacts_dir.exists():
                    print(f"  Artifacts:")
                    for item in artifacts_dir.rglob('*'):
                        if item.is_file():
                            print(f"    - {item.relative_to(artifacts_dir)} ({item.stat().st_size} bytes)")
                else:
                    print(f"  No artifacts directory found")

# Check current model
print("\n" + "=" * 80)
print("Current production model")
print("=" * 80)
model_path = Path('models/production_model.pkl')
if model_path.exists():
    print(f"models/production_model.pkl exists ({model_path.stat().st_size} bytes)")
    print(f"Modified: {model_path.stat().st_mtime}")
else:
    print("models/production_model.pkl NOT FOUND")
