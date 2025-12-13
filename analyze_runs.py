import sqlite3
import pickle
from pathlib import Path
import json

print("=" * 80)
print("DETAILED ANALYSIS OF KEY RUNS")
print("=" * 80)

conn = sqlite3.connect('mlruns/mlflow.db')
cursor = conn.cursor()

# The two key runs
runs_of_interest = {
    'f5f0003602ae4c7a84b2cb6e865a2304': 'final_model_application (Exp 4: final_delivery)',
    '83e2e1ec9b254fc59b4d3bfa7ae75b1f': 'production_lightgbm_189features (Exp 6: production)'
}

for run_id, run_name in runs_of_interest.items():
    print(f"\n{'='*80}")
    print(f"RUN: {run_name}")
    print(f"ID: {run_id}")
    print(f"{'='*80}")
    
    # Get experiment info
    cursor.execute('SELECT experiment_id FROM runs WHERE run_uuid = ?', (run_id,))
    exp_id = cursor.fetchone()[0]
    cursor.execute('SELECT name FROM experiments WHERE experiment_id = ?', (exp_id,))
    exp_name = cursor.fetchone()[0]
    print(f"Experiment: {exp_name} (ID: {exp_id})")
    
    # Get parameters
    cursor.execute('SELECT key, value FROM params WHERE run_uuid = ? ORDER BY key', (run_id,))
    params = cursor.fetchall()
    if params:
        print("\nPARAMETERS:")
        for key, value in params:
            if 'threshold' in key.lower():
                print(f"  >>> {key}: {value}")
            else:
                print(f"  {key}: {value}")
    
    # Get metrics
    cursor.execute('SELECT key, value FROM metrics WHERE run_uuid = ? ORDER BY key', (run_id,))
    metrics = cursor.fetchall()
    if metrics:
        print("\nMETRICS:")
        for key, value in metrics:
            if 'threshold' in key.lower() or 'cost' in key.lower():
                print(f"  >>> {key}: {value}")
            else:
                print(f"  {key}: {value}")
    
    # Get tags
    cursor.execute('SELECT key, value FROM tags WHERE run_uuid = ? ORDER BY key', (run_id,))
    tags = cursor.fetchall()
    if tags:
        print("\nTAGS:")
        for key, value in tags:
            print(f"  {key}: {value}")
    
    # Check artifacts
    print("\nARTIFACTS:")
    mlruns_path = Path('mlruns')
    found = False
    for exp_dir in mlruns_path.iterdir():
        if exp_dir.is_dir() and exp_dir.name not in ['.trash', 'models']:
            run_dir = exp_dir / run_id
            if run_dir.exists():
                found = True
                print(f"  Directory: {run_dir}")
                artifacts_dir = run_dir / 'artifacts'
                if artifacts_dir.exists():
                    for item in artifacts_dir.rglob('*'):
                        if item.is_file():
                            rel_path = item.relative_to(artifacts_dir)
                            size = item.stat().st_size
                            print(f"    - {rel_path} ({size:,} bytes)")
                            
                            # If it's a model file, check if it matches production model
                            if item.name == 'model.pkl' or 'lgb' in item.name.lower():
                                try:
                                    with open(item, 'rb') as f:
                                        model = pickle.load(f)
                                    if hasattr(model, 'n_features_'):
                                        print(f"      Model features: {model.n_features_}")
                                    elif hasattr(model, 'num_features'):
                                        print(f"      Model features: {model.num_features()}")
                                except Exception as e:
                                    print(f"      Could not load model: {e}")
                else:
                    print(f"  No artifacts directory")
                break
    if not found:
        print(f"  Run directory not found in mlruns/")

conn.close()

# Check current production model
print(f"\n{'='*80}")
print("CURRENT PRODUCTION MODEL")
print(f"{'='*80}")
model_path = Path('models/production_model.pkl')
if model_path.exists():
    print(f"File: {model_path}")
    print(f"Size: {model_path.stat().st_size:,} bytes")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        if hasattr(model, 'n_features_'):
            print(f"Features: {model.n_features_}")
        elif hasattr(model, 'num_features'):
            print(f"Features: {model.num_features()}")
        print(f"Model type: {type(model).__name__}")
    except Exception as e:
        print(f"Could not load: {e}")

# Check static predictions
print(f"\n{'='*80}")
print("STATIC PREDICTIONS (Used by API)")
print(f"{'='*80}")
import pandas as pd
pred_path = Path('results/static_model_predictions.parquet')
if pred_path.exists():
    df = pd.read_parquet(pred_path)
    print(f"File: {pred_path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    
    # Calculate optimal threshold
    if 'TARGET' in df.columns and 'prediction_proba' in df.columns:
        print("\nCalculating optimal threshold from predictions...")
        thresholds = []
        costs = []
        cost_fn = 10
        cost_fp = 1
        
        for threshold in [0.01 * i for i in range(1, 100)]:
            y_pred = (df['prediction_proba'] >= threshold).astype(int)
            fn = ((df['TARGET'] == 1) & (y_pred == 0)).sum()
            fp = ((df['TARGET'] == 0) & (y_pred == 1)).sum()
            cost = fn * cost_fn + fp * cost_fp
            thresholds.append(threshold)
            costs.append(cost)
            
            if threshold in [0.33, 0.48, 0.50]:
                tn = ((df['TARGET'] == 0) & (y_pred == 0)).sum()
                tp = ((df['TARGET'] == 1) & (y_pred == 1)).sum()
                print(f"  Threshold {threshold:.2f}: Cost={cost:,}, FN={fn}, FP={fp}, TP={tp}, TN={tn}")
        
        optimal_idx = costs.index(min(costs))
        optimal_threshold = thresholds[optimal_idx]
        optimal_cost = costs[optimal_idx]
        print(f"\n>>> OPTIMAL THRESHOLD: {optimal_threshold:.2f} (Cost: {optimal_cost:,})")
