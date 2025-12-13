import sqlite3

conn = sqlite3.connect('mlruns/mlflow.db')
cursor = conn.cursor()

print('='*80)
print('CHECKING MLRUNS/MLFLOW.DB FOR PRODUCTION RUN')
print('='*80)

# Check what's in Experiment 4
cursor.execute('SELECT run_uuid, name, status FROM runs WHERE experiment_id = 4 ORDER BY start_time DESC')
runs = cursor.fetchall()

print(f'\nExperiment 4 (credit_scoring_final_delivery) runs: {len(runs)}')
for run_uuid, name, status in runs:
    name_str = (name[:50] if name else 'NO NAME')
    print(f'  - {name_str:50s} | {status:10s} | {run_uuid[:8]}')

# Check if the new run exists anywhere
target_uuid = '7ce7c8f6371e43af9ced637e5a4da7f0'
cursor.execute('SELECT run_uuid, experiment_id, name FROM runs WHERE run_uuid = ?', (target_uuid,))
new_run = cursor.fetchone()

print()
if new_run:
    print(f'✓ New run found in experiment {new_run[1]}: {new_run[2]}')
    
    # Check parameters
    cursor.execute('SELECT key, value FROM params WHERE run_uuid = ? AND key IN ("optimal_threshold", "n_features")', (target_uuid,))
    params = cursor.fetchall()
    if params:
        print('  Parameters:')
        for key, val in params:
            print(f'    - {key}: {val}')
    
    # Check artifacts location
    from pathlib import Path
    run_prefix = target_uuid[:2]
    artifacts_path = Path(f'mlruns/{run_prefix}/{target_uuid}/artifacts')
    print(f'\n  Expected artifacts location: {artifacts_path}')
    print(f'  Exists: {artifacts_path.exists()}')
    
    if artifacts_path.exists():
        files = list(artifacts_path.rglob('*'))
        files = [f for f in files if f.is_file()]
        print(f'  Files: {len(files)}')
        for f in files:
            print(f'    - {f.name}')
else:
    print(f'✗ New run NOT found in database!')
    print(f'  Looking for UUID: {target_uuid}')
    
    # List all runs to see what we have
    cursor.execute('SELECT COUNT(*) FROM runs')
    total = cursor.fetchone()[0]
    print(f'\n  Total runs in database: {total}')

conn.close()
