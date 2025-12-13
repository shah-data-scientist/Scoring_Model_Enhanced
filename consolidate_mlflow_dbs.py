"""
Consolidate MLflow databases - copy new production run to original database
"""
import sqlite3
import json

print("="*80)
print("CONSOLIDATING MLFLOW DATABASES")
print("="*80)

# Connect to both databases
source_db = sqlite3.connect('mlflow.db')  # Root - has new production run
target_db = sqlite3.connect('mlruns/mlflow.db')  # Original - has 66 runs

source_cursor = source_db.cursor()
target_cursor = target_db.cursor()

# Get the new production run from source
source_cursor.execute('''
    SELECT e.experiment_id, e.name, r.run_uuid, r.name as run_name, r.status, 
           r.user_id, r.start_time, r.end_time, r.source_type, r.source_name,
           r.entry_point_name
    FROM runs r
    JOIN experiments e ON r.experiment_id = e.experiment_id
    WHERE e.name = 'credit_scoring_final_delivery'
    AND r.name = 'production_lightgbm_189features_final'
''')

run_data = source_cursor.fetchone()
if not run_data:
    print("ERROR: Could not find production run in source database!")
    exit(1)

exp_id, exp_name, run_uuid, run_name, status, user_id, start_time, end_time, source_type, source_name, entry_point = run_data

print(f"✓ Found production run in source DB:")
print(f"  Experiment: {exp_name} (ID: {exp_id})")
print(f"  Run: {run_name} (UUID: {run_uuid})")

# Check if experiment exists in target DB
target_cursor.execute('SELECT experiment_id FROM experiments WHERE name = ?', (exp_name,))
target_exp = target_cursor.fetchone()

if target_exp:
    target_exp_id = target_exp[0]
    print(f"✓ Experiment exists in target DB (ID: {target_exp_id})")
else:
    print(f"✗ Experiment not found in target DB - creating it...")
    target_cursor.execute('''
        INSERT INTO experiments (name, lifecycle_stage, artifact_location)
        VALUES (?, 'active', '')
    ''', (exp_name,))
    target_db.commit()
    target_cursor.execute('SELECT experiment_id FROM experiments WHERE name = ?', (exp_name,))
    target_exp_id = target_cursor.fetchone()[0]
    print(f"✓ Created experiment with ID: {target_exp_id}")

# Check if run already exists in target DB
target_cursor.execute('SELECT run_uuid FROM runs WHERE run_uuid = ?', (run_uuid,))
existing_run = target_cursor.fetchone()

if existing_run:
    print(f"✓ Run already exists in target DB - skipping insertion")
    run_exists = True
else:
    print(f"✓ Inserting run into target DB...")
    # Insert run
    target_cursor.execute('''
        INSERT INTO runs 
        (run_uuid, experiment_id, user_id, status, start_time, end_time, 
         source_type, source_name, entry_point_name, lifecycle_stage)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
    ''', (run_uuid, target_exp_id, user_id, status, start_time, end_time, 
          source_type, source_name, entry_point))
    
    # Copy parameters
    source_cursor.execute('SELECT key, value FROM params WHERE run_uuid = ?', (run_uuid,))
    params = source_cursor.fetchall()
    for key, value in params:
        target_cursor.execute(
            'INSERT OR IGNORE INTO params (run_uuid, key, value) VALUES (?, ?, ?)',
            (run_uuid, key, value)
        )
    print(f"  ✓ Copied {len(params)} parameters")
    
    # Copy metrics
    source_cursor.execute('SELECT key, value, timestamp, step, is_nan FROM metrics WHERE run_uuid = ?', (run_uuid,))
    metrics = source_cursor.fetchall()
    for key, value, timestamp, step, is_nan in metrics:
        target_cursor.execute(
            'INSERT OR IGNORE INTO metrics (run_uuid, key, value, timestamp, step, is_nan) VALUES (?, ?, ?, ?, ?, ?)',
            (run_uuid, key, value, timestamp, step, is_nan)
        )
    print(f"  ✓ Copied {len(metrics)} metrics")
    
    # Copy tags
    source_cursor.execute('SELECT key, value FROM tags WHERE run_uuid = ?', (run_uuid,))
    tags = source_cursor.fetchall()
    for key, value in tags:
        target_cursor.execute(
            'INSERT OR IGNORE INTO tags (run_uuid, key, value) VALUES (?, ?, ?)',
            (run_uuid, key, value)
        )
    print(f"  ✓ Copied {len(tags)} tags")
    
    target_db.commit()
    print(f"✓ Run inserted successfully!")
    run_exists = False

# Verify the run is in target DB
target_cursor.execute('''
    SELECT COUNT(*) FROM runs 
    WHERE experiment_id = ? AND run_uuid = ?
''', (target_exp_id, run_uuid))

final_count = target_cursor.fetchone()[0]
if final_count > 0:
    print(f"\n✓ CONSOLIDATION SUCCESSFUL")
    print(f"  Production run now in: mlruns/mlflow.db")
    print(f"  Total runs in mlruns/mlflow.db: {final_count}")
else:
    print(f"\n✗ CONSOLIDATION FAILED")
    exit(1)

source_db.close()
target_db.close()

print("\n" + "="*80)
print("NEXT STEPS:")
print("="*80)
print("""
1. Start MLflow UI pointing to mlruns/mlflow.db:
   mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

2. Check experiments:
   Navigate to http://localhost:5000/#/experiments
   
3. View production run:
   - Go to: credit_scoring_final_delivery
   - Click: production_lightgbm_189features_final
   - Check: Parameters, Metrics, Artifacts tabs
""")
