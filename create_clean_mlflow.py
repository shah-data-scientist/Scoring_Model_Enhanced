"""
Create a clean, rationalized MLflow database from scratch
This replaces the old database with only the production experiment
"""
import sqlite3
import shutil
from pathlib import Path
from datetime import datetime

print("="*80)
print("CREATING CLEAN MLFLOW DATABASE")
print("="*80)

# Backup the current database
mlruns_db = Path("mlruns/mlflow.db")
if mlruns_db.exists():
    backup_name = f"mlruns/mlflow_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
    shutil.copy(mlruns_db, backup_name)
    print(f"✓ Backed up existing database to: {backup_name}")
    mlruns_db.unlink()
    print(f"✓ Deleted old database")

# Copy the NEW database (from root) to mlruns
root_db = Path("mlflow.db")
if root_db.exists():
    shutil.copy(root_db, mlruns_db)
    print(f"✓ Copied clean database to mlruns/mlflow.db")
else:
    print(f"✗ ERROR: Root database not found at {root_db}")
    exit(1)

# Verify the new database
conn = sqlite3.connect(str(mlruns_db))
cursor = conn.cursor()

# Get experiment count
cursor.execute('SELECT COUNT(*) FROM experiments WHERE lifecycle_stage != "deleted"')
exp_count = cursor.fetchone()[0]

# Get run count
cursor.execute('SELECT COUNT(*) FROM runs')
run_count = cursor.fetchone()[0]

# Check for production run
cursor.execute('''
    SELECT r.run_uuid, r.name, e.name as exp_name
    FROM runs r
    JOIN experiments e ON r.experiment_id = e.experiment_id
    WHERE r.name = 'production_lightgbm_189features_final'
''')
prod_run = cursor.fetchone()

print(f"\n{'='*80}")
print(f"NEW DATABASE STATS")
print(f"{'='*80}")
print(f"  Active Experiments: {exp_count}")
print(f"  Total Runs: {run_count}")

if prod_run:
    run_uuid, run_name, exp_name = prod_run
    print(f"  ✓ Production run: {run_name}")
    print(f"    Experiment: {exp_name}")
    print(f"    Run UUID: {run_uuid}")
    
    # Count parameters, metrics, tags
    cursor.execute('SELECT COUNT(*) FROM params WHERE run_uuid = ?', (run_uuid,))
    params = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM metrics WHERE run_uuid = ?', (run_uuid,))
    metrics = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM tags WHERE run_uuid = ?', (run_uuid,))
    tags = cursor.fetchone()[0]
    
    print(f"    Parameters: {params}")
    print(f"    Metrics: {metrics}")
    print(f"    Tags: {tags}")
else:
    print(f"  ✗ Production run NOT found!")

conn.close()

print(f"\n{'='*80}")
print(f"DONE - Clean database created")
print(f"{'='*80}")
print(f"""
The mlruns/mlflow.db now contains ONLY the production experiment.
Old data has been backed up to mlruns/mlflow_backup_*.db

Next steps:
1. Restart MLflow UI (will automatically use mlruns/mlflow.db)
2. Check if artifacts are visible
3. Add encoder to artifacts if missing
""")
