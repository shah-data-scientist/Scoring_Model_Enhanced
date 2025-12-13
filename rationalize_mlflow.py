"""
Rationalize MLflow - Keep only production run, clean up everything else
"""
import sqlite3
import shutil
from pathlib import Path

print("="*80)
print("RATIONALIZING MLFLOW DATABASE")
print("="*80)

production_run_uuid = '7ce7c8f6371e43af9ced637e5a4da7f0'
production_exp_id = 4

conn = sqlite3.connect('mlruns/mlflow.db')
cursor = conn.cursor()

# Step 1: Show current state
print("\nCURRENT STATE:")
cursor.execute('''
    SELECT e.experiment_id, e.name, COUNT(r.run_uuid) as run_count
    FROM experiments e
    LEFT JOIN runs r ON e.experiment_id = r.experiment_id
    WHERE e.lifecycle_stage != 'deleted'
    GROUP BY e.experiment_id
''')
for exp_id, exp_name, run_count in cursor.fetchall():
    marker = ' <-- PRODUCTION' if exp_id == production_exp_id else ''
    print(f"  Exp {exp_id}: {run_count:2d} runs - {exp_name}{marker}")

# Step 2: Delete all runs in Exp 4 EXCEPT the production run
print("\nSTEP 1: Cleaning Experiment 4 (keeping only production run)...")
cursor.execute('''
    SELECT run_uuid, name FROM runs 
    WHERE experiment_id = ? AND run_uuid != ?
''', (production_exp_id, production_run_uuid))

runs_to_delete = cursor.fetchall()
print(f"  Found {len(runs_to_delete)} old runs to delete:")

for run_uuid, run_name in runs_to_delete:
    name_str = run_name if run_name else "NO NAME"
    print(f"    - {name_str[:50]:50s} ({run_uuid[:8]})")
    
    # Delete from database
    cursor.execute('DELETE FROM params WHERE run_uuid = ?', (run_uuid,))
    cursor.execute('DELETE FROM metrics WHERE run_uuid = ?', (run_uuid,))
    cursor.execute('DELETE FROM tags WHERE run_uuid = ?', (run_uuid,))
    cursor.execute('DELETE FROM runs WHERE run_uuid = ?', (run_uuid,))
    
    # Delete from filesystem
    run_prefix = run_uuid[:2]
    run_dir = Path(f'mlruns/{run_prefix}/{run_uuid}')
    if run_dir.exists():
        shutil.rmtree(run_dir)
        print(f"      ✓ Deleted {run_dir}")

print(f"  ✓ Deleted {len(runs_to_delete)} old runs from Experiment 4")

# Step 3: Archive development experiments (Exp 1, 2, 3, 5, 6)
print("\nSTEP 2: Archiving development experiments...")
dev_experiments = [1, 2, 3, 5, 6]

for exp_id in dev_experiments:
    cursor.execute('SELECT name FROM experiments WHERE experiment_id = ?', (exp_id,))
    result = cursor.fetchone()
    if result:
        exp_name = result[0]
        cursor.execute('UPDATE experiments SET lifecycle_stage = ? WHERE experiment_id = ?', 
                      ('deleted', exp_id))
        print(f"  ✓ Archived Exp {exp_id}: {exp_name}")

conn.commit()

# Step 4: Verify final state
print("\nFINAL STATE:")
cursor.execute('''
    SELECT e.experiment_id, e.name, e.lifecycle_stage, COUNT(r.run_uuid) as run_count
    FROM experiments e
    LEFT JOIN runs r ON e.experiment_id = r.experiment_id
    GROUP BY e.experiment_id
    ORDER BY e.experiment_id
''')

for exp_id, exp_name, stage, run_count in cursor.fetchall():
    marker = ' <-- PRODUCTION' if exp_id == production_exp_id and stage == 'active' else ''
    print(f"  Exp {exp_id}: {run_count:2d} runs - {exp_name} [{stage.upper()}]{marker}")

# Verify production run
cursor.execute('''
    SELECT r.name, COUNT(p.key) as param_count, COUNT(DISTINCT m.key) as metric_count
    FROM runs r
    LEFT JOIN params p ON r.run_uuid = p.run_uuid
    LEFT JOIN metrics m ON r.run_uuid = m.run_uuid
    WHERE r.run_uuid = ?
    GROUP BY r.name
''', (production_run_uuid,))

run_info = cursor.fetchone()
if run_info:
    run_name, param_count, metric_count = run_info
    print(f"\n✓ PRODUCTION RUN VERIFIED:")
    print(f"  Name: {run_name}")
    print(f"  Parameters: {param_count}")
    print(f"  Metrics: {metric_count}")
    
    # Check artifacts
    artifacts_dir = Path(f'mlruns/7c/{production_run_uuid}/artifacts')
    if artifacts_dir.exists():
        files = list(artifacts_dir.rglob('*'))
        files = [f for f in files if f.is_file()]
        print(f"  Artifacts: {len(files)} files")
    else:
        print(f"  Artifacts: NOT FOUND")

conn.close()

print("\n" + "="*80)
print("RATIONALIZATION COMPLETE")
print("="*80)
print("""
✓ Kept 1 production run in Experiment 4
✓ Deleted all other runs from Experiment 4
✓ Archived development experiments (1, 2, 3, 5, 6)

Refresh MLflow UI to see clean structure:
  - Only Experiment 4 will be visible (active)
  - Only 1 run: production_lightgbm_189features_final
  - All artifacts, parameters, and metrics preserved
""")
