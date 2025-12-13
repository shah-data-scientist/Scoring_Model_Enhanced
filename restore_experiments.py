"""Restore all experiments to active"""
import sqlite3

conn = sqlite3.connect('mlruns/mlflow.db')
cursor = conn.cursor()

print("="*80)
print("RESTORING ALL EXPERIMENTS")
print("="*80)

# Show current state
print("\nBEFORE:")
cursor.execute('''
    SELECT experiment_id, name, lifecycle_stage, 
           (SELECT COUNT(*) FROM runs WHERE runs.experiment_id = experiments.experiment_id) as run_count
    FROM experiments
    ORDER BY experiment_id
''')
for exp_id, name, stage, run_count in cursor.fetchall():
    print(f"  Exp {exp_id}: {run_count:2d} runs - {name[:40]:40s} [{stage.upper()}]")

# Restore all experiments
cursor.execute("""
    UPDATE experiments 
    SET lifecycle_stage = 'active' 
    WHERE experiment_id IN (1, 2, 3, 5, 6)
""")
conn.commit()
restored_count = cursor.rowcount

print(f"\n✓ Restored {restored_count} experiments to ACTIVE")

# Show new state
print("\nAFTER:")
cursor.execute('''
    SELECT experiment_id, name, lifecycle_stage,
           (SELECT COUNT(*) FROM runs WHERE runs.experiment_id = experiments.experiment_id) as run_count
    FROM experiments
    WHERE lifecycle_stage = 'active'
    ORDER BY experiment_id
''')
for exp_id, name, stage, run_count in cursor.fetchall():
    print(f"  Exp {exp_id}: {run_count:2d} runs - {name[:40]:40s} [{stage.upper()}]")

total_runs = sum(row[3] for row in cursor.execute('''
    SELECT experiment_id, name, lifecycle_stage,
           (SELECT COUNT(*) FROM runs WHERE runs.experiment_id = experiments.experiment_id) as run_count
    FROM experiments
    WHERE lifecycle_stage = 'active'
''').fetchall())

print(f"\n✓ Total active runs: {total_runs}")

conn.close()

print("\n" + "="*80)
print("COMPLETE - All experiments restored")
print("="*80)
print("\nRefresh MLflow UI to see all 67 runs across 6 experiments")
