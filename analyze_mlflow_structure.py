import sqlite3
import json
from pathlib import Path
from collections import defaultdict

print("=" * 80)
print("COMPREHENSIVE MLflow EXPERIMENT ANALYSIS")
print("=" * 80)

conn = sqlite3.connect('mlruns/mlflow.db')
cursor = conn.cursor()

# Get all experiments
cursor.execute('''
    SELECT experiment_id, name, lifecycle_stage, artifact_location 
    FROM experiments 
    ORDER BY experiment_id
''')
experiments = cursor.fetchall()

print("\n" + "="*80)
print("ALL EXPERIMENTS")
print("="*80)
exp_dict = {}
for exp in experiments:
    exp_id, exp_name, stage, artifact_loc = exp
    exp_dict[exp_id] = {
        'name': exp_name,
        'stage': stage,
        'artifact_location': artifact_loc,
        'runs': []
    }
    print(f"ID: {exp_id:3d} | Name: {exp_name:45s} | Stage: {stage:10s}")

# Get all runs with status
print("\n" + "="*80)
print("ALL RUNS BY EXPERIMENT")
print("="*80)

cursor.execute('''
    SELECT run_uuid, experiment_id, name, status, start_time, end_time, lifecycle_stage
    FROM runs 
    ORDER BY experiment_id, start_time DESC
''')
runs = cursor.fetchall()

for run in runs:
    run_id, exp_id, name, status, start_time, end_time, lifecycle_stage = run
    exp_dict[exp_id]['runs'].append({
        'run_id': run_id,
        'name': name,
        'status': status,
        'lifecycle_stage': lifecycle_stage
    })

for exp_id, exp_info in exp_dict.items():
    if exp_info['runs']:
        print(f"\nExperiment {exp_id}: {exp_info['name']} ({exp_info['stage']})")
        print(f"  Lifecycle stage: {exp_info['stage']}")
        for run in exp_info['runs']:
            print(f"    - {run['name']:40s} (Status: {run['status']:10s}, Stage: {run['lifecycle_stage']})")

# Analyze metrics and parameters
print("\n" + "="*80)
print("RUN PARAMETERS AND METRICS ANALYSIS")
print("="*80)

for exp_id, exp_info in exp_dict.items():
    if exp_info['runs']:
        print(f"\n{'='*60}")
        print(f"Experiment {exp_id}: {exp_info['name']}")
        print(f"{'='*60}")
        
        for run in exp_info['runs']:
            run_id = run['run_id']
            print(f"\n  Run: {run['name']}")
            
            # Parameters
            cursor.execute('SELECT key, value FROM params WHERE run_uuid = ? ORDER BY key', (run_id,))
            params = cursor.fetchall()
            if params:
                print(f"    Parameters ({len(params)}):")
                for key, value in params[:5]:  # Show first 5
                    print(f"      - {key}: {value}")
                if len(params) > 5:
                    print(f"      ... and {len(params)-5} more")
            
            # Metrics
            cursor.execute('SELECT key, value FROM metrics WHERE run_uuid = ? ORDER BY key', (run_id,))
            metrics = cursor.fetchall()
            if metrics:
                print(f"    Metrics ({len(metrics)}):")
                for key, value in metrics[:5]:  # Show first 5
                    print(f"      - {key}: {value}")
                if len(metrics) > 5:
                    print(f"      ... and {len(metrics)-5} more")
            
            # Tags
            cursor.execute('SELECT key, value FROM tags WHERE run_uuid = ? ORDER BY key', (run_id,))
            tags = cursor.fetchall()
            if tags:
                print(f"    Tags ({len(tags)}):")
                for key, value in tags:
                    print(f"      - {key}: {value}")

conn.close()

# Check artifacts in file system
print("\n" + "="*80)
print("ARTIFACTS IN FILE SYSTEM")
print("="*80)

mlruns_path = Path('mlruns')
artifact_count = 0

for exp_dir in sorted(mlruns_path.iterdir()):
    if not exp_dir.is_dir() or exp_dir.name in ['.trash', 'models']:
        continue
    
    exp_name = exp_dir.name
    for run_dir in sorted(exp_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        
        artifacts_dir = run_dir / 'artifacts'
        if artifacts_dir.exists():
            files = list(artifacts_dir.rglob('*'))
            files = [f for f in files if f.is_file()]
            if files:
                artifact_count += 1
                print(f"\nExp {exp_name} / Run {run_dir.name}:")
                for f in files:
                    rel_path = f.relative_to(artifacts_dir)
                    size = f.stat().st_size
                    print(f"  - {rel_path} ({size:,} bytes)")

if artifact_count == 0:
    print("NO ARTIFACTS FOUND IN FILE SYSTEM!")

# Identify duplicate experiments
print("\n" + "="*80)
print("DUPLICATE ANALYSIS")
print("="*80)

exp_names = {}
for exp_id, exp_info in exp_dict.items():
    exp_name = exp_info['name']
    if exp_name not in exp_names:
        exp_names[exp_name] = []
    exp_names[exp_name].append(exp_id)

duplicates = {name: ids for name, ids in exp_names.items() if len(ids) > 1}

if duplicates:
    print("FOUND DUPLICATES:")
    for name, ids in duplicates.items():
        print(f"  {name}: IDs {ids}")
else:
    print("No exact duplicate experiment names found.")

# Identify experiments that are sub-phases of each other
print("\n" + "="*80)
print("RELATED EXPERIMENTS (Different phases of same work)")
print("="*80)

related_groups = defaultdict(list)
for exp_id, exp_info in exp_dict.items():
    name = exp_info['name'].lower()
    
    # Group by base name
    if 'credit_scoring' in name:
        # Extract the main subject
        for keyword in ['model_selection', 'feature_engineering', 'optimization', 'final_delivery', 'production']:
            if keyword in name:
                related_groups[keyword].append((exp_id, exp_info['name']))
                break

for keyword, exps in sorted(related_groups.items()):
    if exps:
        print(f"\n{keyword.upper()}:")
        for exp_id, exp_name in exps:
            print(f"  ID {exp_id}: {exp_name}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total Experiments: {len(exp_dict)}")
print(f"Total Runs: {sum(len(e['runs']) for e in exp_dict.values())}")
print(f"Experiments with artifacts in filesystem: {artifact_count}")
print(f"Duplicate experiment names: {len(duplicates)}")
