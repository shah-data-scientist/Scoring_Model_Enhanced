"""
Fix production run: Update name and copy artifacts
"""
import sqlite3
import shutil
from pathlib import Path

print("="*80)
print("FIXING PRODUCTION RUN")
print("="*80)

run_uuid = '7ce7c8f6371e43af9ced637e5a4da7f0'
run_name = 'production_lightgbm_189features_final'

# Fix 1: Update run name in database
print("\n1. Updating run name in database...")
conn = sqlite3.connect('mlruns/mlflow.db')
cursor = conn.cursor()

cursor.execute('UPDATE runs SET name = ? WHERE run_uuid = ?', (run_name, run_uuid))
conn.commit()

cursor.execute('SELECT name FROM runs WHERE run_uuid = ?', (run_uuid,))
updated_name = cursor.fetchone()[0]
print(f"   ✓ Run name updated to: {updated_name}")

conn.close()

# Fix 2: Copy artifacts from root mlruns to mlruns/ directory
print("\n2. Copying artifacts...")

# Source: artifacts were created in root mlruns during create_production_run.py
source_run_dir = Path(f'mlruns/1/{run_uuid}')  # Created in experiment 1 in root db
target_run_dir = Path(f'mlruns/7c/{run_uuid}')  # Should be in mlruns/7c (first 2 chars)

print(f"   Source: {source_run_dir}")
print(f"   Target: {target_run_dir}")

if source_run_dir.exists():
    print(f"   ✓ Source directory found")
    
    # Create target directory
    target_run_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy artifacts folder
    source_artifacts = source_run_dir / 'artifacts'
    target_artifacts = target_run_dir / 'artifacts'
    
    if source_artifacts.exists():
        if target_artifacts.exists():
            shutil.rmtree(target_artifacts)
        shutil.copytree(source_artifacts, target_artifacts)
        
        # Count files
        artifact_files = list(target_artifacts.rglob('*'))
        artifact_files = [f for f in artifact_files if f.is_file()]
        print(f"   ✓ Copied {len(artifact_files)} artifact files")
        for f in artifact_files:
            rel_path = f.relative_to(target_artifacts)
            print(f"     - {rel_path} ({f.stat().st_size} bytes)")
    else:
        print(f"   ✗ Source artifacts not found: {source_artifacts}")
        
    # Copy meta.yaml if it exists
    source_meta = source_run_dir / 'meta.yaml'
    target_meta = target_run_dir / 'meta.yaml'
    if source_meta.exists():
        shutil.copy2(source_meta, target_meta)
        print(f"   ✓ Copied meta.yaml")
        
else:
    print(f"   ✗ Source directory not found")
    print(f"   Searching for artifacts in other locations...")
    
    # Try to find artifacts in mlruns subdirectories
    for exp_dir in Path('mlruns').iterdir():
        if exp_dir.is_dir() and exp_dir.name not in ['.trash', 'models']:
            potential_source = exp_dir / run_uuid
            if potential_source.exists():
                print(f"   ✓ Found run in: {potential_source}")
                source_run_dir = potential_source
                
                # Create target and copy
                target_run_dir.mkdir(parents=True, exist_ok=True)
                source_artifacts = source_run_dir / 'artifacts'
                target_artifacts = target_run_dir / 'artifacts'
                
                if source_artifacts.exists():
                    if target_artifacts.exists():
                        shutil.rmtree(target_artifacts)
                    shutil.copytree(source_artifacts, target_artifacts)
                    
                    artifact_files = list(target_artifacts.rglob('*'))
                    artifact_files = [f for f in artifact_files if f.is_file()]
                    print(f"   ✓ Copied {len(artifact_files)} artifact files from {exp_dir.name}")
                    for f in artifact_files:
                        rel_path = f.relative_to(target_artifacts)
                        print(f"     - {rel_path}")
                break

# Fix 3: Verify
print("\n3. Verifying fixes...")
conn = sqlite3.connect('mlruns/mlflow.db')
cursor = conn.cursor()

cursor.execute('SELECT name FROM runs WHERE run_uuid = ?', (run_uuid,))
final_name = cursor.fetchone()[0]

print(f"   ✓ Run name in DB: {final_name}")

target_artifacts_final = Path(f'mlruns/7c/{run_uuid}/artifacts')
if target_artifacts_final.exists():
    files = list(target_artifacts_final.rglob('*'))
    files = [f for f in files if f.is_file()]
    print(f"   ✓ Artifacts directory exists: {len(files)} files")
else:
    print(f"   ✗ Artifacts directory still missing")

conn.close()

print("\n" + "="*80)
print("DONE - Refresh MLflow UI to see changes")
print("="*80)
