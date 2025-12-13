"""Check artifact paths and fix if needed"""
import sqlite3
from pathlib import Path

# Check database artifact_uri
conn = sqlite3.connect('mlruns/mlflow.db')
cursor = conn.cursor()

print("="*80)
print("ARTIFACT PATH INVESTIGATION")
print("="*80)

# Get artifact URI from database
cursor.execute('SELECT artifact_uri FROM runs WHERE run_uuid = ?', 
               ('7ce7c8f6371e43af9ced637e5a4da7f0',))
result = cursor.fetchone()
if result:
    artifact_uri = result[0]
    print(f"\n1. Artifact URI in database:")
    print(f"   {artifact_uri}")
else:
    print("ERROR: Run not found!")
    exit(1)

# Check if artifacts exist on filesystem
artifact_path = Path('mlruns/7c/7ce7c8f6371e43af9ced637e5a4da7f0/artifacts')
print(f"\n2. Filesystem check:")
print(f"   Path: {artifact_path}")
print(f"   Exists: {artifact_path.exists()}")
print(f"   Absolute: {artifact_path.resolve()}")

if artifact_path.exists():
    files = list(artifact_path.glob('*'))
    print(f"\n3. Files found: {len(files)}")
    for f in files:
        print(f"   ✓ {f.name} ({f.stat().st_size:,} bytes)")
else:
    print("\n❌ Artifacts directory NOT FOUND!")

# Check if artifact_uri needs fixing
print(f"\n4. Artifact URI Analysis:")

# Expected artifact_uri for MLflow
expected_uri = './mlruns/7c/7ce7c8f6371e43af9ced637e5a4da7f0/artifacts'

if artifact_uri is None:
    print(f"   ❌ CRITICAL: artifact_uri is NULL in database!")
    print(f"\n5. FIXING artifact_uri...")
    cursor.execute('UPDATE runs SET artifact_uri = ? WHERE run_uuid = ?',
                  (expected_uri, '7ce7c8f6371e43af9ced637e5a4da7f0'))
    conn.commit()
    print(f"   ✓ Set artifact_uri to: {expected_uri}")
elif artifact_uri.startswith('file:///'):
    print("   Format: file:// URI (good)")
    uri_path = artifact_uri.replace('file:///', '')
    print(f"   Points to: {uri_path}")
    print(f"\n5. Artifact URI is correct ✓")
elif artifact_uri.startswith('./'):
    print("   Format: Relative path (good for MLflow)")
    print(f"\n5. Artifact URI is correct ✓")
elif artifact_uri != expected_uri:
    print(f"\n⚠️  ISSUE DETECTED:")
    print(f"   Current:  {artifact_uri}")
    print(f"   Expected: {expected_uri}")
    
    print(f"\n5. FIXING artifact_uri...")
    cursor.execute('UPDATE runs SET artifact_uri = ? WHERE run_uuid = ?',
                  (expected_uri, '7ce7c8f6371e43af9ced637e5a4da7f0'))
    conn.commit()
    print(f"   ✓ Updated to: {expected_uri}")
else:
    print(f"\n5. Artifact URI is correct ✓")

conn.close()

print("\n" + "="*80)
print("RESTART MLFLOW UI for changes to take effect")
print("="*80)
