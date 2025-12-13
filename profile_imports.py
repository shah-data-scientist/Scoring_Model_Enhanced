"""Profile import times to identify bottlenecks."""
import time
import sys

def time_import(module_name):
    """Time how long it takes to import a module."""
    start = time.time()
    try:
        __import__(module_name)
        duration = time.time() - start
        print(f"[OK] {module_name:30s} {duration:6.3f}s")
        return duration
    except Exception as e:
        duration = time.time() - start
        print(f"[FAIL] {module_name:30s} {duration:6.3f}s - {e}")
        return duration

print("=" * 70)
print("PROFILING MODULE IMPORT TIMES")
print("=" * 70)

total = 0

print("\n1. CORE STREAMLIT IMPORTS:")
total += time_import('streamlit')

print("\n2. DATA PROCESSING IMPORTS:")
total += time_import('pandas')
total += time_import('numpy')

print("\n3. PLOTTING IMPORTS (HEAVY!):")
total += time_import('matplotlib')
total += time_import('matplotlib.pyplot')
total += time_import('seaborn')

print("\n4. OTHER IMPORTS:")
total += time_import('requests')

print("\n5. PROJECT IMPORTS:")
sys.path.insert(0, '.')
total += time_import('src.config')
total += time_import('backend.models')
total += time_import('backend.auth')
total += time_import('backend.database')

print("\n" + "=" * 70)
print(f"TOTAL IMPORT TIME: {total:.3f}s")
print("=" * 70)
