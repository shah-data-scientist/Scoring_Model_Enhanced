"""Check production model structure"""
import joblib
import os
from pathlib import Path

os.chdir('mlruns/7c/7ce7c8f6371e43af9ced637e5a4da7f0/artifacts')

print("Loading production_model.pkl...")
model = joblib.load('production_model.pkl')

print(f"\n1. Model type: {type(model)}")
print(f"2. Model class: {model.__class__.__name__}")

# Check if it's a pipeline
if hasattr(model, 'named_steps'):
    print(f"3. IS A PIPELINE")
    print(f"   Steps: {list(model.named_steps.keys())}")
    for step_name, step_obj in model.named_steps.items():
        print(f"   - {step_name}: {type(step_obj).__name__}")
else:
    print(f"3. NOT A PIPELINE - Raw LightGBM model")

# Check for encoder attributes
if hasattr(model, 'encoder') or hasattr(model, 'label_encoder'):
    print("4. HAS encoder attribute")
else:
    print("4. NO encoder attribute")

# Check attributes
print(f"\n5. Model attributes:")
attrs = [a for a in dir(model) if not a.startswith('_')]
for attr in attrs[:20]:  # First 20 non-private attributes
    print(f"   - {attr}")
