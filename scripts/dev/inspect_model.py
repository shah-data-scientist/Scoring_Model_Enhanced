import pickle
import sys
from pathlib import Path

try:
    model_path = Path("models/production_model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    print(f"Model Type: {type(model)}")
    print(f"Model Class: {model.__class__.__name__}")
    if hasattr(model, 'steps'):
        print("Pipeline Steps:")
        for name, step in model.steps:
            print(f"  - {name}: {type(step)}")

except Exception as e:
    print(f"Error: {e}")
