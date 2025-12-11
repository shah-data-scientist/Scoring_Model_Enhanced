
import pickle
import sys
import pandas as pd
from pathlib import Path

def inspect_pickle(path):
    print(f"Inspecting {path}...")
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        
        print(f"Type: {type(obj)}")
        
        if hasattr(obj, 'steps'):
            print("Object is a Pipeline!")
            print("Steps:")
            for name, step in obj.steps:
                print(f"  - {name}: {type(step)}")
                if hasattr(step, 'mean_') or hasattr(step, 'scale_'):
                    print(f"    (Has scaler attributes!)")
        else:
            print("Object is NOT a Pipeline.")
            if hasattr(obj, 'feature_importances_'):
                print(f"Has feature importances (likely a tree model)")
                
    except Exception as e:
        print(f"Error loading pickle: {e}")

if __name__ == "__main__":
    base = Path("models")
    inspect_pickle(base / "production_model.pkl")
    inspect_pickle(base / "best_xgboost_model.pkl")
