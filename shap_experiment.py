import os
import sys
import pickle
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# Project configuration
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
CONFIG_DIR = PROJECT_ROOT / "config"
OUTPUT_DIR = PROJECT_ROOT / "shap_experiments"
OUTPUT_DIR.mkdir(exist_ok=True)

def experiment_shap():
    print("--- SHAP Experimentation Script ---")
    
    # 1. Load Model
    model_path = MODELS_DIR / "production_model.pkl"
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # 2. Load Feature Names
    print("Loading feature names...")
    with open(CONFIG_DIR / "model_features.txt") as f:
        feature_names = [line.strip() for line in f if line.strip()]
    
    # 3. Load Sample Data
    print("Loading sample data (precomputed features)...")
    data_path = DATA_DIR / "precomputed_features.parquet"
    df = pd.read_parquet(data_path)
    
    # Ensure columns match and SK_ID_CURR is dropped if present
    if 'SK_ID_CURR' in df.columns:
        df = df.set_index('SK_ID_CURR')
    
    # Take a sample for faster processing
    X = df[feature_names].head(100)
    print(f"Using {len(X)} samples for SHAP calculation.")
    
    # 4. Initialize SHAP Explainer
    print("Initializing SHAP Explainer...")
    # For LightGBM, TreeExplainer is best
    explainer = shap.TreeExplainer(model)
    
    # 5. Calculate SHAP Values
    print("Calculating SHAP values...")
    shap_values = explainer(X)
    
    # 6. Experiment with different plots
    
    # A. Summary Plot (Standard)
    print("Generating Summary Plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_summary_plot.png")
    plt.close()
    
    # B. Waterfall Plot (Individual)
    print("Generating Waterfall Plot for first sample...")
    plt.figure(figsize=(10, 6))
    # Note: shap_values[0] is for the first instance
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_waterfall_plot.png")
    plt.close()
    
    # C. Force Plot (Individual) - Saving as HTML since it's interactive
    print("Generating Force Plot for first sample...")
    force_plot = shap.force_plot(
        explainer.expected_value, 
        shap_values.values[0], 
        X.iloc[0],
        matplotlib=True,
        show=False
    )
    plt.savefig(OUTPUT_DIR / "03_force_plot_static.png", bbox_inches='tight')
    plt.close()
    
    # D. Bar Plot (Global Importance)
    print("Generating Global Bar Plot...")
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_bar_plot.png")
    plt.close()

    # E. Beeswarm Plot
    print("Generating Beeswarm Plot...")
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_beeswarm_plot.png")
    plt.close()

    print(f"\nExperiment complete. Plots saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    try:
        experiment_shap()
    except Exception as e:
        print(f"Error during experimentation: {e}")
        import traceback
        traceback.print_exc()
