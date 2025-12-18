import pickle
import json
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import textwrap
from scipy.special import expit # Sigmoid function
from api.preprocessing_pipeline import PreprocessingPipeline

# Config
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
CONFIG_DIR = PROJECT_ROOT / "config"
OUTPUT_DIR = PROJECT_ROOT / "shap_experiments"
OUTPUT_DIR.mkdir(exist_ok=True)

REAL_ID = 118298
DISPLAY_ID = 112233

def get_feature_source_short(name, raw_features):
    agg_prefixes = ["BUREAU_", "PREV_", "POS_", "CC_", "INST_"]
    if name in raw_features: return "RW"
    elif name.startswith("POLY_"): return "PL"
    elif any(name.startswith(pre) for pre in agg_prefixes): return "AG"
    else: return "DM"

def generate_probability_waterfall():
    # 1. Load Model and Data
    print(f"Loading model and calculating for Client {DISPLAY_ID}...")
    with open(MODELS_DIR / "production_model.pkl", 'rb') as f:
        model = pickle.load(f)
    
    with open(CONFIG_DIR / "model_features.txt") as f:
        feature_names = [line.strip() for line in f if line.strip()]
    
    with open(CONFIG_DIR / "all_raw_features.json") as f:
        raw_features_config = json.load(f)
        raw_feature_list = raw_features_config.get("application.csv", [])

    df_mv = pd.read_parquet(DATA_DIR / "precomputed_features.parquet")
    mv_row = df_mv[df_mv['SK_ID_CURR'] == REAL_ID].iloc[0]
    X_mv = mv_row[feature_names].to_frame().T
    
    # 2. Probability and RV Calculation
    prob = model.predict_proba(X_mv)[0, 1]
    
    print("Calculating Real Values (RV)...")
    dataframes = {}
    for csv_file in SAMPLES_DIR.glob("*.csv"):
        df_temp = pd.read_csv(csv_file)
        if 'SK_ID_CURR' in df_temp.columns:
            dataframes[csv_file.name] = df_temp[df_temp['SK_ID_CURR'] == REAL_ID]
        elif csv_file.name == 'previous_application.csv':
             dataframes[csv_file.name] = df_temp[df_temp['SK_ID_CURR'] == REAL_ID]

    pipeline = PreprocessingPipeline(use_precomputed=False)
    
    # Fixed call with keyword arguments to avoid positional mismatch
    df_rv = pipeline.aggregate_data(
        application_df=dataframes.get('application.csv'),
        bureau_df=dataframes.get('bureau.csv'),
        previous_application_df=dataframes.get('previous_application.csv')
    )
    df_rv = pipeline.create_engineered_features(df_rv)
    df_rv = pipeline.encode_and_clean(df_rv)
    df_rv = pipeline.align_features(df_rv)
    rv_row = df_rv.iloc[0]

    # 3. Generate SHAP
    explainer = shap.TreeExplainer(model)
    shap_values_obj = explainer(X_mv)
    instance_shap = shap_values_obj[0]
    
    # Base probability calculation (log-odds to probability)
    base_prob = expit(explainer.expected_value)

    # 4. Plotting
    max_display = 12
    feature_order = np.argsort(np.abs(instance_shap.values))
    top_indices = feature_order[-max_display:]
    
    fig = plt.figure(figsize=(18, 14))
    ax = fig.add_subplot(111)
    plt.subplots_adjust(left=0.4) 
    
    shap.plots.waterfall(instance_shap, max_display=max_display, show=False)
    ax = plt.gca()
    ax.set_yticklabels([])
    y_ticks = ax.get_yticks()
    
    for i, idx in enumerate(top_indices):
        name = feature_names[idx]
        short_type = get_feature_source_short(name, raw_feature_list)
        rv = rv_row[name]
        rv_str = f"{rv:.2f}" if not pd.isna(rv) else "NA"
        
        clean_name = name.replace("_", " ")
        wrapped_name = "\n".join(textwrap.wrap(clean_name, width=28))
        y_pos = y_ticks[i+1] if len(y_ticks) > max_display else y_ticks[i]
        
        ax.text(-0.05, y_pos, wrapped_name, transform=ax.get_yaxis_transform(), ha='right', va='center', fontweight='bold', fontsize=12)
        
        tick_spacing = y_ticks[1] - y_ticks[0]
        y_mid = y_pos - (tick_spacing * 0.45)
        if i > 0 or len(y_ticks) > max_display:
            ax.text(-0.05, y_mid, f"[{short_type}] RV={rv_str}", transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=10, color='#666666', fontstyle='italic')

    plt.title(f"Feature Explanation - Client {DISPLAY_ID}\nPredicted Probability: {prob:.1%}", 
              fontsize=20, pad=40, fontweight='bold')
    
    plt.figtext(0.1, 0.02, f"Types: RW=Raw, AG=Aggregate, PL=Polynomial, DM=Domain | Base Prob: {base_prob:.1%}", 
                fontsize=10, color='#333333', style='italic')
    
    output_path = OUTPUT_DIR / f"final_prob_waterfall_{DISPLAY_ID}.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Successfully generated plot at {output_path}")

if __name__ == "__main__":
    generate_probability_waterfall()