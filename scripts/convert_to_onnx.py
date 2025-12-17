import pickle
import numpy as np
import onnxmltools
import onnxruntime as rt
from pathlib import Path
from onnxmltools.convert.common.data_types import FloatTensorType

# Paths
MODEL_PATH = Path("models/production_model.pkl")
ONNX_PATH = Path("models/production_model.onnx")

def convert():
    print("Loading pickle model...")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    print(f"Model type: {type(model)}")
    
    # define initial types
    # LightGBM usually expects float inputs
    # 189 features
    initial_types = [('float_input', FloatTensorType([None, 189]))]

    print("Converting to ONNX...")
    # Use convert_lightgbm for LGBMClassifier
    onnx_model = onnxmltools.convert_lightgbm(model, initial_types=initial_types)
    
    print(f"Saving to {ONNX_PATH}...")
    onnxmltools.utils.save_model(onnx_model, str(ONNX_PATH))
    print("Conversion complete.")

    # Verification
    print("\nVerifying model...")
    
    # Create dummy input
    dummy_input = np.random.rand(1, 189).astype(np.float32)
    
    # Pickle prediction
    pickle_pred = model.predict(dummy_input)
    pickle_proba = model.predict_proba(dummy_input)
    
    # ONNX prediction
    sess = rt.InferenceSession(str(ONNX_PATH))
    input_name = sess.get_inputs()[0].name
    
    # Outputs: label, probabilities
    # For LGBMClassifier, usually:
    # output 0: label (int)
    # output 1: probabilities (zipmap or array)
    
    onnx_results = sess.run(None, {input_name: dummy_input})
    
    onnx_label = onnx_results[0][0]
    onnx_probs = onnx_results[1][0] # This might be a dict {0: prob0, 1: prob1} or array
    
    print(f"Pickle Prediction: {pickle_pred[0]}")
    print(f"ONNX Prediction: {onnx_label}")
    
    if int(pickle_pred[0]) == int(onnx_label):
        print("✅ Predictions match!")
    else:
        print("❌ Predictions do NOT match!")
        
    print(f"Pickle Proba: {pickle_proba[0]}")
    print(f"ONNX Proba: {onnx_probs}")

if __name__ == "__main__":
    convert()
