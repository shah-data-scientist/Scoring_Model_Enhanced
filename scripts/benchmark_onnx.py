import time
import pickle
import numpy as np
import onnxruntime as rt
from pathlib import Path

MODEL_PATH = Path("models/production_model.pkl")
ONNX_PATH = Path("models/production_model.onnx")
N_ITER = 1000
N_FEATURES = 189

def benchmark():
    # Load Pickle
    print("Loading Pickle model...")
    with open(MODEL_PATH, "rb") as f:
        pickle_model = pickle.load(f)
        
    # Load ONNX
    print("Loading ONNX model...")
    sess = rt.InferenceSession(str(ONNX_PATH))
    input_name = sess.get_inputs()[0].name
    
    # Data
    input_data = np.random.rand(1, N_FEATURES).astype(np.float32)
    
    # Warmup
    pickle_model.predict_proba(input_data)
    sess.run(None, {input_name: input_data})
    
    # Benchmark Pickle
    print(f"Benchmarking Pickle ({N_ITER} iters)...")
    start = time.time()
    for _ in range(N_ITER):
        pickle_model.predict_proba(input_data)
    pickle_time = (time.time() - start) / N_ITER * 1000 # ms
    
    # Benchmark ONNX
    print(f"Benchmarking ONNX ({N_ITER} iters)...")
    start = time.time()
    for _ in range(N_ITER):
        sess.run(None, {input_name: input_data})
    onnx_time = (time.time() - start) / N_ITER * 1000 # ms
    
    print(f"\n--- Results (Avg over {N_ITER} runs) ---")
    print(f"Pickle (Sklearn): {pickle_time:.4f} ms/request")
    print(f"ONNX Runtime:     {onnx_time:.4f} ms/request")
    print(f"Speedup:          {pickle_time / onnx_time:.2f}x")

if __name__ == "__main__":
    benchmark()
