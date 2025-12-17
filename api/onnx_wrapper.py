import numpy as np
import onnxruntime as rt

class ONNXModelWrapper:
    """Wrapper for ONNX Runtime session to mimic Sklearn API."""
    
    def __init__(self, model_path: str):
        # Load session
        self.sess = rt.InferenceSession(model_path)
        self.input_name = self.sess.get_inputs()[0].name
        
        # Inspect outputs
        outputs = self.sess.get_outputs()
        self.output_label = outputs[0].name
        self.output_proba = outputs[1].name if len(outputs) > 1 else None
        
        # Mock attributes for compatibility
        # Check input shape (often [None, n_features])
        input_shape = self.sess.get_inputs()[0].shape
        self.n_features_in_ = input_shape[1] if len(input_shape) > 1 and isinstance(input_shape[1], int) else 189 # Fallback
        
        print(f"ONNX Model loaded. Input: {self.input_name}, Output: {self.output_label}")
        
    def predict(self, X):
        """Predict class labels."""
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        X = X.astype(np.float32)
        
        results = self.sess.run([self.output_label], {self.input_name: X})
        return results[0]
        
    def predict_proba(self, X):
        """Predict class probabilities."""
        if not self.output_proba:
            raise NotImplementedError("Model does not output probabilities")
            
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        X = X.astype(np.float32)
        
        results = self.sess.run([self.output_proba], {self.input_name: X})
        probs = results[0]
        
        # Handle ZipMap output (list of dicts) common in sklearn-onnx
        if isinstance(probs, list) and isinstance(probs[0], dict):
            # Convert to array [[prob_0, prob_1], ...]
            n_samples = len(probs)
            # Assuming binary classification with classes 0 and 1
            # We need to be careful if keys are strings or ints
            prob_array = np.zeros((n_samples, 2))
            
            for i, p_map in enumerate(probs):
                # Try int keys first, then string
                prob_array[i, 0] = p_map.get(0, p_map.get('0', 0.0))
                prob_array[i, 1] = p_map.get(1, p_map.get('1', 0.0))
                
            return prob_array
            
        return probs
