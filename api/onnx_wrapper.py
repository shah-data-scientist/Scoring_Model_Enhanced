import numpy as np
import onnxruntime as rt

class ONNXModelWrapper:
    """Wrapper for ONNX Runtime session to mimic Sklearn API.

    Optimized for fast batch inference with:
    - Multi-threaded execution
    - Memory-efficient data types
    - Pre-allocated output buffers
    """

    def __init__(self, model_path: str):
        # Configure session options for performance
        sess_options = rt.SessionOptions()
        sess_options.intra_op_num_threads = 4  # Parallel ops within nodes
        sess_options.inter_op_num_threads = 2  # Parallel execution of nodes
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Try to use CPU execution provider with optimizations
        providers = ['CPUExecutionProvider']

        # Load session with optimizations
        self.sess = rt.InferenceSession(model_path, sess_options, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name

        # Inspect outputs
        outputs = self.sess.get_outputs()
        self.output_label = outputs[0].name
        self.output_proba = outputs[1].name if len(outputs) > 1 else None

        # Mock attributes for compatibility
        input_shape = self.sess.get_inputs()[0].shape
        self.n_features_in_ = input_shape[1] if len(input_shape) > 1 and isinstance(input_shape[1], int) else 189

        print(f"ONNX Model loaded (optimized). Input: {self.input_name}, Features: {self.n_features_in_}")
        
    def predict(self, X):
        """Predict class labels."""
        X = self._prepare_input(X)
        results = self.sess.run([self.output_label], {self.input_name: X})
        return results[0]

    def predict_proba(self, X):
        """Predict class probabilities."""
        if not self.output_proba:
            raise NotImplementedError("Model does not output probabilities")

        X = self._prepare_input(X)
        results = self.sess.run([self.output_proba], {self.input_name: X})
        probs = results[0]

        # Handle ZipMap output (list of dicts) common in sklearn-onnx
        if isinstance(probs, list) and len(probs) > 0 and isinstance(probs[0], dict):
            return self._convert_zipmap_to_array(probs)

        return probs

    def _prepare_input(self, X):
        """Prepare input array for inference (optimized)."""
        if not isinstance(X, np.ndarray):
            X = np.asarray(X, dtype=np.float32)
        elif X.dtype != np.float32:
            X = X.astype(np.float32, copy=False)

        # Ensure C-contiguous for optimal memory access
        if not X.flags['C_CONTIGUOUS']:
            X = np.ascontiguousarray(X)

        return X

    def _convert_zipmap_to_array(self, probs):
        """Convert ZipMap output to numpy array (vectorized)."""
        n_samples = len(probs)
        prob_array = np.empty((n_samples, 2), dtype=np.float32)

        # Determine key type from first element
        first_keys = set(probs[0].keys())
        use_int_keys = 0 in first_keys or 1 in first_keys

        for i, p_map in enumerate(probs):
            if use_int_keys:
                prob_array[i, 0] = p_map.get(0, 0.0)
                prob_array[i, 1] = p_map.get(1, 0.0)
            else:
                prob_array[i, 0] = p_map.get('0', 0.0)
                prob_array[i, 1] = p_map.get('1', 0.0)

        return prob_array
