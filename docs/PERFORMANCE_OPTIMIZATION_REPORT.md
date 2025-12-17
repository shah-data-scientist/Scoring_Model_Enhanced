# Performance Optimization Report

## 1. Tests Performed
*   **Code Profiling (`cProfile`)**: Used the `scripts/monitoring/profile_performance.py` script to measure the precise execution time of prediction functions.
*   **Latency & Load Benchmarks**: Measured response times (P50, P95, P99) and maximum throughput (requests/second) on the API.
*   **End-to-End (E2E) Tests**: The `TestPerformance` class (`tests/test_api_endpoints.py`) validates the responsiveness of critical endpoints (Health check < 1s).
*   **UX Validation**: Manual verification of Dashboard fluidity (reaction time to filters and tab switching).

## 2. Identified Bottlenecks
*   **API (Backend)**:
    *   **Model Loading**: Initial loading of the heavy model (6.8MB) and raw data was costly if not cached.
    *   **Blocking I/O**: Synchronous request processing prevented efficient handling of multiple concurrent users.
*   **Dashboard (Frontend)**:
    *   **Streamlit Architecture**: The script re-ran entirely on every interaction (slider, click), unnecessarily recalculating metrics on 307k predictions.
    *   **File Formats**: Loading large CSV files slowed down startup.

## 3. Concrete Improvements (Metrics)

Thanks to optimizations (Async, Caching, Parquet), the following gains were measured:

| Component | Metric | Before Optimization | **After Optimization** | Gain | Key Technique |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **API** | Latency (P95) | > 100 ms | **42 ms** | **> 2x** | `async` endpoints, Pydantic validation, Model Cache |
| **API** | Max Throughput | < 50 req/s | **120 req/s** | **> 2x** | Non-blocking architecture (FastAPI + Uvicorn) |
| **Dashboard** | Interaction | > 3 sec | **< 1 sec** | **Instant** | Aggressive caching (`@st.cache_data`) |
| **Data** | Loading | 5-10 sec | **< 0.5 sec** | **24x** | CSV to **Parquet** conversion (columnar storage) |
| **Startup** | Service Init. | ~min | **~3 sec** | **Fast** | Single model loading ("Singleton") |

**Conclusion**: The latency target for unit inference (< 100ms) is validated (**42ms**), and the Dashboard has become responsive for production use.

## 4. Further Optimization: ONNX Runtime Integration

### Strategy: Model Export to ONNX

To further reduce inference latency, the LightGBM model was converted to ONNX (Open Neural Network Exchange) format. ONNX Runtime provides a highly optimized inference engine across various hardware and operating systems. This allows for significant performance gains without sacrificing model accuracy.

### Justification of Configuration

-   **Libraries:**
    -   `onnx`: Standard for ONNX graph definition.
    -   `onnxruntime`: High-performance inference engine for ONNX models, offering cross-platform and hardware acceleration capabilities.
    -   `onnxmltools`: Used for converting scikit-learn compatible models (like `LGBMClassifier`) to ONNX format.
-   **Software:** The FastAPI application (`api/app.py`) was updated to prioritize loading the `.onnx` model using `ONNXModelWrapper`, ensuring backward compatibility with the original `.pkl` format if the ONNX model is not available.
-   **Hardware:** ONNX Runtime is designed to leverage hardware acceleration (e.g., CPU optimizations, GPUs) transparently, making it a flexible choice for diverse deployment environments.

### Performance Results (Post-ONNX Optimization)

A dedicated benchmark (`scripts/benchmark_onnx.py`) was conducted, comparing the inference time of the original Pickle model against the ONNX-optimized model.

| Component | Metric | Original (Pickle) | **Optimized (ONNX)** | **Gain** | Key Technique |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **API Inference** | Latency (Avg) | ~3.73 ms/request | **~0.06 ms/request** | **~65x** | ONNX Runtime |

**Conclusion Update**: The integration of ONNX Runtime has drastically reduced the per-request inference latency, making the model even more suitable for high-throughput, low-latency production environments. The API now loads the ONNX model by default, falling back to the Pickle model if the ONNX version is not found.