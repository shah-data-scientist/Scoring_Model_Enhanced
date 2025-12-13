# Anonymization Test Results: SK_ID_CURR Impact on Predictions

## Executive Summary

**RESULT: Predictions DIFFER when SK_ID_CURR is anonymized** ⚠️

However, this is **NOT because SK_ID_CURR is a model feature**. It's because the API uses a **precomputed feature cache** that maps SK_ID_CURRs to preprocessed features.

## Test Results

### Test 1: Full Dataset Anonymization (50 clients)
- **Original data**: 50 predictions
- **Anonymized data** (IDs changed to 900000-999999 range): 50 predictions
- **Binary predictions match**: 45/50 (90%)
- **Prediction mismatches**: 5 cases switched classification (0→1 or 1→0)
- **Probability differences**: 0.004 to 0.145 (mean: 0.037)

### Test 2: Determinism Check (Same data twice)
- **Run 1 vs Run 2**: 100% identical
- **Max difference**: 0.0 (perfect match)
- **Conclusion**: API is perfectly deterministic for same inputs

### Test 3: Single Client, Multiple IDs
```
Same client data with 3 different SK_ID_CURRs:

SK_ID_CURR   Prediction   Probability   Risk Level
111761       0            0.11346042    LOW        (original ID)
888888       0            0.16268797    LOW        (new ID)
999999       0            0.16268797    LOW        (new ID)
```

**Key Finding**: 
- Original ID (111761): prob=0.113
- New IDs (888888, 999999): prob=0.163 **(identical to each other)**
- **Conclusion**: New IDs produce identical predictions (deterministic), but differ from original

## Root Cause Analysis

### The Precomputed Feature Cache

**Location**: [api/preprocessing_pipeline.py](api/preprocessing_pipeline.py#L40-L80)

```python
def __init__(self, feature_names_path: Path = None, use_precomputed: bool = True):
    """Initialize preprocessing pipeline.

    Args:
        use_precomputed: If True, use precomputed features for known applications
    """
    self.use_precomputed = use_precomputed
    self.precomputed_features = None  # Loads cached features for known SK_ID_CURRs
```

### How It Works

```
┌─────────────────────────────────────────────────────────────┐
│  API Batch Prediction Flow                                  │
└─────────────────────────────────────────────────────────────┘

 Input: application.csv with SK_ID_CURRs
         ↓
    ┌───────────────────────────────────────┐
    │ Check if SK_ID_CURR in cache          │
    └───────────────────────────────────────┘
         ↓                    ↓
    ┌─────────┐         ┌──────────────────┐
    │ CACHED  │         │  UNCACHED        │
    │ (Fast)  │         │  (Live compute)  │
    └─────────┘         └──────────────────┘
         │                      │
         │   Merge bureau       │
         │   aggregations       │
         │   with application   │
         │   features           │
         │                      │
         ↓                      ↓
    ┌─────────────────────────────────┐
    │  189 features ready for model   │
    └─────────────────────────────────┘
         ↓
    Model.predict_proba()
```

**Key Insight**: 
- **Known SK_ID_CURRs** (e.g., 111761 from test data): Use **precomputed training features**
- **Unknown SK_ID_CURRs** (e.g., 888888, 999999): Use **live-computed features** from raw CSV merges

### Why Features Differ

**Precomputed features** (training time):
- Computed in batch during model training
- Aggregations done on full datasets
- Potential for different sorting/merge orders
- Cached as-is from training environment

**Live-computed features** (API time):
- Computed from uploaded CSV files
- Merges happen in API request context
- Different code path through preprocessing
- Potentially different aggregation results

### Example: Bureau Aggregations

```python
# Training: bureau.csv has ALL clients
bureau_agg = bureau.groupby('SK_ID_CURR').agg({
    'DAYS_CREDIT': ['mean', 'max', 'min'],  # Computed from all records
    'AMT_CREDIT_SUM': ['sum', 'mean']
})

# API: bureau.csv has ONLY uploaded clients  
bureau_agg = bureau.groupby('SK_ID_CURR').agg({
    'DAYS_CREDIT': ['mean', 'max', 'min'],  # Computed from subset
    'AMT_CREDIT_SUM': ['sum', 'mean']
})
```

Even with identical source data, subtle differences can arise from:
- Merge strategies (left vs inner join)
- Handling of NaNs during aggregation
- Float precision in intermediate calculations
- Order of operations in feature engineering

## Why This Matters

### ✅ Good News:
1. **SK_ID_CURR is NOT a model feature** - it's just an identifier
2. **API is deterministic** - same input always produces same output
3. **Cache is intentional** - designed for performance optimization

### ⚠️ Concerns:
1. **Inconsistent predictions** for same underlying data
2. **Production risk**: Real clients would get cached features, but new applications get live features
3. **Reproducibility**: Training predictions ≠ API predictions for same SK_ID_CURR

### 🔍 Impact on Your Question:
**"Do results for test data correspond to submission.csv?"**

**Answer: YES, but with caveats**:
- **For SK_ID_CURRs in training/test set**: API uses precomputed features → matches submission.csv
- **For NEW SK_ID_CURRs** (anonymized or real new applications): API computes live features → may differ slightly

## Recommendations

### Option 1: Disable Precomputed Cache (Most Consistent)
```python
# In api/app.py
pipeline = PreprocessingPipeline(use_precomputed=False)
```

**Pros**:
- ✅ Consistent predictions regardless of SK_ID_CURR
- ✅ True end-to-end preprocessing validation
- ✅ Matches production behavior for new applications

**Cons**:
- ⚠️ Slower (all features computed live)
- ⚠️ Different from training environment

### Option 2: Keep Cache, Document Behavior (Current State)
**Pros**:
- ✅ Fast predictions for known IDs
- ✅ Matches training predictions exactly

**Cons**:
- ⚠️ Two-tier system (cached vs live)
- ⚠️ Anonymization breaks predictions
- ⚠️ New applications may differ from test set

### Option 3: Regenerate Cache from API Pipeline
```python
# Run preprocessing pipeline on all test data
# Save results as new precomputed cache
# Ensures cache matches live computation
```

**Pros**:
- ✅ Best of both worlds (speed + consistency)
- ✅ Cache matches live computation

**Cons**:
- ⚠️ Requires one-time regeneration
- ⚠️ Must maintain cache consistency

## Conclusion

**Your anonymized predictions differ because the API uses a precomputed feature cache.**

When you anonymize SK_ID_CURR:
- Known IDs → Cached features from training
- New IDs → Live-computed features from CSV files

This creates **two preprocessing paths** that can produce **slightly different features** due to:
- Different aggregation contexts
- Different merge operations
- Different execution environments

**The good news**: This proves SK_ID_CURR is NOT a model feature - it's just used for cache lookup.

**The concern**: For true production deployment, all clients should use the same preprocessing path (live computation) to ensure consistency.

## Files Created During Testing

- `data/end_user_tests_anonymized/` - Anonymized test data (50 clients with IDs 900000-999999)
- `data/test_single_client/` - Single client with 3 different IDs (111761, 888888, 999999)
- `test_anonymization.py` - Full dataset anonymization test
- `test_determinism.py` - API determinism validation
- `test_single_client.py` - Single client ID sensitivity test
- `compare_predictions.py` - Comparison with submission.csv

## Clean-up Commands

```powershell
# Remove test directories
Remove-Item 'data\end_user_tests_anonymized' -Recurse -Force
Remove-Item 'data\test_single_client' -Recurse -Force

# Remove test scripts
Remove-Item 'test_*.py', 'compare_predictions.py', 'investigate_anonymization.py', 'debug_features.py'
```
