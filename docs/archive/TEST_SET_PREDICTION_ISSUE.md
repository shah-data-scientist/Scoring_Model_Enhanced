# Critical Issue: Test Set Prediction Discrepancies

## Problem Summary

The batch API produces **drastically different predictions** for test set applications compared to the official submission.csv file.

## Evidence

Testing with `data/end_user_tests` (10 test set applications):

| SK_ID | API Prediction | Submission.csv | Difference |
|-------|----------------|----------------|------------|
| 383433 | **29.71%** | **95.38%** | **65.67%** ❌ |
| 394113 | **28.20%** | **94.82%** | **66.62%** ❌ |
| 337338 | **48.47%** | **94.67%** | **46.20%** ❌ |
| 123343 | 27.68% | 1.74% | 25.94% ❌ |
| 274541 | 11.17% | 30.73% | 19.56% ❌ |
| 208662 | 21.47% | 1.59% | 19.88% ❌ |
| 427622 | 20.47% | 1.79% | 18.68% ❌ |
| 430300 | 15.68% | 1.46% | 14.22% ❌ |
| 125031 | 17.65% | 30.73% | 13.07% ❌ |
| 451383 | 28.85% | 30.73% | 1.88% ✓ |

**Average difference: 29.6%** - Completely unacceptable!

## Root Cause

The preprocessing pipeline in [api/preprocessing_pipeline.py](api/preprocessing_pipeline.py) produces **different features** than the preprocessing used to create X_test.csv (which was used for submission.csv).

### Verification Test

For test application 383433:
- ✅ **X_test.csv** (row 38552) → Model prediction: **92.0%**
- ✅ **submission.csv** → Listed value: **95.4%** (close match)
- ❌ **Batch API** → Prediction: **29.7%** (HUGE discrepancy!)

This proves the batch API's preprocessing pipeline is fundamentally different.

## Why This Happened

### Training Set vs Test Set Processing

**Training Set (Working Correctly):**
1. X_train.csv + X_val.csv were created during initial data preprocessing
2. We combined them → precomputed_features.parquet
3. Batch API uses lookup for training IDs → **predictions match correctly**

**Test Set (Broken):**
1. X_test.csv was created during initial data preprocessing
2. Batch API processes test applications through full pipeline from scratch
3. The full pipeline in `api/preprocessing_pipeline.py` is **DIFFERENT** from the original preprocessing
4. Different features → completely different predictions

## The Preprocessing Pipeline Mismatch

### Original Preprocessing (used to create X_test.csv)
Location: Unknown - need to find the original preprocessing script

Likely steps:
1. Load raw data
2. Aggregate auxiliary tables
3. Create domain features (`src.domain_features.create_domain_features`)
4. Apply feature engineering
5. Encode categoricals
6. Impute missing values
7. Align with model features
8. Save as X_test.csv

### Batch API Preprocessing
Location: [api/preprocessing_pipeline.py](api/preprocessing_pipeline.py)

Steps in `_process_full_pipeline`:
1. Aggregate auxiliary tables (`src.feature_aggregation`)
2. Create domain features (`src.domain_features.create_domain_features`)
3. Encode categoricals (`src.feature_engineering.encode_categorical_features`)
4. Apply one-hot encoding
5. Impute missing values (`src.feature_engineering.impute_missing_values`)
6. Align with model features

**Problem:** Even though both use similar steps and the same modules, there are subtle differences in:
- How aggregations are performed
- Which features are created
- How categoricals are encoded (drop_first parameter, categories seen)
- How missing values are imputed
- Feature alignment logic

## Impact

### What Works ✅
- **Training set predictions** (307,511 applications): Using precomputed features from X_train + X_val
- **Known training IDs**: Instant lookup, 100% correct features
- **API performance**: Fast startup, efficient processing

### What's Broken ❌
- **Test set predictions** (48,744 applications): Processed through full pipeline
- **New applications**: Would also go through the broken full pipeline
- **Production deployment**: Cannot be used as-is - predictions are wrong!

## Solutions

### Option 1: Add Precomputed Test Features (Quick Fix)

**Approach:**
1. Load X_test.csv (the correct test features)
2. Combine with precomputed_features.parquet (train features)
3. Now batch API has lookup for BOTH train and test applications

**Pros:**
- Fast fix (< 1 hour)
- Guarantees correct predictions for both train and test
- Maintains fast API performance

**Cons:**
- Only works for known applications (train + test = 356,255 total)
- New applications (not in train/test) still go through broken pipeline
- Doesn't fix the root cause

**Files to modify:**
- `scripts/combine_train_val_to_parquet.py` → Include X_test.csv
- `data/processed/precomputed_features.parquet` → Add test applications

### Option 2: Fix the Preprocessing Pipeline (Proper Solution)

**Approach:**
1. Find the original preprocessing script that created X_train/X_val/X_test
2. Compare with `api/preprocessing_pipeline.py` line by line
3. Identify all differences
4. Update API preprocessing to match original logic exactly
5. Test with end_user_tests to verify predictions match

**Pros:**
- Fixes root cause
- Works for ANY application (train, test, or new)
- Proper long-term solution

**Cons:**
- Time-consuming (requires detailed investigation)
- May reveal complex differences in feature engineering
- Risk of breaking training set predictions during fixes

**Files to investigate:**
- Original preprocessing: Need to find (check `scripts/pipeline/` or `src/`)
- `api/preprocessing_pipeline.py` - Full pipeline implementation
- `src/feature_aggregation.py` - Aggregation logic
- `src/domain_features.py` - Domain feature creation
- `src/feature_engineering.py` - Categorical encoding, imputation

### Option 3: Hybrid Approach (Recommended)

**Immediate (Option 1):**
1. Add X_test.csv to precomputed features
2. Deploy batch API with lookup for all 356,255 known applications
3. Document that new applications require investigation

**Long-term (Option 2):**
1. Investigate and fix preprocessing pipeline differences
2. Validate against both train and test sets
3. Enable processing of truly new applications

## Recommended Next Steps

1. **Immediate (Today):**
   - Run `scripts/combine_train_test_to_parquet.py` (create this)
   - Update precomputed features to include test set
   - Restart API and re-test with end_user_tests
   - Verify all 10 predictions now match submission.csv

2. **Short-term (This Week):**
   - Find original preprocessing script
   - Document preprocessing differences
   - Create comprehensive test suite

3. **Long-term:**
   - Fix preprocessing pipeline to match original
   - Enable processing of new applications
   - Add integration tests

## Current Status

- ✅ Training set predictions: Working correctly (307,511 apps)
- ❌ Test set predictions: Broken (48,744 apps, avg 29.6% error)
- ❌ New applications: Would be broken (unknown count)

**Conclusion:** The batch API cannot be deployed to production in its current state. Option 1 (add test features to lookup) is the fastest path to a working system for known applications.
