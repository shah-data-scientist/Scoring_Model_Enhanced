# Prediction Differences Explained

## Summary

The "prediction discrepancies" between the batch API and `train_predictions.csv` are **NOT a bug** - they are **expected and correct** behavior.

## Why Predictions Differ

### train_predictions.csv = Cross-Validation Predictions

From [scripts/pipeline/apply_best_model.py:180-196](scripts/pipeline/apply_best_model.py#L180-L196):

```python
# Generate out-of-fold cross-validation predictions
y_proba_cv = cross_val_predict(
    model, X_full_proc, y_full,
    cv=StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE),
    method='predict_proba'
)[:, 1]

# Save for evaluation
train_preds_df = pd.DataFrame({
    'TARGET': y_full,
    'PROBABILITY': y_proba_cv
})
train_preds_df.to_csv(RESULTS_DIR / 'train_predictions.csv', index=False)
```

**Cross-validation predictions:**
- Data split into N folds (typically 5)
- For each fold:
  - Train model on other 4 folds
  - Predict on this fold (model hasn't seen these samples)
- Combine all out-of-fold predictions
- **Purpose:** Unbiased performance evaluation

**Key point:** Each sample is predicted by a model variant that did NOT see that sample during training.

### Production Model = Trained on ALL Data

From [scripts/pipeline/apply_best_model.py:224-233](scripts/pipeline/apply_best_model.py#L224-L233):

```python
# Train final model on all data
print("\nTraining final model on all data...")
model.fit(X_full_proc, y_full)

# Save as production model
mlflow.sklearn.log_model(
    model,
    "final_model",
    registered_model_name="CreditScoringModel"
)
```

**Production model:**
- Trained on the ENTIRE dataset (X_train + X_val = 307,511 samples)
- Has seen ALL training samples
- Used for actual predictions in production
- **Purpose:** Maximum predictive performance

**Key point:** The production model has seen every training sample, so its predictions will be different (and typically more confident).

## Example Comparison

Sample ID: 100003
- **CV Prediction:** 29.98% (model variant hadn't seen this sample)
- **Production Prediction:** 18.32% (final model has seen this sample)
- **Difference:** 11.66%

This is **expected** because:
1. The CV model is more uncertain (hasn't seen the sample)
2. The production model is more confident (has learned from this sample)

## What Should the Batch API Return?

**The batch API should return PRODUCTION MODEL predictions** because:

✅ **Correct:**
- Production model predictions are what the deployed system will actually predict
- These are the predictions end users will see
- Using the fully-trained model maximizes performance
- Consistent with the API's purpose: make real predictions on new data

❌ **Incorrect:**
- CV predictions were only for evaluation during model development
- They represent models that are deliberately handicapped (missing data)
- Not useful for production decision-making

## Verification Results

### Parquet Conversion ✅
- Combined X_train.csv (215,257) + X_val.csv (92,254) = 307,511 samples
- Features match perfectly: All 189 features identical
- Load time: 11.9s (CSV) → 0.5s (Parquet) = **24.1x faster**

### Batch API Testing ✅
```
[PASS] API Health Check
[PASS] File Validation
[PASS] Batch Prediction
[PASS] Error Handling

Total: 4/4 tests passed
```

### Lookup-Based Preprocessing ✅
- Successfully loads 307,511 precomputed features at startup
- Retrieves correct features for known application IDs
- API startup: < 5 seconds (vs 12+ minutes before)

## Conclusion

The batch API is **working correctly**. The prediction differences are:
1. **Expected** - Different models (CV vs full training)
2. **Intentional** - Production model should use all available data
3. **Appropriate** - Production predictions are what users need

The system is ready for production use. The "discrepancies" are actually evidence that:
- The production model is better trained (saw more data)
- The evaluation was properly done (CV for unbiased metrics)
- The API is using the correct production model

## Additional Notes

If you need to compare API predictions against training data for validation purposes:
1. Use the production model to generate fresh predictions on X_train + X_val
2. Compare those against API predictions (should be identical)
3. DON'T compare against train_predictions.csv (those are CV predictions)

The cross-validation predictions in train_predictions.csv are valuable for:
- Model evaluation metrics (unbiased performance estimates)
- Dashboard visualizations showing realistic model performance
- Understanding model generalization capability

But for actual production predictions, always use the final production model.
