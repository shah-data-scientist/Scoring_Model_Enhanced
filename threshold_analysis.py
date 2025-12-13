import pandas as pd
import numpy as np

print("=" * 80)
print("THRESHOLD ANALYSIS FROM STATIC PREDICTIONS")
print("=" * 80)

df = pd.read_parquet('results/static_model_predictions.parquet')
print(f"Total predictions: {len(df):,}")
print(f"Positive class (TARGET=1): {(df['TARGET']==1).sum():,} ({(df['TARGET']==1).sum()/len(df)*100:.2f}%)")
print(f"Negative class (TARGET=0): {(df['TARGET']==0).sum():,} ({(df['TARGET']==0).sum()/len(df)*100:.2f}%)")

# Business costs
cost_fn = 10  # False negative cost
cost_fp = 1   # False positive cost

print(f"\nBusiness costs: FN={cost_fn}, FP={cost_fp}")

# Test specific thresholds
test_thresholds = [0.30, 0.33, 0.338, 0.35, 0.40, 0.45, 0.48, 0.50]

print(f"\n{'Threshold':<12}{'Cost':<12}{'FN':<8}{'FP':<8}{'TP':<8}{'TN':<8}{'Precision':<12}{'Recall':<12}{'F1':<12}")
print("="*100)

results = []
for threshold in test_thresholds:
    y_pred = (df['PROBABILITY'] >= threshold).astype(int)
    
    fn = ((df['TARGET'] == 1) & (y_pred == 0)).sum()
    fp = ((df['TARGET'] == 0) & (y_pred == 1)).sum()
    tp = ((df['TARGET'] == 1) & (y_pred == 1)).sum()
    tn = ((df['TARGET'] == 0) & (y_pred == 0)).sum()
    
    cost = fn * cost_fn + fp * cost_fp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    results.append({
        'threshold': threshold,
        'cost': cost,
        'fn': fn,
        'fp': fp,
        'tp': tp,
        'tn': tn,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })
    
    print(f"{threshold:<12.3f}{cost:<12,}{fn:<8,}{fp:<8,}{tp:<8,}{tn:<8,}{precision:<12.4f}{recall:<12.4f}{f1:<12.4f}")

# Find optimal with fine granularity
print("\n" + "="*80)
print("FINDING OPTIMAL THRESHOLD (0.01 steps)")
print("="*80)

thresholds = np.arange(0.01, 1.0, 0.01)
costs = []

for threshold in thresholds:
    y_pred = (df['PROBABILITY'] >= threshold).astype(int)
    fn = ((df['TARGET'] == 1) & (y_pred == 0)).sum()
    fp = ((df['TARGET'] == 0) & (y_pred == 1)).sum()
    cost = fn * cost_fn + fp * cost_fp
    costs.append(cost)

optimal_idx = np.argmin(costs)
optimal_threshold = thresholds[optimal_idx]
optimal_cost = costs[optimal_idx]

print(f"\n>>> OPTIMAL THRESHOLD: {optimal_threshold:.2f}")
print(f">>> OPTIMAL COST: {optimal_cost:,}")

# Show neighboring thresholds
print(f"\nThresholds around optimal:")
for i in range(max(0, optimal_idx-3), min(len(thresholds), optimal_idx+4)):
    t = thresholds[i]
    c = costs[i]
    marker = " <-- OPTIMAL" if i == optimal_idx else ""
    print(f"  {t:.2f}: Cost = {c:,}{marker}")

# Compare to 0.33 (from MLflow run)
print("\n" + "="*80)
print("COMPARISON: 0.33 (MLflow) vs 0.48 (Current Optimal)")
print("="*80)

for t in [0.33, 0.48]:
    y_pred = (df['PROBABILITY'] >= t).astype(int)
    fn = ((df['TARGET'] == 1) & (y_pred == 0)).sum()
    fp = ((df['TARGET'] == 0) & (y_pred == 1)).sum()
    tp = ((df['TARGET'] == 1) & (y_pred == 1)).sum()
    tn = ((df['TARGET'] == 0) & (y_pred == 0)).sum()
    cost = fn * cost_fn + fp * cost_fp
    
    print(f"\nThreshold: {t:.2f}")
    print(f"  Total Cost: {cost:,}")
    print(f"  False Negatives: {fn:,} (cost: {fn*cost_fn:,})")
    print(f"  False Positives: {fp:,} (cost: {fp*cost_fp:,})")
    print(f"  True Positives: {tp:,}")
    print(f"  True Negatives: {tn:,}")
    print(f"  Precision: {tp/(tp+fp):.4f}")
    print(f"  Recall: {tp/(tp+fn):.4f}")

cost_diff = costs[np.where(thresholds == 0.33)[0][0]] - costs[optimal_idx]
print(f"\n>>> Cost difference: Using 0.33 instead of 0.48 increases cost by {cost_diff:,}")

# Check if both models are the same by comparing prediction distributions
print("\n" + "="*80)
print("PROBABILITY DISTRIBUTION ANALYSIS")
print("="*80)
print(f"Min probability: {df['PROBABILITY'].min():.6f}")
print(f"Max probability: {df['PROBABILITY'].max():.6f}")
print(f"Mean probability: {df['PROBABILITY'].mean():.6f}")
print(f"Median probability: {df['PROBABILITY'].median():.6f}")
print(f"\nPercentiles:")
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"  {p}th: {df['PROBABILITY'].quantile(p/100):.6f}")
