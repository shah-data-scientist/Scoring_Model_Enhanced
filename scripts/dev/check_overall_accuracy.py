"""Check overall prediction accuracy across all samples."""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent

# Load original training predictions
train_preds = pd.read_csv(PROJECT_ROOT / 'results' / 'train_predictions.csv')
app_train = pd.read_csv(PROJECT_ROOT / 'data' / 'application_train.csv')
train_preds['SK_ID_CURR'] = app_train['SK_ID_CURR'].values

# Load precomputed predictions
precomp_preds = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'precomputed_predictions.csv')

# Merge on SK_ID_CURR
merged = train_preds.merge(
    precomp_preds,
    on='SK_ID_CURR',
    suffixes=('_train', '_precomp')
)

# Calculate differences
merged['diff'] = abs(merged['PROBABILITY_train'] - merged['PROBABILITY_precomp'])

print("="*80)
print("OVERALL PREDICTION ACCURACY ANALYSIS")
print("="*80)
print(f"\nTotal samples: {len(merged):,}")
print("\nDifference statistics:")
print(f"  Mean difference: {merged['diff'].mean():.4f} ({merged['diff'].mean()*100:.2f}%)")
print(f"  Median difference: {merged['diff'].median():.4f} ({merged['diff'].median()*100:.2f}%)")
print(f"  Max difference: {merged['diff'].max():.4f} ({merged['diff'].max()*100:.2f}%)")
print(f"  Min difference: {merged['diff'].min():.4f} ({merged['diff'].min()*100:.2f}%)")

# Categorize by accuracy
perfect = (merged['diff'] < 0.0001).sum()
very_close = (merged['diff'] < 0.01).sum()
close = (merged['diff'] < 0.05).sum()
moderate = (merged['diff'] < 0.10).sum()
large = (merged['diff'] >= 0.10).sum()

print("\nAccuracy categories:")
print(f"  Perfect match (< 0.01%): {perfect:,} ({perfect/len(merged)*100:.1f}%)")
print(f"  Very close (< 1%): {very_close:,} ({very_close/len(merged)*100:.1f}%)")
print(f"  Close (< 5%): {close:,} ({close/len(merged)*100:.1f}%)")
print(f"  Moderate (< 10%): {moderate:,} ({moderate/len(merged)*100:.1f}%)")
print(f"  Large (>= 10%): {large:,} ({large/len(merged)*100:.1f}%)")

# Find worst offenders
worst = merged.nlargest(10, 'diff')[['SK_ID_CURR', 'PROBABILITY_train', 'PROBABILITY_precomp', 'diff']]
print("\nWorst 10 mismatches:")
for _, row in worst.iterrows():
    print(f"  ID {int(row['SK_ID_CURR'])}: Train={row['PROBABILITY_train']:.4f}, "
          f"Precomp={row['PROBABILITY_precomp']:.4f}, Diff={row['diff']:.4f}")

print("="*80)
