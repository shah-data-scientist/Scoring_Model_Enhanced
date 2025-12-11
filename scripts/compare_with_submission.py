"""Compare end_user_tests predictions with submission.csv."""
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Load end_user_tests predictions
end_user_preds = pd.read_csv(PROJECT_ROOT / 'results' / 'end_user_test_predictions.csv')
print("End User Test Predictions:")
print(f"  Applications: {len(end_user_preds)}")
print(f"  IDs: {sorted(end_user_preds['sk_id_curr'].tolist())}\n")

# Load submission.csv (test set predictions)
submission = pd.read_csv(PROJECT_ROOT / 'results' / 'submission.csv')
print("Submission.csv (Test Set):")
print(f"  Applications: {len(submission)}")
print(f"  Sample IDs: {sorted(submission['SK_ID_CURR'].head(10).tolist())}\n")

# Check if any end_user IDs are in submission
end_user_ids = set(end_user_preds['sk_id_curr'].tolist())
submission_ids = set(submission['SK_ID_CURR'].tolist())

common_ids = end_user_ids & submission_ids

print("="*80)
print("CHECKING IF END_USER_TESTS IDs ARE IN SUBMISSION")
print("="*80)

if common_ids:
    print(f"\n[FOUND] {len(common_ids)} common IDs found in submission.csv")
    print(f"Common IDs: {sorted(common_ids)}\n")

    print("="*80)
    print("COMPARISON: End User Tests vs Submission.csv")
    print("="*80)
    print(f"{'SK_ID':>10s} {'End User':>12s} {'Submission':>12s} {'Difference':>12s}")
    print("-"*80)

    for sk_id in sorted(common_ids):
        end_user_prob = end_user_preds[end_user_preds['sk_id_curr'] == sk_id]['probability'].values[0]
        submission_prob = submission[submission['SK_ID_CURR'] == sk_id]['TARGET'].values[0]
        diff = abs(end_user_prob - submission_prob)

        match_str = "MATCH!" if diff < 0.0001 else f"{diff:.4f}"
        print(f"{sk_id:10d} {end_user_prob:12.4f} {submission_prob:12.4f} {match_str:>12s}")

    print("="*80)
else:
    print("\n[INFO] No common IDs found.")
    print("This means the end_user_tests applications are from the TRAINING set,")
    print("not the TEST set.\n")

    # Check if they're in the training set instead
    print("Checking if IDs are in training set...")
    app_train = pd.read_csv(PROJECT_ROOT / 'data' / 'application_train.csv')
    train_ids = set(app_train['SK_ID_CURR'].tolist())

    common_with_train = end_user_ids & train_ids

    if common_with_train:
        print(f"\n[FOUND] All {len(common_with_train)} end_user_tests IDs are in TRAINING set")
        print(f"IDs: {sorted(common_with_train)}\n")

        # Load training predictions for comparison
        train_preds = pd.read_csv(PROJECT_ROOT / 'results' / 'train_predictions.csv')
        train_preds['SK_ID_CURR'] = app_train['SK_ID_CURR'].values

        print("="*80)
        print("COMPARISON: End User Tests vs Training Predictions (CV)")
        print("="*80)
        print(f"{'SK_ID':>10s} {'API Pred':>12s} {'Train CV':>12s} {'Difference':>12s} {'Note':>20s}")
        print("-"*80)

        for sk_id in sorted(common_with_train):
            api_prob = end_user_preds[end_user_preds['sk_id_curr'] == sk_id]['probability'].values[0]
            train_prob = train_preds[train_preds['SK_ID_CURR'] == sk_id]['PROBABILITY'].values[0]
            diff = abs(api_prob - train_prob)

            note = "Expected diff" if diff > 0.01 else "Close match"
            print(f"{sk_id:10d} {api_prob:12.4f} {train_prob:12.4f} {diff:12.4f} {note:>20s}")

        print("="*80)
        print("\nNOTE: Differences are EXPECTED because:")
        print("  - API uses production model (trained on ALL data)")
        print("  - train_predictions.csv uses CV predictions (out-of-fold)")
        print("  See PREDICTION_DIFFERENCES_EXPLAINED.md for details.")
        print("="*80)
    else:
        print("[WARNING] IDs not found in training set either!")
        print(f"End user IDs: {sorted(end_user_ids)}")
        print(f"Sample train IDs: {sorted(list(train_ids)[:10])}")
