"""Compare end_user_test predictions with submission.csv"""
import pandas as pd

# Load both files
submission = pd.read_csv('results/submission.csv')
end_user_test = pd.read_csv('results/end_user_test_predictions.csv')

print("\n" + "="*70)
print("PREDICTION COMPARISON: end_user_test vs submission.csv")
print("="*70)

print(f"\nFile Info:")
print(f"  submission.csv:                {len(submission):,} rows (complete test set)")
print(f"  end_user_test_predictions.csv: {len(end_user_test):,} rows (sample)")

print(f"\nColumns:")
print(f"  submission.csv:                {', '.join(submission.columns)}")
print(f"  end_user_test_predictions.csv: {', '.join(end_user_test.columns)}")

print("\n" + "-"*70)
print("Checking if end_user_test IDs match submission.csv predictions...")
print("-"*70 + "\n")

matches = 0
mismatches = []
not_found = []

for _, row in end_user_test.iterrows():
    id_val = row['sk_id_curr']
    
    # Find matching row in submission
    sub_rows = submission[submission['SK_ID_CURR'] == id_val]
    
    if len(sub_rows) > 0:
        sub_row = sub_rows.iloc[0]
        
        # Compare predictions
        end_user_pred = int(row['prediction'])
        submission_pred = int(sub_row['PREDICTION'])
        
        # Compare probabilities
        end_user_prob = float(row['probability'])
        submission_prob = float(sub_row['TARGET'])
        
        prob_diff = abs(end_user_prob - submission_prob)
        
        if end_user_pred == submission_pred:
            status = "✓ MATCH"
            color_indicator = ""
            matches += 1
        else:
            status = "✗ MISMATCH"
            color_indicator = " <-- PROBLEM!"
            mismatches.append(id_val)
        
        print(f"  ID {id_val:6d}: {status}")
        print(f"    Prediction:  EndUser={end_user_pred}, Submission={submission_pred}{color_indicator}")
        print(f"    Probability: EndUser={end_user_prob:.4f}, Submission={submission_prob:.4f} (diff={prob_diff:.6f})")
        print()
    else:
        print(f"  ID {id_val}: ✗ NOT FOUND in submission.csv\n")
        not_found.append(id_val)

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\n  Total IDs checked:     {len(end_user_test)}")
print(f"  Matches:               {matches}")
print(f"  Mismatches:            {len(mismatches)}")
print(f"  Not found:             {len(not_found)}")

if matches == len(end_user_test):
    print(f"\n  ✓✓✓ ALL PREDICTIONS MATCH! ✓✓✓")
else:
    print(f"\n  ⚠ ISSUES FOUND:")
    if mismatches:
        print(f"    Mismatched IDs: {mismatches}")
    if not_found:
        print(f"    Not found IDs: {not_found}")

print("\n" + "="*70 + "\n")
