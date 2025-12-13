"""Identify which features are dataset-level (affected by cache)."""
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

print("="*80)
print("DATASET-LEVEL FEATURES IN YOUR MODEL")
print("="*80)

# Load feature names
feature_file = PROJECT_ROOT / 'data' / 'processed' / 'feature_names.csv'
if feature_file.exists():
    features = pd.read_csv(feature_file)
    print(f"\nTotal features: {len(features)}")
    
    # Search for group-based features
    dataset_level = features[features['feature'].str.contains('VS_|RATIO|NORMALIZED', case=False, na=False)]
    
    print(f"\n{len(dataset_level)} potential dataset-level features:")
    print("-"*80)
    for feat in dataset_level['feature'].head(20):
        print(f"  - {feat}")
    
    if len(dataset_level) > 20:
        print(f"  ... and {len(dataset_level) - 20} more")
else:
    print(f"Feature names file not found: {feature_file}")

# Check advanced_features.py for group transforms
print("\n" + "="*80)
print("CONFIRMED DATASET-LEVEL FEATURES (from code)")
print("="*80)

dataset_features = [
    "INCOME_VS_OCCUPATION_TYPE_MEAN",
    "INCOME_VS_ORGANIZATION_TYPE_MEAN", 
    "INCOME_VS_NAME_EDUCATION_TYPE_MEAN",
    "CREDIT_VS_CONTRACT_MEAN"
]

print("\nThese 4 features use groupby().transform('mean'):")
print("(Computed from group means ACROSS the dataset)")
for i, feat in enumerate(dataset_features, 1):
    print(f"  {i}. {feat}")

print("\n" + "="*80)
print("HOW THEY WORK")
print("="*80)

print("""
Example: INCOME_VS_OCCUPATION_TYPE_MEAN

Training (300K clients):
  1. Calculate mean income for each occupation across ALL 300K clients
     - Laborers: 150,000 (from 50,000 laborers)
     - Managers: 250,000 (from 30,000 managers)
     
  2. For each client, compute: client_income / occupation_mean
     - Client with Laborer job, 120K income: 120K / 150K = 0.80
     
API (50 uploaded clients):
  1. Calculate mean income for each occupation across ONLY 50 clients
     - Laborers: 180,000 (from 5 laborers in upload)
     - Managers: 300,000 (from 3 managers in upload)
     
  2. For same client: 120K / 180K = 0.67

**Result: Same client, different feature value (0.80 vs 0.67)**

This is why:
- Known SK_ID_CURR → Uses cached training feature (0.80)
- New SK_ID_CURR   → Uses live-computed feature (0.67)
""")

print("\n" + "="*80)
print("WHY THIS MATTERS")
print("="*80)

print("""
These 4 features tell the model:
"How does this client compare to others in their category?"

But the comparison group CHANGES:
- Training: Compare to 300,000 clients
- API: Compare to 50 uploaded clients

Small sample (50) has different statistics than large sample (300K)!

Example occupation distribution:
  Training: Laborers = 35%, Managers = 15%, Others = 50%
  Your 50:  Laborers = 20%, Managers = 10%, Others = 70%

Different distribution → Different group means → Different features → Different predictions!
""")

print("\n" + "="*80)
print("AGGREGATIONS THAT DON'T CAUSE PROBLEMS")
print("="*80)

print("""
Per-client aggregations (these are fine):
  - BUREAU_AMT_CREDIT_SUM_MEAN: Average credit per client's bureau records
  - PREV_AMT_APPLICATION_MEAN: Average application amount per client
  - CC_BALANCE_MEAN: Average credit card balance per client

These use: df.groupby('SK_ID_CURR').agg(...)
→ Each client's aggregations are independent
→ Changing SK_ID_CURR doesn't affect the numbers

Dataset-level features (these cause problems):
  - INCOME_VS_OCCUPATION_TYPE_MEAN: Ratio to occupation group mean
  - Uses: df.groupby('OCCUPATION_TYPE')['INCOME'].transform('mean')
  → Group means depend on WHO ELSE is in the dataset
  → Changing the dataset changes the feature values
""")

print("\n" + "="*80)
print("SOLUTION OPTIONS")
print("="*80)

print("""
1. **Remove dataset-level features** (retrain without them)
   ✓ Ensures consistency between training and API
   ✗ Lose potentially useful features
   
2. **Store global statistics** (save occupation means from training)
   ✓ API can use same group means as training
   ✗ Requires code changes and additional artifacts
   
3. **Disable cache** (use_precomputed=False)
   ✓ All predictions use same computation path
   ✗ Slower (must compute all features live)
   ⚠ BUT still inconsistent (API uses 50 clients, training used 300K)
   
4. **Hybrid: Cache global stats only**
   ✓ Per-client features computed live
   ✓ Dataset-level features use training statistics
   ✓ Best of both worlds
   ✗ Most complex to implement

Recommendation: Option 4 (cache global statistics from training)
""")
