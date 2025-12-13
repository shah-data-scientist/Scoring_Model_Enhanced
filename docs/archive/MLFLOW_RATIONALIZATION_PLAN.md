# MLflow Rationalization Strategy

## Current State Analysis

### Experiments Summary
- **Total Experiments:** 7
- **Total Runs:** 66
- **Experiments with Artifacts:** 26 (many duplicates)

### Key Experiments (Production-Related)
1. **Experiment 4: credit_scoring_final_delivery** (ID: 4)
   - Run: `final_model_application`
   - Has model hyperparameters and optimal_threshold metric (0.338)
   - **Missing artifacts** (model file not found)

2. **Experiment 6: credit_scoring_production** (ID: 6)
   - Run: `production_lightgbm_189features`
   - Has model metadata (189 features, stage: production)
   - **Missing artifacts** (only empty models/ folder)

### Other Experiments
- **Exp 1:** credit_scoring_model_selection (model comparison)
- **Exp 2:** credit_scoring_feature_engineering_cv (16 runs with duplicate artifacts)
- **Exp 3:** credit_scoring_optimization_fbeta (Optuna hyperparameter tuning)
- **Exp 5:** test_experiment (4 test runs from old project)

### Issues Identified
1. **Duplicate Artifacts:** Same artifacts stored in multiple experiment directories (Exp 2, 26, 57, 7b, etc.)
2. **Missing Model Files:** Both final_delivery and production runs don't have model pickle files
3. **Scattered Metadata:** Optimal threshold in Exp 4, model metadata in Exp 6
4. **Test Artifacts:** Experiment 5 contains old test runs

---

## Rationalization Plan

### Strategy: Consolidate to Single Production Experiment

1. **Keep:** Experiment 4 (`credit_scoring_final_delivery`) as PRIMARY production experiment
   - Rationale: Contains model hyperparameters and threshold optimization
   - Run to update: `final_model_application`
   - Add: Model artifact (production_model.pkl), Metrics, Complete documentation

2. **Merge:** Experiment 6 into Experiment 4
   - Copy metadata from Exp 6 to Exp 4
   - Delete Experiment 6 after merge

3. **Archive:** Experiments 1, 2, 3 (keep for reference, don't use in API)
   - These are development/research experiments
   - Tag them appropriately

4. **Delete:** Experiment 5 (test_experiment)
   - Contains old test runs with no production value

5. **Clean:** Remove duplicate artifacts in mlruns/

---

## Implementation Steps

### Phase 1: Create Clean Production Run
Create a new run in Experiment 4 with:
- Model file: models/production_model.pkl
- Predictions: results/static_model_predictions.parquet (or reference)
- Metrics: All performance metrics
- Parameters: Model hyperparameters + optimal_threshold (0.48)
- Tags: Clear documentation

### Phase 2: Update MLflow Database
- Delete Experiment 6
- Mark Experiments 1-3 as archived
- Keep Experiment 4 as production

### Phase 3: Clean Artifacts
- Remove duplicate artifacts from mlruns/
- Keep only one copy of each unique artifact

### Phase 4: Update API
- Update api/app.py to load model from MLflow
- Use MLflow tracking URI for model retrieval
- Implement fallback to local pickle if needed

### Phase 5: Verification
- Test MLflow UI can display production experiment
- Test API can load model from MLflow
- Verify all artifacts are visible in UI

---

## Final Structure

After rationalization:

```
MLflow Experiments:
├── 1: credit_scoring_model_selection [ARCHIVED - Development]
├── 2: credit_scoring_feature_engineering_cv [ARCHIVED - Development]
├── 3: credit_scoring_optimization_fbeta [ARCHIVED - Development]
└── 4: credit_scoring_final_delivery [PRODUCTION]
    └── Run: production_lightgbm_189features_v2 [ACTIVE]
        ├── Parameters:
        │   ├── class_weight: balanced
        │   ├── learning_rate: 0.0188...
        │   ├── max_depth: 10
        │   ├── n_estimators: 968
        │   ├── optimal_threshold: 0.48
        │   └── ...
        ├── Metrics:
        │   ├── accuracy: 0.712...
        │   ├── precision: 0.192...
        │   ├── recall: 0.671...
        │   ├── f1: 0.299...
        │   ├── roc_auc: 0.812...
        │   ├── business_cost: 151536
        │   └── ...
        ├── Tags:
        │   ├── stage: production
        │   ├── status: deployed
        │   ├── description: Production LightGBM model...
        │   └── ...
        └── Artifacts:
            ├── model.pkl (production model)
            ├── feature_importance.csv
            ├── confusion_matrix.png
            ├── model_metadata.json
            └── ...
```

---

## Benefits

1. **Single Source of Truth:** One production experiment, easy to find
2. **Complete Artifacts:** Model, metrics, features all in one place
3. **Visible in MLflow UI:** All data properly stored and displayed
4. **API Integration:** API can load directly from MLflow
5. **Clean History:** Development experiments archived, easy to reference
6. **No Duplicates:** Remove redundant artifact copies
