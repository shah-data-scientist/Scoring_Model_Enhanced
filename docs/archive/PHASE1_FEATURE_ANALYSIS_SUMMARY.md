# Phase 1: Feature Importance & Raw Feature Analysis - Summary

## Overview

This document summarizes the feature importance analysis performed on the production LightGBM model to identify critical raw features required for the API.

## Model Information

- **Model Type**: LightGBM Classifier
- **Total Features**: 189
- **Number of Classes**: 2 (binary classification)
- **Model Location**: `models/production_model.pkl`
- **MLflow Registry**: `models:/credit_scoring_production_model/Production`
- **MLflow Run ID**: `83e2e1ec9b254fc59b4d3bfa7ae75b1f`

## Feature Importance Analysis Results

### Critical Features Threshold
- **Threshold**: 85% cumulative importance
- **Critical Model Features**: 80 features (out of 189)
- **Coverage**: 84.97% of total model importance

### Top 10 Most Important Model Features

| Rank | Feature | Importance | Cumulative |
|------|---------|------------|------------|
| 1 | DAYS_BIRTH | 101.0 | 3.4% |
| 2 | DAYS_EMPLOYED | 95.0 | 6.5% |
| 3 | EXT_SOURCE_MEAN | 93.0 | 9.6% |
| 4 | CREDIT_TO_GOODS_RATIO | 88.0 | 12.6% |
| 5 | AMT_ANNUITY | 84.0 | 15.4% |
| 6 | AMT_CREDIT | 79.0 | 18.0% |
| 7 | EXT_SOURCE_1 | 59.0 | 20.0% |
| 8 | INST_IS_LATE_MEAN | 58.0 | 21.9% |
| 9 | BUREAU_DAYS_CREDIT_ENDDATE_MAX | 53.0 | 23.7% |
| 10 | DAYS_ID_PUBLISH | 52.0 | 25.4% |

## Raw Feature Requirements

### Feature Distribution by Table

| Source Table | # Features | Total Importance | Description |
|--------------|------------|------------------|-------------|
| previous_application.csv | 44 | 478.0 | Aggregated features from previous loan applications |
| bureau.csv | 32 | 348.0 | Aggregated features from credit bureau history |
| installments_payments.csv | 29 | 333.0 | Aggregated features from payment installments |
| application.csv | 70 | ~1,500 | Direct and engineered features from main application |
| POS_CASH_balance.csv | 14 | 53.0 | Aggregated features from POS and cash loans |

### Top 15 Most Important Raw Features

| Rank | Source | Importance | Type |
|------|--------|------------|------|
| 1 | previous_application.csv | 478.0 | Aggregations |
| 2 | bureau.csv | 348.0 | Aggregations |
| 3 | installments_payments.csv | 333.0 | Aggregations |
| 4 | application.csv:EXT_SOURCE_1 | 226.0 | Direct feature |
| 5 | application.csv:EXT_SOURCE_3 | 218.0 | Direct feature |
| 6 | application.csv:EXT_SOURCE_2 | 213.0 | Direct feature |
| 7 | application.csv:AMT_CREDIT | 167.0 | Direct feature |
| 8 | application.csv:AMT_ANNUITY | 127.0 | Direct feature |
| 9 | application.csv:DAYS_BIRTH | 101.0 | Direct feature |
| 10 | application.csv:DAYS_EMPLOYED | 95.0 | Direct feature |
| 11 | application.csv:AMT_GOODS_PRICE | 88.0 | Direct feature |
| 12 | application.csv:AMT_INCOME_TOTAL | 58.0 | Direct feature |
| 13 | POS_CASH_balance.csv | 53.0 | Aggregations |
| 14 | application.csv:DAYS_ID_PUBLISH | 52.0 | Direct feature |
| 15 | application.csv:REGION_RATING_CLIENT | 52.0 | Direct feature |

## Files Required for API

Based on the feature importance analysis and existing `required_files.json`, the API must accept the following CSV files:

### Mandatory Files (All Required)

1. **application.csv** - Main application data (PRIMARY)
   - 46 critical specific columns identified
   - Contains most engineered features (CREDIT_TO_GOODS_RATIO, EXT_SOURCE_MEAN, etc.)

2. **bureau.csv** - Credit bureau history
   - Required for 32 aggregated BUREAU_* features
   - Importance: 348.0 (2nd highest)

3. **bureau_balance.csv** - Bureau monthly balance
   - Required for bureau aggregations
   - Used in conjunction with bureau.csv

4. **previous_application.csv** - Previous loan applications
   - Required for 44 aggregated PREV_* features
   - Importance: 478.0 (HIGHEST)

5. **installments_payments.csv** - Payment installment history
   - Required for 29 aggregated INST_* features
   - Importance: 333.0 (3rd highest)

6. **POS_CASH_balance.csv** - POS and cash loan balance
   - Required for 14 aggregated POS_* features
   - Importance: 53.0

7. **credit_card_balance.csv** - Credit card monthly balance
   - Marked as required in original configuration
   - May have zero direct feature importance but needed for completeness

## Critical Raw Features - application.csv (46 features)

These are specific columns from application.csv that have been identified as critical:

### Continuous Features (17)
- AMT_ANNUITY
- AMT_CREDIT
- AMT_GOODS_PRICE
- AMT_INCOME_TOTAL
- DAYS_BIRTH
- DAYS_EMPLOYED
- DAYS_ID_PUBLISH
- DAYS_LAST_PHONE_CHANGE
- DAYS_REGISTRATION
- EXT_SOURCE_1
- EXT_SOURCE_2
- EXT_SOURCE_3
- FLOORSMAX_AVG
- OWN_CAR_AGE
- REGION_RATING_CLIENT
- INST_DPD_MAX
- INST_DPD_MEAN

### Categorical Features (9)
- CODE_GENDER
- FLAG_OWN_CAR
- NAME_EDUCATION_TYPE_Higher_education
- NAME_FAMILY_STATUS_Married
- INST_IS_LATE_MEAN
- INST_PAYMENT_DIFF_MEAN
- INST_PAYMENT_DIFF_SUM
- INST_PAYMENT_RATIO_MAX
- INST_PAYMENT_RATIO_MEAN

### Document Flags (20)
- FLAG_DOCUMENT_2 through FLAG_DOCUMENT_21

## API Validation Strategy

### For application.csv:
- **Validate specific columns**: Check that 46 critical columns are present
- **Threshold**: 85% of critical columns must be present
- **Action if missing**: Return specific error message listing missing critical columns

### For other files (bureau, previous_application, etc.):
- **Validate file presence**: Ensure all 7 CSV files are uploaded
- **Column validation**: Less strict - accept if file structure is reasonable
- **Reason**: These files are used for aggregations where exact column requirements vary

## Configuration Files Generated

1. **config/all_raw_features.json**
   - Contains all 189 mapped raw features
   - Organized by source CSV file

2. **config/critical_raw_features.json**
   - Contains 46 critical features from application.csv
   - Used for API validation
   - 85% importance threshold

3. **config/raw_feature_importance.json**
   - Detailed mapping of each raw feature to model features
   - Includes importance scores and contributing model features

4. **config/required_files.json**
   - Lists all 7 required CSV files
   - Existing file preserved

## Key Insights

1. **EXT_SOURCE features dominate**: EXT_SOURCE_1, EXT_SOURCE_2, and EXT_SOURCE_3 together account for ~657 importance points across their direct and engineered forms (MEAN, MIN, MAX)

2. **Aggregations are critical**: Features from other tables (bureau, previous_application, installments) contribute significantly (1,159 combined importance)

3. **Temporal features matter**: DAYS_BIRTH, DAYS_EMPLOYED, DAYS_ID_PUBLISH are all in top 15

4. **Credit ratio features**: Engineered features like CREDIT_TO_GOODS_RATIO are highly important

5. **Payment behavior**: INST_IS_LATE_MEAN (installment lateness) is 8th most important feature

## Next Steps for Phase 2

1. **API Endpoint Design**:
   - Accept multipart/form-data with 7 CSV files
   - Validate file presence
   - Validate critical columns for application.csv

2. **Data Preprocessing Pipeline**:
   - Reuse existing preprocessing from training
   - Perform same aggregations for bureau, previous_application, etc.
   - Apply same feature engineering

3. **Error Handling**:
   - Missing file errors
   - Missing critical column errors
   - Data type validation errors

4. **Batch Processing**:
   - Process multiple applications in uploaded files
   - Return predictions with risk levels
   - Store raw data in PostgreSQL

## Files Modified/Created

- ✅ `scripts/analysis/register_production_model.py` - Model registration
- ✅ `scripts/analysis/extract_feature_importance_lightgbm.py` - Feature analysis
- ✅ `models/production_model.pkl` - Updated with correct LightGBM model
- ✅ `config/all_raw_features.json` - All features
- ✅ `config/critical_raw_features.json` - Critical features
- ✅ `config/raw_feature_importance.json` - Detailed importance mapping
- ✅ `scripts/analysis/output/model_feature_importance_189.csv` - Full importance table
- ✅ `docs/PHASE1_FEATURE_ANALYSIS_SUMMARY.md` - This document

---

**Date**: 2025-12-11
**Analysis Tool**: LightGBM feature_importances_
**Model Version**: Production (v2, Run ID: 83e2e1ec9b254fc59b4d3bfa7ae75b1f)
