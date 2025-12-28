
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
from api.preprocessing_pipeline import PreprocessingPipeline

class TestPipelineCoverage:

    @pytest.fixture
    def mock_scaler(self):
        scaler = MagicMock()
        scaler.transform.return_value = np.array([[0.1, 0.2]])
        return scaler

    @pytest.fixture
    def mock_dataframes(self):
        """Create minimal valid dataframes for testing."""
        app_df = pd.DataFrame({
            'SK_ID_CURR': [100001, 100002],
            'AMT_INCOME_TOTAL': [100000.0, 200000.0],
            'CODE_GENDER': ['M', 'F'],
            'Feature_A': [1, 2]
        })
        return {'application.csv': app_df}

    @patch("api.preprocessing_pipeline.joblib.load")
    @patch("api.preprocessing_pipeline.pd.read_csv")
    def test_init_loads_artifacts(self, mock_read_csv, mock_load, mock_scaler):
        """Test initialization loads scaler and feature names."""
        # Mock feature names
        mock_read_csv.return_value = pd.DataFrame({'feature': ['Feature_A', 'Feature_B']})
        mock_load.return_value = mock_scaler

        # Mock existance of files
        with patch("pathlib.Path.exists", return_value=True):
            pipeline = PreprocessingPipeline(use_precomputed=False)
            
        assert pipeline.scaler is not None
        assert pipeline.expected_features == ['Feature_A', 'Feature_B']

    @patch("pathlib.Path.exists", return_value=False)
    def test_init_handles_missing_artifacts(self, mock_exists):
        """Test initialization handles missing files gracefully."""
        pipeline = PreprocessingPipeline(use_precomputed=False)
        assert pipeline.scaler is None
        assert pipeline.expected_features is None

    @patch("api.preprocessing_pipeline.pd.read_parquet")
    def test_init_loads_precomputed_parquet(self, mock_read_parquet):
        """Test loading precomputed features from parquet."""
        mock_df = pd.DataFrame({
            'SK_ID_CURR': [100, 101],
            'Feat1': [0.1, 0.2]
        })
        mock_read_parquet.return_value = mock_df

        with patch("pathlib.Path.exists", side_effect=lambda: True): # All exist
             pipeline = PreprocessingPipeline(use_precomputed=True)
        
        assert pipeline.precomputed_features is not None
        assert 100 in pipeline.precomputed_features.index

    def test_align_features_missing_and_extra(self):
        """Test align_features adds missing cols and removes extra cols."""
        pipeline = PreprocessingPipeline(use_precomputed=False)
        pipeline.expected_features = ['Feat_A', 'Feat_B']
        
        # Input has Extra_C, missing Feat_B
        df = pd.DataFrame({
            'SK_ID_CURR': [1],
            'Feat_A': [10],
            'Extra_C': [99]
        })
        
        result = pipeline.align_features(df)
        
        assert 'SK_ID_CURR' in result.columns
        assert 'Feat_B' in result.columns # Added
        assert 'Extra_C' not in result.columns # Removed
        assert result['Feat_B'].iloc[0] == 0 # Filled with 0
        assert list(result.columns) == ['SK_ID_CURR', 'Feat_A', 'Feat_B']

    def test_encode_and_clean(self):
        """Test categorical encoding and column cleaning."""
        pipeline = PreprocessingPipeline(use_precomputed=False)
        
        df = pd.DataFrame({
            'SK_ID_CURR': [1, 2],
            'Cat_Col': ['A', 'B'], # Low cardinality
            'Dirty Name': [1, 1]
        })
        
        result = pipeline.encode_and_clean(df)
        
        # Check encoding (A should be dropped if drop_first=True, B becomes column)
        # Note: pandas get_dummies behavior depends on categories found.
        # If A and B are present, and drop_first=True:
        # Cat_Col_B should exist (1 if B, 0 if A)
        assert 'Cat_Col_B' in result.columns
        assert 'Cat_Col' not in result.columns # Original dropped
        
        # Check cleaning
        assert 'Dirty_Name' in result.columns

    def test_impute_missing_values(self):
        """Test imputation using global medians or batch medians."""
        pipeline = PreprocessingPipeline(use_precomputed=False)
        pipeline.medians = {'Val': 50.0} # Global median
        
        df = pd.DataFrame({
            'SK_ID_CURR': [1, 2],
            'Val': [np.nan, 100.0],
            'Other': [np.nan, 20.0] # No global median
        })
        
        result = pipeline.impute_missing_values(df)
        
        # Global median used
        assert result['Val'].iloc[0] == 50.0 
        
        # Batch median used for 'Other' (20.0 median of [NaN, 20])
        assert result['Other'].iloc[0] == 20.0

    @patch("api.preprocessing_pipeline.aggregate_bureau")
    @patch("api.preprocessing_pipeline.aggregate_previous_applications")
    def test_process_full_pipeline_flow(self, mock_prev, mock_bureau, mock_dataframes):
        """Test the full process flow with mocks."""
        pipeline = PreprocessingPipeline(use_precomputed=False)
        # Mock expected features to trigger scaling logic
        pipeline.expected_features = ['AMT_INCOME_TOTAL', 'CODE_GENDER_M'] 
        
        # Mock aggregations to return empty DF to avoid merge errors
        mock_bureau.return_value = pd.DataFrame({'SK_ID_CURR': [100001, 100002]})
        mock_prev.return_value = pd.DataFrame({'SK_ID_CURR': [100001, 100002]})
        
        # Mock creation of domain features to simple return
        with patch.object(pipeline, 'create_engineered_features', side_effect=lambda x: x):
            # Mock encode to produce expected feature
            with patch.object(pipeline, 'encode_and_clean') as mock_encode:
                mock_encode.return_value = pd.DataFrame({
                    'SK_ID_CURR': [100001, 100002],
                    'AMT_INCOME_TOTAL': [100000.0, 200000.0],
                    'CODE_GENDER_M': [1, 0]
                })
                
                # Run process
                features, sk_ids = pipeline.process(mock_dataframes, keep_sk_id=True)
                
                assert len(features) == 2
                assert 'AMT_INCOME_TOTAL' in features.columns

    def test_process_missing_application_csv(self):
        """Test process raises error if application.csv missing."""
        pipeline = PreprocessingPipeline(use_precomputed=False)
        with pytest.raises(ValueError):
            pipeline.process({})

    def test_process_uses_precomputed_mix(self):
        """Test mixing precomputed and new applications."""
        pipeline = PreprocessingPipeline(use_precomputed=True)
        
        # Setup precomputed for ID 100
        pipeline.precomputed_features = pd.DataFrame({
            'Feat1': [0.5]
        }, index=[100])
        
        # Input has ID 100 (Known) and 200 (Unknown)
        app_df = pd.DataFrame({
            'SK_ID_CURR': [100, 200],
            'Feat1': [999, 999] # Raw values
        })
        
        # Mock full pipeline for the unknown one
        with patch.object(pipeline, '_process_full_pipeline') as mock_full:
            mock_full.return_value = pd.DataFrame({
                'SK_ID_CURR': [200],
                'Feat1': [0.9] # Processed value
            })
            
            features, sk_ids = pipeline.process({'application.csv': app_df})
            
            # Result should have 2 rows
            assert len(features) == 2
            # ID 100 should come from precomputed (0.5)
            # ID 200 should come from full pipeline (0.9)
            val_100 = features.loc[features['SK_ID_CURR'] == 100, 'Feat1'].iloc[0]
            val_200 = features.loc[features['SK_ID_CURR'] == 200, 'Feat1'].iloc[0]
            
            assert val_100 == 0.5
            assert val_200 == 0.9

