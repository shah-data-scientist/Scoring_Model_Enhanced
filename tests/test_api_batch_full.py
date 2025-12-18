import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch, ANY
import pandas as pd
import numpy as np
import io
import json

from api.app import app
from api.batch_predictions import router

# Override dependencies
from backend.database import get_db

@pytest.fixture
def mock_db_session():
    return MagicMock()

def create_csv_file(filename, content=None):
    if content is None:
        df = pd.DataFrame({'SK_ID_CURR': [100001, 100002], 'col1': [1, 2]})
        content = df.to_csv(index=False).encode('utf-8')
    return (filename, content, 'text/csv')

def create_valid_files():
    # Minimal valid files
    return {
        'application': create_csv_file('application.csv'),
        'bureau': create_csv_file('bureau.csv'),
        'bureau_balance': create_csv_file('bureau_balance.csv'),
        'previous_application': create_csv_file('previous_application.csv'),
        'credit_card_balance': create_csv_file('credit_card_balance.csv'),
        'installments_payments': create_csv_file('installments_payments.csv'),
        'pos_cash_balance': create_csv_file('POS_CASH_balance.csv'),
    }

class TestBatchPredictEndpoint:

    @patch('api.batch_predictions.validate_all_files')
    @patch('api.batch_predictions.get_preprocessing_pipeline')
    @patch('api.batch_predictions.crud')
    @patch('api.batch_predictions.get_file_summaries')
    def test_predict_batch_success(self, mock_summaries, mock_crud, mock_pipeline_getter, mock_validate, test_app_client, mock_db_session):
        # Mock validation
        mock_validate.return_value = {'application.csv': pd.DataFrame({'SK_ID_CURR': [100001]})}
        mock_summaries.return_value = {'application.csv': {'rows': 1}}
        
        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline_getter.return_value = mock_pipeline
        # Now returns 3 values: scaled_df, unscaled_df, sk_id_curr
        mock_pipeline.process.return_value = (
            pd.DataFrame([[1, 2]], columns=['f1', 'f2']), 
            pd.DataFrame([[1, 2]], columns=['f1', 'f2']), # unscaled same as scaled for mock
            pd.Series([100001])
        )
        
        # Mock batch record
        mock_batch = MagicMock()
        mock_batch.id = 123
        mock_crud.create_prediction_batch.return_value = mock_batch
        
        # Mock model in api.app
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0])
        mock_model.predict_proba.return_value = np.array([[0.8, 0.2]]) # 0.2 probability of default
        
        with patch('api.app.model', mock_model):
            # Also patch ModelValidator to pass
            with patch('api.model_validator.ModelValidator.check_model_loaded', return_value=True):
                with patch('api.model_validator.ModelValidator.validate_model_attributes', return_value=True):
                    files = create_valid_files()
                    response = test_app_client.post("/batch/predict", files=files)

        assert response.status_code == 200, f"Response detail: {response.json() if response.status_code != 200 else ''}"
        data = response.json()
        assert data['success'] is True
        assert len(data['predictions']) == 1
        assert data['predictions'][0]['risk_level'] == 'LOW' # 0.2 < 0.3


    @patch('api.batch_predictions.validate_all_files')
    def test_predict_batch_validation_error(self, mock_validate, test_app_client):
        mock_validate.side_effect = Exception("Validation failed")
        
        # Mock model to avoid ModelValidator error
        mock_model = MagicMock()
        with patch('api.app.model', mock_model):
            with patch('api.model_validator.ModelValidator.check_model_loaded', return_value=True):
                with patch('api.model_validator.ModelValidator.validate_model_attributes', return_value=True):
                    files = create_valid_files()
                    response = test_app_client.post("/batch/predict", files=files)
        
        assert response.status_code == 400
        assert "Validation failed" in response.json()['detail']


    @patch('api.batch_predictions.validate_all_files')
    @patch('api.batch_predictions.get_preprocessing_pipeline')
    @patch('api.batch_predictions.crud')
    def test_predict_batch_preprocessing_error(self, mock_crud, mock_pipeline_getter, mock_validate, test_app_client, mock_db_session):
        # Mock validation success
        mock_validate.return_value = {'application.csv': pd.DataFrame({'SK_ID_CURR': [100001]})}
        
        # Mock pipeline failure
        mock_pipeline = MagicMock()
        mock_pipeline_getter.return_value = mock_pipeline
        mock_pipeline.process.side_effect = Exception("Preprocessing error")
        
        # Mock crud to return a batch object
        mock_batch = MagicMock()
        mock_batch.id = 1
        mock_crud.create_prediction_batch.return_value = mock_batch

        # Mock model
        mock_model = MagicMock()
        with patch('api.app.model', mock_model):
            with patch('api.model_validator.ModelValidator.check_model_loaded', return_value=True):
                with patch('api.model_validator.ModelValidator.validate_model_attributes', return_value=True):
                    files = create_valid_files()
                    response = test_app_client.post("/batch/predict", files=files)
        
        assert response.status_code == 500
        assert "Preprocessing error" in response.json()['detail']


    @patch('api.batch_predictions.validate_all_files')
    @patch('api.batch_predictions.get_preprocessing_pipeline')
    @patch('api.batch_predictions.crud')
    def test_predict_batch_prediction_error(self, mock_crud, mock_pipeline_getter, mock_validate, test_app_client, mock_db_session):
        mock_validate.return_value = {'application.csv': pd.DataFrame({'SK_ID_CURR': [100001]})}
        
        mock_pipeline = MagicMock()
        mock_pipeline_getter.return_value = mock_pipeline
        mock_pipeline.process.return_value = (
            pd.DataFrame([[1, 2]], columns=['f1', 'f2']), 
            pd.DataFrame([[1, 2]], columns=['f1', 'f2']),
            pd.Series([100001])
        )
        
        mock_batch = MagicMock()
        mock_batch.id = 1
        mock_crud.create_prediction_batch.return_value = mock_batch
        
        mock_model = MagicMock()
        mock_model.predict.side_effect = Exception("Model error")
        
        with patch('api.app.model', mock_model):
            with patch('api.model_validator.ModelValidator.check_model_loaded', return_value=True):
                with patch('api.model_validator.ModelValidator.validate_model_attributes', return_value=True):
                    files = create_valid_files()
                    response = test_app_client.post("/batch/predict", files=files)
        
        assert response.status_code == 500
        assert "Prediction failed" in response.json()['detail']

    
    def test_history_endpoint(self, test_app_client, mock_db_session):
        with patch('api.batch_predictions.crud') as mock_crud_module:
            mock_batch = MagicMock()
            mock_batch.id = 1
            mock_batch.status.value = 'completed'
            mock_batch.created_at = pd.Timestamp.now()
            mock_batch.completed_at = pd.Timestamp.now()
            mock_batch.total_applications = 1
            mock_batch.processed_applications = 1
            mock_batch.avg_probability = 0.5
            mock_batch.risk_low_count = 0
            mock_batch.risk_medium_count = 1
            mock_batch.risk_high_count = 0
            mock_batch.risk_critical_count = 0
            mock_batch.processing_time_seconds = 10.0
            
            mock_crud_module.get_recent_batches.return_value = [mock_batch]
            
            response = test_app_client.get("/batch/history")
            assert response.status_code == 200
            assert response.json()['success'] is True
            assert len(response.json()['batches']) == 1

    
    def test_download_endpoint_json(self, test_app_client):
        with patch('api.batch_predictions.crud') as mock_crud_module:
            mock_batch = MagicMock()
            mock_batch.batch_name = "test_batch"
            mock_crud_module.get_batch.return_value = mock_batch
            
            mock_pred = MagicMock()
            mock_pred.sk_id_curr = 123
            mock_pred.prediction = 0
            mock_pred.probability = 0.1
            mock_pred.risk_level.value = "LOW"
            mock_pred.shap_values = json.dumps({"f1": 0.1})
            mock_pred.top_features = json.dumps([{'feature': 'f1', 'shap_value': 0.1}])
            
            mock_crud_module.get_batch_predictions.return_value = [mock_pred]
            
            response = test_app_client.get("/batch/history/1/download?format=json")
            assert response.status_code == 200
            assert response.json()['success'] is True
            assert response.json()['predictions'][0]['SK_ID_CURR'] == 123

    def test_download_endpoint_csv(self, test_app_client):
        with patch('api.batch_predictions.crud') as mock_crud_module:
            mock_batch = MagicMock()
            mock_crud_module.get_batch.return_value = mock_batch
            
            mock_pred = MagicMock()
            mock_pred.sk_id_curr = 123
            mock_pred.prediction = 0
            mock_pred.probability = 0.1
            mock_pred.risk_level.value = "LOW"
            
            mock_crud_module.get_batch_predictions.return_value = [mock_pred]
            
            response = test_app_client.get("/batch/history/1/download?format=csv")
            assert response.status_code == 200
            assert response.headers['content-type'].startswith('text/csv')

    def test_download_endpoint_batch_not_found(self, test_app_client):
        with patch('api.batch_predictions.crud') as mock_crud_module:
            mock_crud_module.get_batch.return_value = None
            
            response = test_app_client.get("/batch/history/999/download")
            assert response.status_code == 404
            assert "Batch 999 not found" in response.json()['detail']

    def test_download_endpoint_no_predictions_found(self, test_app_client):
        with patch('api.batch_predictions.crud') as mock_crud_module:
            mock_batch = MagicMock()
            mock_crud_module.get_batch.return_value = mock_batch
            mock_crud_module.get_batch_predictions.return_value = []
            
            response = test_app_client.get("/batch/history/1/download")
            assert response.status_code == 404
            assert "No predictions found for batch 1" in response.json()['detail']

    def test_get_batch_statistics(self, test_app_client):
        with patch('api.batch_predictions.crud') as mock_crud_module:
            mock_crud_module.get_batch_statistics.return_value = {
                'total_batches': 10,
                'completed_batches': 8,
                'total_predictions': 100,
                'risk_distribution': {'LOW': 50, 'MEDIUM': 30, 'HIGH': 20, 'CRITICAL': 0}
            }
            mock_crud_module.get_average_processing_time.return_value = 15.5
            mock_crud_module.get_daily_prediction_counts.return_value = [{"date": "2023-01-01", "count": 5}]
            
            response = test_app_client.get("/batch/statistics")
            assert response.status_code == 200
            assert response.json()['success'] is True
            assert response.json()['statistics']['total_batches'] == 10
            assert len(response.json()['daily_predictions']) == 1