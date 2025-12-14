"""
Tests for JSON serialization utilities.
These tests ensure NaN/Inf handling works correctly.
"""
import pytest
import numpy as np
import pandas as pd
from api.json_utils import sanitize_for_json, dataframe_to_json_safe, validate_numeric_value


class TestSanitizeForJson:
    """Test the sanitize_for_json function."""
    
    def test_dataframe_with_nan(self):
        """Test DataFrame with NaN values."""
        df = pd.DataFrame({
            'a': [1, 2, np.nan],
            'b': [4, np.nan, 6]
        })
        result = sanitize_for_json(df)
        
        assert result[2]['a'] is None
        assert result[1]['b'] is None
        assert result[0]['a'] == 1
    
    def test_dataframe_with_inf(self):
        """Test DataFrame with Inf values."""
        df = pd.DataFrame({
            'a': [1, np.inf, 3],
            'b': [4, 5, -np.inf]
        })
        result = sanitize_for_json(df)
        
        assert result[1]['a'] is None
        assert result[2]['b'] is None
        assert result[0]['a'] == 1
    
    def test_series_with_nan_inf(self):
        """Test Series with NaN and Inf."""
        series = pd.Series([1, np.nan, np.inf, -np.inf, 5])
        result = sanitize_for_json(series)
        
        assert result[0] == 1
        assert result[1] is None
        assert result[2] is None
        assert result[3] is None
        assert result[4] == 5
    
    def test_dict_with_nan(self):
        """Test dictionary with NaN values."""
        data = {
            'a': 1,
            'b': np.nan,
            'c': [1, np.nan, 3]
        }
        result = sanitize_for_json(data)
        
        assert result['a'] == 1
        assert result['b'] is None
        assert result['c'][1] is None
    
    def test_numpy_scalars(self):
        """Test numpy scalar types."""
        assert sanitize_for_json(np.int64(5)) == 5
        assert sanitize_for_json(np.float64(3.14)) == 3.14
        assert sanitize_for_json(np.float64(np.nan)) is None
        assert sanitize_for_json(np.float64(np.inf)) is None
    
    def test_python_float_nan_inf(self):
        """Test Python float NaN and Inf."""
        assert sanitize_for_json(float('nan')) is None
        assert sanitize_for_json(float('inf')) is None
        assert sanitize_for_json(float('-inf')) is None
        assert sanitize_for_json(3.14) == 3.14
    
    def test_nested_structures(self):
        """Test deeply nested structures."""
        data = {
            'level1': {
                'level2': [
                    {'a': 1, 'b': np.nan},
                    {'a': np.inf, 'b': 3}
                ]
            }
        }
        result = sanitize_for_json(data)
        
        assert result['level1']['level2'][0]['b'] is None
        assert result['level1']['level2'][1]['a'] is None


class TestDataframeToJsonSafe:
    """Test the dataframe_to_json_safe function."""
    
    def test_records_orient(self):
        """Test with records orientation."""
        df = pd.DataFrame({
            'a': [1, np.nan, 3],
            'b': [np.inf, 5, 6]
        })
        result = dataframe_to_json_safe(df, orient='records')
        
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[1]['a'] is None
        assert result[0]['b'] is None
    
    def test_dict_orient(self):
        """Test with dict orientation."""
        df = pd.DataFrame({
            'a': [1, np.nan],
            'b': [np.inf, 4]
        })
        result = dataframe_to_json_safe(df, orient='dict')
        
        assert isinstance(result, dict)
        assert result['a'][1] is None
        assert result['b'][0] is None


class TestValidateNumericValue:
    """Test the validate_numeric_value function."""
    
    def test_valid_values(self):
        """Test with valid numeric values."""
        assert validate_numeric_value(5) == 5.0
        assert validate_numeric_value(3.14) == 3.14
        assert validate_numeric_value(np.float64(2.5)) == 2.5
    
    def test_nan_raises_error(self):
        """Test that NaN raises ValueError."""
        with pytest.raises(ValueError, match="cannot be NaN"):
            validate_numeric_value(np.nan)
        
        with pytest.raises(ValueError, match="cannot be NaN"):
            validate_numeric_value(float('nan'))
    
    def test_inf_raises_error(self):
        """Test that Inf raises ValueError."""
        with pytest.raises(ValueError, match="cannot be Inf"):
            validate_numeric_value(np.inf)
        
        with pytest.raises(ValueError, match="cannot be Inf"):
            validate_numeric_value(float('inf'))
        
        with pytest.raises(ValueError, match="cannot be Inf"):
            validate_numeric_value(float('-inf'))
    
    def test_custom_field_name(self):
        """Test custom field name in error message."""
        with pytest.raises(ValueError, match="score cannot be NaN"):
            validate_numeric_value(np.nan, field_name="score")
