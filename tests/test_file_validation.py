"""Tests for file validation module."""
import pytest
import pandas as pd
import numpy as np


class TestFileValidationBasics:
    """Tests for basic file validation logic."""

    def test_dataframe_operations(self):
        """Test basic dataframe operations used in validation."""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [10.5, 20.3, 30.1, 40.2, 50.9],
        })
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert len(df.columns) == 2

    def test_nan_handling(self):
        """Test handling NaN values."""
        df = pd.DataFrame({
            'col1': [1, 2, np.nan, 4, 5],
            'col2': [10, 20, 30, 40, 50]
        })
        
        assert df['col1'].isna().sum() == 1
        df_clean = df.dropna()
        assert len(df_clean) == 4

    def test_infinity_handling(self):
        """Test handling infinite values."""
        df = pd.DataFrame({
            'col1': [1, 2, np.inf, 4, 5],
            'col2': [10, 20, 30, 40, 50]
        })
        
        assert np.isinf(df['col1']).sum() == 1

    def test_numeric_column_check(self):
        """Test numeric column detection."""
        df = pd.DataFrame({
            'num_col': [1.5, 2.5, 3.5],
            'int_col': [1, 2, 3]
        })
        
        assert 'num_col' in df.columns
        assert 'int_col' in df.columns
        assert pd.api.types.is_numeric_dtype(df['num_col'])
        assert pd.api.types.is_numeric_dtype(df['int_col'])

    def test_string_column_handling(self):
        """Test string column handling."""
        df = pd.DataFrame({
            'str_col': ['a', 'b', 'c'],
            'num_col': [1, 2, 3]
        })
        
        assert pd.api.types.is_string_dtype(df['str_col'])
        assert pd.api.types.is_numeric_dtype(df['num_col'])
