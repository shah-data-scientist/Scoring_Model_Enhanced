"""
JSON serialization utilities for handling pandas DataFrames and numpy values.
Ensures NaN, Inf, and other non-JSON-compliant values are properly handled.
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Union


def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively sanitize an object for JSON serialization.
    Replaces NaN, Inf, and -Inf with None.
    
    Args:
        obj: Any Python object (DataFrame, dict, list, scalar, etc.)
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, pd.DataFrame):
        # Replace inf values with NaN first
        df_safe = obj.replace([np.inf, -np.inf], np.nan)
        # Convert to dict, then recursively sanitize to replace NaN with None
        result = df_safe.to_dict(orient='records')
        return [{k: (None if pd.isna(v) else v) for k, v in row.items()} for row in result]
    
    elif isinstance(obj, pd.Series):
        series_safe = obj.replace([np.inf, -np.inf], np.nan)
        # Convert to list and replace NaN with None
        return [None if pd.isna(x) else x for x in series_safe.tolist()]
    
    elif isinstance(obj, dict):
        return {key: sanitize_for_json(value) for key, value in obj.items()}
    
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    
    elif isinstance(obj, (np.integer, np.floating)):
        # Handle numpy scalars
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj.item()
    
    elif isinstance(obj, float):
        # Handle Python floats
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    
    elif pd.isna(obj):
        # Handle pandas NA/NaT
        return None
    
    return obj


def dataframe_to_json_safe(df: pd.DataFrame, orient: str = 'records') -> Union[List[Dict], Dict]:
    """
    Convert a pandas DataFrame to JSON-safe structure.
    
    Args:
        df: DataFrame to convert
        orient: Orientation for to_dict (default: 'records')
        
    Returns:
        JSON-serializable dictionary or list
    """
    df_safe = df.replace([np.inf, -np.inf], np.nan)
    result = df_safe.to_dict(orient=orient)
    
    # Post-process to replace NaN with None
    if orient == 'records':
        return [{k: (None if pd.isna(v) else v) for k, v in row.items()} for row in result]
    elif orient == 'dict':
        return {k: {idx: (None if pd.isna(v) else v) for idx, v in col.items()} for k, col in result.items()}
    else:
        # For other orientations, use recursive sanitization
        return sanitize_for_json(result)


def validate_numeric_value(value: Any, field_name: str = "value") -> float:
    """
    Validate and sanitize a numeric value.
    Raises ValueError if value is NaN or Inf.
    
    Args:
        value: Value to validate
        field_name: Name of the field for error messages
        
    Returns:
        Validated float value
        
    Raises:
        ValueError: If value is NaN or Inf
    """
    if pd.isna(value):
        raise ValueError(f"{field_name} cannot be NaN")
    
    if isinstance(value, (float, np.floating)) and np.isinf(value):
        raise ValueError(f"{field_name} cannot be Inf")
    
    return float(value)
