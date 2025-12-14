"""
JSON serialization utilities for Streamlit - handles NaN/Inf values.
Standalone version that doesn't depend on api module.
"""
import numpy as np
import pandas as pd
from typing import Any


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
