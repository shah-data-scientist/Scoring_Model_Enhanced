"""File Validation Module for Batch Predictions

Validates uploaded CSV files against requirements:
- All 7 required files present
- Critical columns present in application.csv
- Basic data quality checks
"""



import json
from pathlib import Path

import pandas as pd
from fastapi import HTTPException, UploadFile, status

# Load configuration

PROJECT_ROOT = Path(__file__).parent.parent

CONFIG_DIR = PROJECT_ROOT / "config"



# Load required files configuration

with open(CONFIG_DIR / "required_files.json") as f:

    REQUIRED_FILES_CONFIG = json.load(f)



# Load critical features configuration

with open(CONFIG_DIR / "critical_raw_features.json") as f:

    CRITICAL_FEATURES_CONFIG = json.load(f)



# Required file names

REQUIRED_FILES = set(REQUIRED_FILES_CONFIG.keys())



# Critical columns for application.csv

CRITICAL_APPLICATION_COLUMNS = set(CRITICAL_FEATURES_CONFIG["application.csv"])

CRITICAL_THRESHOLD = CRITICAL_FEATURES_CONFIG["_metadata"]["threshold"]





class FileValidationError(Exception):
    """Custom exception for file validation errors."""






def validate_file_presence(uploaded_files: dict[str, UploadFile]) -> tuple[bool, list[str]]:
    """Validate that all required CSV files are present.

    Args:
        uploaded_files: Dictionary of {filename: UploadFile}



    Returns:
        Tuple of (is_valid, missing_files)

    """
    uploaded_filenames = set(uploaded_files.keys())

    missing_files = REQUIRED_FILES - uploaded_filenames



    return (len(missing_files) == 0, sorted(list(missing_files)))





def validate_application_columns(df: pd.DataFrame) -> tuple[bool, list[str], float]:
    """Validate that critical columns are present in application.csv.

    Args:
        df: Application DataFrame



    Returns:
        Tuple of (is_valid, missing_columns, coverage_percentage)

    """
    df_columns = set(df.columns)

    missing_columns = CRITICAL_APPLICATION_COLUMNS - df_columns



    # Calculate coverage

    present_columns = CRITICAL_APPLICATION_COLUMNS - missing_columns

    coverage = len(present_columns) / len(CRITICAL_APPLICATION_COLUMNS)



    # Check if meets threshold

    is_valid = coverage >= CRITICAL_THRESHOLD



    return (is_valid, sorted(list(missing_columns)), coverage)





def validate_csv_structure(file: UploadFile, filename: str) -> pd.DataFrame:
    """Validate CSV file structure and load into DataFrame.

    Args:
        file: Uploaded CSV file

        filename: Name of the file



    Returns:
        Loaded DataFrame



    Raises:
        FileValidationError: If file cannot be read or is invalid

    """
    try:

        # Read CSV

        df = pd.read_csv(file.file)



        # Reset file pointer for potential re-reading

        file.file.seek(0)



        # Basic validations

        if df.empty:

            raise FileValidationError(f"{filename} is empty (0 rows)")



        if len(df.columns) == 0:

            raise FileValidationError(f"{filename} has no columns")



        # Check for SK_ID_CURR in main tables

        if filename in ['application.csv']:

            if 'SK_ID_CURR' not in df.columns:

                raise FileValidationError(

                    f"{filename} must contain 'SK_ID_CURR' column"

                )



        return df



    except pd.errors.EmptyDataError:

        raise FileValidationError(f"{filename} is empty or invalid CSV format")

    except pd.errors.ParserError as e:

        raise FileValidationError(f"{filename} parsing error: {str(e)}")

    except Exception as e:

        raise FileValidationError(f"{filename} error: {str(e)}")





def validate_all_files(uploaded_files: dict[str, UploadFile]) -> dict[str, pd.DataFrame]:
    """Comprehensive validation of all uploaded CSV files.

    Args:
        uploaded_files: Dictionary of {filename: UploadFile}



    Returns:
        Dictionary of {filename: DataFrame}



    Raises:
        HTTPException: If validation fails

    """
    # Step 1: Check file presence

    files_present, missing_files = validate_file_presence(uploaded_files)



    if not files_present:

        raise HTTPException(

            status_code=status.HTTP_400_BAD_REQUEST,

            detail={

                "error": "Missing required files",

                "missing_files": missing_files,

                "required_files": sorted(list(REQUIRED_FILES))

            }

        )



    # Step 2: Load and validate structure

    dataframes = {}

    errors = []



    for filename, file in uploaded_files.items():

        try:

            df = validate_csv_structure(file, filename)

            dataframes[filename] = df

        except FileValidationError as e:

            errors.append(str(e))



    if errors:

        raise HTTPException(

            status_code=status.HTTP_400_BAD_REQUEST,

            detail={

                "error": "File structure validation failed",

                "errors": errors

            }

        )



    # Step 3: Validate critical columns in application.csv

    if 'application.csv' in dataframes:

        is_valid, missing_cols, coverage = validate_application_columns(

            dataframes['application.csv']

        )



        if not is_valid:

            raise HTTPException(

                status_code=status.HTTP_400_BAD_REQUEST,

                detail={

                    "error": "Critical columns missing in application.csv",

                    "missing_columns": missing_cols,

                    "coverage": f"{coverage*100:.1f}%",

                    "required_coverage": f"{CRITICAL_THRESHOLD*100:.1f}%",

                    "message": f"Only {len(CRITICAL_APPLICATION_COLUMNS) - len(missing_cols)}/{len(CRITICAL_APPLICATION_COLUMNS)} critical columns present"

                }

            )



    return dataframes





def get_file_summaries(dataframes: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """Get summary information for uploaded files.

    Args:
        dataframes: Dictionary of {filename: DataFrame}



    Returns:
        Dictionary of file summaries

    """
    summaries = {}



    for filename, df in dataframes.items():

        summaries[filename] = {

            "rows": len(df),

            "columns": len(df.columns),

            "memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,

            "has_sk_id_curr": 'SK_ID_CURR' in df.columns,

            "unique_ids": int(df['SK_ID_CURR'].nunique()) if 'SK_ID_CURR' in df.columns else None

        }



    return summaries





def validate_sk_id_consistency(dataframes: dict[str, pd.DataFrame]) -> tuple[bool, str]:
    """Validate that SK_ID_CURR values in auxiliary tables exist in application.csv.

    Args:
        dataframes: Dictionary of {filename: DataFrame}



    Returns:
        Tuple of (is_valid, message)

    """
    if 'application.csv' not in dataframes:

        return False, "application.csv not found"



    app_ids = set(dataframes['application.csv']['SK_ID_CURR'].unique())



    warnings = []



    # Check each auxiliary table

    auxiliary_tables = {

        'bureau.csv': 'SK_ID_CURR',

        'previous_application.csv': 'SK_ID_CURR',

        'POS_CASH_balance.csv': 'SK_ID_CURR',

        'credit_card_balance.csv': 'SK_ID_CURR',

        'installments_payments.csv': 'SK_ID_CURR'

    }



    for filename, id_col in auxiliary_tables.items():

        if filename in dataframes and id_col in dataframes[filename].columns:

            aux_ids = set(dataframes[filename][id_col].unique())

            orphan_ids = aux_ids - app_ids



            if orphan_ids:

                warnings.append(

                    f"{filename}: {len(orphan_ids)} IDs not in application.csv"

                )



    if warnings:

        return True, "Warning: " + "; ".join(warnings)


    return True, "All SK_ID_CURR values are consistent"





# Load all raw features

with open(CONFIG_DIR / "all_raw_features.json") as f:

    ALL_RAW_FEATURES = json.load(f)





def validate_input_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validates and cleans the input DataFrame for prediction.

    Args:
        df (pd.DataFrame): The input DataFrame.



    Returns:
        pd.DataFrame: The validated and cleaned DataFrame with columns

                      ordered as in ALL_RAW_FEATURES.



    Raises:
        ValueError: If the DataFrame is empty or contains missing required columns.

    """
    if df.empty:

        raise ValueError("Input DataFrame is empty.")



    # Convert all column names to string type

    df.columns = df.columns.astype(str)



    # Ensure all required columns are present

    current_columns = set(df.columns)

    required_columns = set(ALL_RAW_FEATURES)

    missing_columns = required_columns - current_columns



    if missing_columns:

        raise ValueError(f"Missing required columns: {', '.join(sorted(list(missing_columns)))}")



    # Remove extra columns and log a warning

    extra_columns = current_columns - required_columns

    if extra_columns:

        print(f"Warning: Extra columns found and removed: {', '.join(sorted(list(extra_columns)))}")

        # Drop extra columns

        df = df.drop(columns=list(extra_columns))



    # Ensure column order matches ALL_RAW_FEATURES

    df = df[ALL_RAW_FEATURES]



    return df



