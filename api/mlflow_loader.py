"""
MLflow Integration Module for API

This module provides utilities to load models from MLflow instead of local files.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

def load_model_from_mlflow(
    experiment_name: str = "credit_scoring_final_delivery",
    run_name: Optional[str] = None,
    fallback_path: Optional[Path] = None
):
    """
    Load model from MLflow with optional fallback to local file.
    
    Parameters:
    -----------
    experiment_name : str
        MLflow experiment name
    run_name : str, optional
        Specific run name to load. If None, loads the latest run.
    fallback_path : Path, optional
        Path to fallback pickle file if MLflow fails
        
    Returns:
    --------
    tuple: (model, metadata_dict)
    
    Example:
    --------
    >>> model, metadata = load_model_from_mlflow(
    ...     experiment_name="credit_scoring_final_delivery",
    ...     fallback_path=Path("models/production_model.pkl")
    ... )
    """
    
    try:
        import mlflow
        import mlflow.lightgbm
        
        # Set tracking URI to mlruns database (production)
        mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
        
        # Get experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if not experiment:
                raise ValueError(f"Experiment '{experiment_name}' not found in MLflow")
            
            exp_id = experiment.experiment_id
        except Exception as e:
            logger.warning(f"Could not get experiment from MLflow: {e}")
            if fallback_path and fallback_path.exists():
                return _load_fallback_model(fallback_path)
            raise
        
        # Get run
        if run_name:
            # Search for specific run by name
            runs = mlflow.search_runs(
                experiment_ids=[exp_id],
                filter_string=f"tags.mlflow.runName = '{run_name}'",
                max_results=1
            )
            if runs.empty:
                raise ValueError(f"Run '{run_name}' not found in experiment '{experiment_name}'")
            run_id = runs.iloc[0]['run_id']
        else:
            # Get latest run
            runs = mlflow.search_runs(
                experiment_ids=[exp_id],
                max_results=1,
                order_by=["start_time DESC"]
            )
            if runs.empty:
                raise ValueError(f"No runs found in experiment '{experiment_name}'")
            run_id = runs.iloc[0]['run_id']
        
        logger.info(f"Loading model from MLflow:")
        logger.info(f"  Experiment: {experiment_name}")
        logger.info(f"  Run ID: {run_id}")
        
        # Load model
        run = mlflow.get_run(run_id)
        
        # Try to load LightGBM model
        try:
            model = mlflow.lightgbm.load_model(f"runs:/{run_id}/model")
            logger.info("[OK] Loaded LightGBM model from MLflow")
        except Exception as e:
            logger.warning(f"Could not load LightGBM model: {e}. Trying pickle artifact...")
            # Try to load pickle artifact directly from artifact_uri
            import pickle
            
            # Get artifact_uri from run
            artifact_uri = run.info.artifact_uri
            
            # Convert artifact_uri to local path
            if artifact_uri.startswith('file:///'):
                artifacts_dir = Path(artifact_uri.replace('file:///', ''))
            elif artifact_uri.startswith('./'):
                artifacts_dir = Path(artifact_uri)
            else:
                # Fallback: construct path from run_id
                artifacts_dir = Path('mlruns') / run_id[:2] / run_id / 'artifacts'
            
            model_file = artifacts_dir / "production_model.pkl"
            
            if not model_file.exists():
                raise FileNotFoundError(f"No model artifact found at {model_file}")
            
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"[OK] Loaded model from pickle artifact: {model_file}")
        
        # Extract metadata
        metadata = {
            'source': 'mlflow',
            'experiment': experiment_name,
            'run_id': run_id,
            'run_name': run.info.run_name,
            'type': type(model).__name__,
            'status': run.info.status,
        }
        
        # Add parameters as metadata
        if run.data.params:
            metadata['parameters'] = run.data.params
        
        # Add metrics as metadata
        if run.data.metrics:
            metadata['metrics'] = run.data.metrics
        
        # Add tags as metadata
        if run.data.tags:
            metadata['tags'] = run.data.tags
        
        logger.info(f"✓ Model loaded successfully from MLflow")
        logger.info(f"  Run: {run.info.run_name}")
        logger.info(f"  Optimal Threshold: {metadata['parameters'].get('optimal_threshold', 'N/A')}")
        
        return model, metadata
        
    except Exception as e:
        logger.error(f"Error loading from MLflow: {e}")
        
        # Fallback to local file
        if fallback_path and fallback_path.exists():
            logger.info(f"Falling back to local file: {fallback_path}")
            return _load_fallback_model(fallback_path)
        
        raise


def _load_fallback_model(filepath: Path):
    """Load model from local pickle file."""
    import pickle
    from datetime import datetime
    
    logger.info(f"Loading model from file: {filepath}")
    
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    metadata = {
        'source': 'file',
        'path': str(filepath),
        'loaded_at': datetime.now().isoformat(),
        'type': type(model).__name__,
    }
    
    logger.info(f"✓ Model loaded from file: {filepath}")
    
    return model, metadata


def get_mlflow_run_info(
    experiment_name: str = "credit_scoring_final_delivery",
    run_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get detailed information about a specific MLflow run.
    
    Returns:
    --------
    dict: Run information including parameters, metrics, tags
    """
    
    import mlflow
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    # Get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        return {'error': f"Experiment '{experiment_name}' not found"}
    
    # Get run
    if run_name:
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{run_name}'",
            max_results=1
        )
        if runs.empty:
            return {'error': f"Run '{run_name}' not found"}
        run_id = runs.iloc[0]['run_id']
    else:
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=1,
            order_by=["start_time DESC"]
        )
        if runs.empty:
            return {'error': f"No runs found in experiment"}
        run_id = runs.iloc[0]['run_id']
    
    # Get run details
    run = mlflow.get_run(run_id)
    
    return {
        'experiment_name': experiment_name,
        'run_id': run_id,
        'run_name': run.info.run_name,
        'status': run.info.status,
        'parameters': run.data.params,
        'metrics': run.data.metrics,
        'tags': run.data.tags,
    }


def list_mlflow_experiments():
    """List all MLflow experiments."""
    import mlflow
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    experiments = mlflow.search_experiments()
    
    return [
        {
            'id': exp.experiment_id,
            'name': exp.name,
            'lifecycle_stage': exp.lifecycle_stage,
        }
        for exp in experiments
    ]
