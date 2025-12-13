"""Production logging utility for API monitoring."""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Configure production logger
def setup_production_logger(log_dir: str = "logs") -> logging.Logger:
    """Set up JSON logger for production monitoring."""
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True, parents=True)
    
    logger = logging.getLogger("credit_scoring_api")
    logger.setLevel(logging.INFO)
    
    # File handler for predictions
    prediction_handler = logging.FileHandler(
        log_path / "predictions.jsonl",
        encoding="utf-8"
    )
    prediction_handler.setLevel(logging.INFO)
    
    # File handler for errors
    error_handler = logging.FileHandler(
        log_path / "errors.jsonl",
        encoding="utf-8"
    )
    error_handler.setLevel(logging.ERROR)
    
    # Console handler for development
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    logger.addHandler(prediction_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)
    
    return logger


def log_prediction(
    logger: logging.Logger,
    sk_id_curr: int,
    probability: float,
    prediction: int,
    risk_level: str,
    processing_time_ms: float,
    source: str = "api",
    metadata: Dict[str, Any] = None
) -> None:
    """Log a single prediction in JSON format."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": "prediction",
        "sk_id_curr": sk_id_curr,
        "probability": round(probability, 4),
        "prediction": prediction,
        "risk_level": risk_level,
        "processing_time_ms": round(processing_time_ms, 2),
        "source": source,
    }
    
    if metadata:
        log_entry["metadata"] = metadata
    
    logger.info(json.dumps(log_entry))


def log_batch_prediction(
    logger: logging.Logger,
    predictions: List[Dict[str, Any]],
    total_time_ms: float,
    num_applications: int,
    metadata: Dict[str, Any] = None
) -> None:
    """Log batch prediction summary."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": "batch_prediction",
        "num_applications": num_applications,
        "total_time_ms": round(total_time_ms, 2),
        "avg_time_per_app_ms": round(total_time_ms / num_applications, 2) if num_applications > 0 else 0,
        "risk_distribution": {
            "LOW": sum(1 for p in predictions if p.get("risk_level") == "LOW"),
            "MEDIUM": sum(1 for p in predictions if p.get("risk_level") == "MEDIUM"),
            "HIGH": sum(1 for p in predictions if p.get("risk_level") == "HIGH"),
        },
        "probability_stats": {
            "min": round(min(p.get("probability", 0) for p in predictions), 4) if predictions else 0,
            "max": round(max(p.get("probability", 0) for p in predictions), 4) if predictions else 0,
            "avg": round(sum(p.get("probability", 0) for p in predictions) / len(predictions), 4) if predictions else 0,
        }
    }
    
    if metadata:
        log_entry["metadata"] = metadata
    
    logger.info(json.dumps(log_entry))


def log_error(
    logger: logging.Logger,
    error_type: str,
    error_message: str,
    endpoint: str = None,
    metadata: Dict[str, Any] = None
) -> None:
    """Log an error event."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": "error",
        "error_type": error_type,
        "error_message": error_message,
    }
    
    if endpoint:
        log_entry["endpoint"] = endpoint
    
    if metadata:
        log_entry["metadata"] = metadata
    
    logger.error(json.dumps(log_entry))
