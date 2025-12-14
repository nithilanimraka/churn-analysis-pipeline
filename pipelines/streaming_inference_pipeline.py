import os
import sys
import json
import pandas as pd
import logging
import numpy as np
import time
from typing import Dict, Any, Optional, List
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model_inference import ModelInference
from utils.config import get_model_config
from utils.mlflow_utils import MLflowTracker


def initialize_inference_system(
    model_path: str = 'artifacts/models/churn_model.pkl',
    encoders_path: str = 'artifacts/encode',
    scaler_path: str = 'artifacts/models/scaler.pkl',
    use_mlflow_registry: bool = True
) -> ModelInference:
    """
    Initialize the inference system with comprehensive logging and error handling.
    Implements fallback chain: Production → Staging → None → Local pickle file
    
    Args:
        model_path: Path to the trained model (fallback)
        encoders_path: Path to the encoders directory
        scaler_path: Path to the saved scaler
        use_mlflow_registry: Whether to attempt loading from MLflow registry
        
    Returns:
        Initialized ModelInference instance
        
    Raises:
        Exception: If initialization fails
    """
    logger.info(f"\n{'='*80}")
    logger.info("INITIALIZING STREAMING INFERENCE SYSTEM")
    logger.info(f"{'='*80}")
    
    model_loaded = False
    model = None
    
    # Try loading from MLflow Model Registry first (with fallback chain)
    if use_mlflow_registry:
        try:
            logger.info("Attempting to load model from MLflow Model Registry...")
            mlflow_tracker = MLflowTracker()
            
            # Fallback chain: Production → Staging → None (latest)
            model = mlflow_tracker.load_model_from_registry(stage="Production")
            
            if model is not None:
                model_loaded = True
                logger.info("✓ Model loaded from MLflow registry")
            else:
                logger.warning("⚠ Could not load model from MLflow registry, falling back to local file")
        except Exception as e:
            logger.warning(f"⚠ MLflow registry load failed: {e}, falling back to local file")
    
    try:
        # If MLflow failed or was disabled, load from local file
        if not model_loaded:
            logger.info(f"Loading model from local file: {model_path}")
            inference = ModelInference(model_path)
        else:
            # Create ModelInference with the MLflow model
            # We need to save it temporarily or modify ModelInference to accept model object
            import joblib
            temp_model_path = 'artifacts/models/temp_mlflow_model.pkl'
            os.makedirs(os.path.dirname(temp_model_path), exist_ok=True)
            joblib.dump(model, temp_model_path)
            inference = ModelInference(temp_model_path)
            logger.info("✓ Using model from MLflow registry")
        
        # Load encoders if directory exists
        if os.path.exists(encoders_path):
            logger.info(f"Loading encoders from: {encoders_path}")
            inference.load_encoders(encoders_path)
        else:
            logger.warning(f"⚠ Encoders directory not found: {encoders_path}")
            logger.info("Proceeding without encoders (may affect prediction accuracy)")
        
        # Load scaler if exists
        if os.path.exists(scaler_path):
            logger.info(f"Loading scaler from: {scaler_path}")
            inference.load_scaler(scaler_path)
        else:
            logger.warning(f"⚠ Scaler file not found: {scaler_path}")
            logger.info("Proceeding without scaler (may affect prediction accuracy)")
        
        logger.info("✓ Streaming inference system initialized successfully")
        logger.info(f"{'='*80}\n")
        
        return inference
        
    except Exception as e:
        logger.error(f"✗ Failed to initialize inference system: {str(e)}")
        raise


def streaming_inference(
    inference: ModelInference, 
    data: Dict[str, Any]
) -> Dict[str, str]:
    """
    Perform streaming inference with comprehensive logging and error handling.
    
    Args:
        inference: Initialized ModelInference instance
        data: Input data dictionary for prediction
        
    Returns:
        Prediction result dictionary
        
    Raises:
        ValueError: If input parameters are invalid
        Exception: For any prediction errors
    """
    logger.info(f"\n{'='*70}")
    logger.info("STREAMING INFERENCE REQUEST")
    logger.info(f"{'='*70}")
    
    # Input validation
    if inference is None:
        logger.error("✗ ModelInference instance cannot be None")
        raise ValueError("ModelInference instance cannot be None")
    
    if not data or not isinstance(data, dict):
        logger.error("✗ Input data must be a non-empty dictionary")
        raise ValueError("Input data must be a non-empty dictionary")
    
    try:
        logger.info("Processing inference request...")
        logger.info(f"Input data keys: {list(data.keys())}")
        
        # Time the inference
        start_time = time.time()
        
        # Perform prediction
        prediction_result = inference.predict(data)
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        logger.info("✓ Streaming inference completed successfully")
        logger.info(f"Result: {prediction_result}")
        logger.info(f"Inference time: {inference_time*1000:.2f}ms")
        logger.info(f"{'='*70}\n")
        
        return prediction_result
        
    except Exception as e:
        logger.error(f"✗ Streaming inference failed: {str(e)}")
        raise


# Initialize the global inference system
try:
    logger.info("Initializing global inference system...")
    inference = initialize_inference_system()
except Exception as e:
    logger.error(f"Failed to initialize global inference system: {str(e)}")
    inference = None


if __name__ == '__main__':
    # Example telco customer data for churn prediction
    sample_customer_data = {
        "customerID": "7590-VHVEG",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": "29.85"
    }
    
    if inference is not None:
        pred = streaming_inference(inference, sample_customer_data)
        print(f"\n{'='*70}")
        print(f"PREDICTION RESULT: {pred}")
        print(f"{'='*70}")
    else:
        logger.error("Cannot perform inference - initialization failed")