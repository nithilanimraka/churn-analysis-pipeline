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


def initialize_inference_system(
    model_path: str = 'artifacts/models/churn_model.pkl',
    encoders_path: str = 'artifacts/encode',
    scaler_path: str = 'artifacts/models/scaler.pkl'
) -> ModelInference:
    """
    Initialize the inference system with comprehensive logging and error handling.
    
    Args:
        model_path: Path to the trained model
        encoders_path: Path to the encoders directory
        scaler_path: Path to the saved scaler
        
    Returns:
        Initialized ModelInference instance
        
    Raises:
        Exception: If initialization fails
    """
    logger.info(f"\n{'='*80}")
    logger.info("INITIALIZING STREAMING INFERENCE SYSTEM")
    logger.info(f"{'='*80}")
    
    try:
        # Initialize model inference
        logger.info("Creating ModelInference instance...")
        inference = ModelInference(model_path)
        
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