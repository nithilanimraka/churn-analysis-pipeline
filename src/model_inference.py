import json
import logging
import os
import joblib
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler

# Add parent directory to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config import get_encoding_config, get_scaling_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
Example input data for Telco Customer Churn prediction:
{
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
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 29.85,
  "TotalCharges": 29.85
}
"""
class ModelInference:
    """
    Enhanced model inference class with comprehensive logging and error handling.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the model inference system.
        
        Args:
            model_path: Path to the trained model file
            
        Raises:
            ValueError: If model_path is invalid
            FileNotFoundError: If model file doesn't exist
        """
        logger.info(f"\n{'='*60}")
        logger.info("INITIALIZING MODEL INFERENCE")
        logger.info(f"{'='*60}")
        
        if not model_path or not isinstance(model_path, str):
            logger.error("✗ Invalid model path provided")
            raise ValueError("Invalid model path provided")
            
        self.model_path = model_path
        self.encoders = {}
        self.scaler = None
        self.model = None
        
        logger.info(f"Model Path: {model_path}")
        
        try:
            # Load model and configurations
            self.load_model()
            self.encoding_config = get_encoding_config()
            self.scaling_config = get_scaling_config()
            
            logger.info("✓ Model inference system initialized successfully")
            logger.info(f"{'='*60}\n")
            
        except Exception as e:
            logger.error(f"✗ Failed to initialize model inference: {str(e)}")
            raise

    def load_model(self) -> None:
        """
        Load the trained model from disk with validation.
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: For any loading errors
        """
        logger.info("Loading trained model...")
        
        if not os.path.exists(self.model_path):
            logger.error(f"✗ Model file not found: {self.model_path}")
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            self.model = joblib.load(self.model_path)
            file_size = os.path.getsize(self.model_path) / (1024**2)  # MB
            
            logger.info(f"✓ Model loaded successfully:")
            logger.info(f"  • Model Type: {type(self.model).__name__}")
            logger.info(f"  • File Size: {file_size:.2f} MB")
            
        except Exception as e:
            logger.error(f"✗ Failed to load model: {str(e)}")
            raise

    def load_encoders(self, encoders_dir: str) -> None:
        """
        Load feature encoders from directory with validation and logging.
        
        Args:
            encoders_dir: Directory containing encoder JSON files
            
        Raises:
            FileNotFoundError: If encoders directory doesn't exist
            Exception: For any loading errors
        """
        logger.info(f"\n{'='*50}")
        logger.info("LOADING FEATURE ENCODERS")
        logger.info(f"{'='*50}")
        
        if not os.path.exists(encoders_dir):
            logger.error(f"✗ Encoders directory not found: {encoders_dir}")
            raise FileNotFoundError(f"Encoders directory not found: {encoders_dir}")
        
        try:
            encoder_files = [f for f in os.listdir(encoders_dir) if f.endswith('_encoder.json')]
            
            if not encoder_files:
                logger.warning("⚠ No encoder files found in directory")
                return
            
            logger.info(f"Found {len(encoder_files)} encoder files")
            
            for file in encoder_files:
                feature_name = file.split('_encoder.json')[0]
                file_path = os.path.join(encoders_dir, file)
                
                with open(file_path, 'r') as f:
                    encoder_data = json.load(f)
                    self.encoders[feature_name] = encoder_data
                    
                logger.info(f"  ✓ Loaded encoder for '{feature_name}': {len(encoder_data)} mappings")
            
            logger.info(f"✓ All encoders loaded successfully")
            logger.info(f"{'='*50}\n")
            
        except Exception as e:
            logger.error(f"✗ Failed to load encoders: {str(e)}")
            raise

    def load_scaler(self, scaler_path: str) -> None:
        """
        Load the feature scaler from disk with validation and logging.
        
        Args:
            scaler_path: Path to the saved scaler file
            
        Raises:
            FileNotFoundError: If scaler file doesn't exist
            Exception: For any loading errors
        """
        logger.info(f"\n{'='*50}")
        logger.info("LOADING FEATURE SCALER")
        logger.info(f"{'='*50}")
        
        if not os.path.exists(scaler_path):
            logger.error(f"✗ Scaler file not found: {scaler_path}")
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        
        try:
            self.scaler = joblib.load(scaler_path)
            file_size = os.path.getsize(scaler_path) / 1024  # KB
            
            logger.info(f"✓ Scaler loaded successfully:")
            logger.info(f"  • Scaler Type: {type(self.scaler).__name__}")
            logger.info(f"  • File Size: {file_size:.2f} KB")
            logger.info(f"{'='*50}\n")
            
        except Exception as e:
            logger.error(f"✗ Failed to load scaler: {str(e)}")
            raise

    def preprocess_input(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess input data for model prediction with comprehensive logging.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Preprocessed DataFrame ready for prediction
            
        Raises:
            ValueError: If input data is invalid
            Exception: For any preprocessing errors
        """
        logger.info(f"\n{'='*50}")
        logger.info("PREPROCESSING INPUT DATA")
        logger.info(f"{'='*50}")
        
        if not data or not isinstance(data, dict):
            logger.error("✗ Input data must be a non-empty dictionary")
            raise ValueError("Input data must be a non-empty dictionary")
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame([data])
            logger.info(f"✓ Input data converted to DataFrame: {df.shape}")
            logger.info(f"  • Input features: {list(df.columns)}")
            
            # Handle TotalCharges - convert to numeric (may be string with spaces)
            if 'TotalCharges' in df.columns:
                logger.info("Converting TotalCharges to numeric...")
                df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
                if df['TotalCharges'].isna().any():
                    logger.warning("  ⚠ TotalCharges contained non-numeric values, converted to NaN")
                    df['TotalCharges'] = df['TotalCharges'].fillna(0)
            
            # Apply nominal encoders (gender, Partner, PhoneService, etc.)
            if self.encoders:
                logger.info("Applying feature encoders...")
                for col, encoder in self.encoders.items():
                    # Skip Churn encoder (target variable)
                    if col == 'Churn':
                        continue
                        
                    if col in df.columns:
                        original_value = df[col].iloc[0]
                        df[col] = df[col].map(encoder)
                        encoded_value = df[col].iloc[0]
                        
                        if pd.isna(encoded_value):
                            logger.warning(f"  ⚠ '{col}': '{original_value}' not in encoder mappings")
                        else:
                            logger.info(f"  ✓ Encoded '{col}': {original_value} → {encoded_value}")
            else:
                logger.warning("⚠ No encoders loaded - skipping encoding step")

            # Apply ordinal encoding for Contract
            if 'ordinal_mappings' in self.encoding_config:
                logger.info("Applying ordinal encoding...")
                for col, mapping in self.encoding_config['ordinal_mappings'].items():
                    if col in df.columns:
                        original_value = df[col].iloc[0]
                        df[col] = df[col].map(mapping)
                        encoded_value = df[col].iloc[0]
                        
                        if pd.isna(encoded_value):
                            logger.warning(f"  ⚠ '{col}': '{original_value}' not in ordinal mappings")
                        else:
                            logger.info(f"  ✓ Ordinal encoded '{col}': {original_value} → {encoded_value}")

            # Drop customerID if present
            if 'customerID' in df.columns:
                df = df.drop(columns=['customerID'])
                logger.info("  ✓ Dropped 'customerID' column")
            
            # Apply feature scaling
            if self.scaler is not None and 'columns_to_scale' in self.scaling_config:
                columns_to_scale = self.scaling_config['columns_to_scale']
                existing_scale_columns = [col for col in columns_to_scale if col in df.columns]
                
                if existing_scale_columns:
                    logger.info(f"Applying feature scaling to: {existing_scale_columns}")
                    for col in existing_scale_columns:
                        original_value = df[col].iloc[0]
                        logger.info(f"  • '{col}' before scaling: {original_value:.2f}")
                    
                    df[existing_scale_columns] = self.scaler.transform(df[existing_scale_columns])
                    
                    for col in existing_scale_columns:
                        scaled_value = df[col].iloc[0]
                        logger.info(f"  ✓ '{col}' after scaling: {scaled_value:.4f}")
                else:
                    logger.warning("⚠ No columns to scale found in input data")
            else:
                logger.warning("⚠ Scaler not loaded - skipping scaling step")
            
            logger.info(f"✓ Preprocessing completed - Final shape: {df.shape}")
            logger.info(f"  • Final features: {list(df.columns)}")
            logger.info(f"{'='*50}\n")
            
            return df
            
        except Exception as e:
            logger.error(f"✗ Preprocessing failed: {str(e)}")
            raise
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, str]:
        """
        Make prediction on input data with comprehensive logging.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Dictionary containing prediction status and confidence
            
        Raises:
            ValueError: If input data is invalid
            Exception: For any prediction errors
        """
        logger.info(f"\n{'='*60}")
        logger.info("MAKING PREDICTION")
        logger.info(f"{'='*60}")
        
        if not data:
            logger.error("✗ Input data cannot be empty")
            raise ValueError("Input data cannot be empty")
        
        if self.model is None:
            logger.error("✗ Model not loaded")
            raise ValueError("Model not loaded")
        
        try:
            # Preprocess input data
            processed_data = self.preprocess_input(data)
            
            # Make prediction
            logger.info("Generating predictions...")
            y_pred = self.model.predict(processed_data)
            y_proba = self.model.predict_proba(processed_data)[:, 1]
            
            # Process results
            prediction = int(y_pred[0])
            probability = float(y_proba[0])
            
            status = 'Churn' if prediction == 1 else 'Retain'
            confidence = round(probability * 100, 2)
            
            result = {
                "Status": status,
                "Confidence": f"{confidence}%"
            }
            
            logger.info("✓ Prediction completed:")
            logger.info(f"  • Raw Prediction: {prediction}")
            logger.info(f"  • Raw Probability: {probability:.4f}")
            logger.info(f"  • Final Status: {status}")
            logger.info(f"  • Confidence: {confidence}%")
            logger.info(f"{'='*60}\n")
            
            return result
            
        except Exception as e:
            logger.error(f"✗ Prediction failed: {str(e)}")
            raise