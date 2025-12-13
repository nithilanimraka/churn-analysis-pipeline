import joblib
import os
import logging
from typing import Dict, Any
from datetime import datetime
from xgboost import XGBClassifier
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseModelBuilder(ABC):
    def __init__(
                self,
                model_name: str,
                **kwargs
                ):
        self.model_name = model_name
        self.model = None
        self.model_params = kwargs
        logger.info(f"Initialized {model_name} model builder with params: {kwargs}")
    
    @abstractmethod
    def build_model(self):
        pass

    def save_model(self, filepath):
        try:
            if self.model is None:
                raise ValueError("No model to save. Build the model first.")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            logger.info(f"Saving {self.model_name} model to {filepath}")
            joblib.dump(self.model, filepath)
            logger.info(f"Successfully saved {self.model_name} model")
            
        except ValueError as e:
            logger.error(f"Value error while saving model: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error saving model to {filepath}: {str(e)}")
            raise
        
    def load_model(self, filepath):
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Model file not found: {filepath}")
            
            logger.info(f"Loading {self.model_name} model from {filepath}")
            self.model = joblib.load(filepath)
            logger.info(f"Successfully loaded {self.model_name} model")
            
        except FileNotFoundError as e:
            logger.error(f"File not found error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error loading model from {filepath}: {str(e)}")
            raise

class RandomForestModelBuilder(BaseModelBuilder):
    def __init__(self, **kwargs):
        default_params = {
                        'max_depth': 10,
                        'n_estimators': 100,
                        'min_samples_split': 2,
                        'min_samples_leaf': 1,
                        'random_state': 42
                        }
        default_params.update(kwargs)
        super().__init__('RandomForest', **default_params)

    def build_model(self):
        try:
            logger.info(f"Building RandomForest model with params: {self.model_params}")
            self.model = RandomForestClassifier(**self.model_params)
            logger.info("RandomForest model built successfully")
            return self.model
        except Exception as e:
            logger.error(f"Error building RandomForest model: {str(e)}")
            raise
    
class XGBoostModelBuilder(BaseModelBuilder):
    def __init__(self, **kwargs):
        default_params = {
                        'max_depth': 10,
                        'n_estimators': 100,
                        'random_state': 42
                        }
        default_params.update(kwargs)
        super().__init__('XGBoost', **default_params)

    def build_model(self):
        try:
            logger.info(f"Building XGBoost model with params: {self.model_params}")
            self.model = XGBClassifier(**self.model_params)
            logger.info("XGBoost model built successfully")
            return self.model
        except Exception as e:
            logger.error(f"Error building XGBoost model: {str(e)}")
            raise