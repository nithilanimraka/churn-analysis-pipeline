import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union
import time
import warnings
import logging
from sklearn.metrics import (
                            accuracy_score,
                            precision_score,
                            recall_score,
                            f1_score,
                            confusion_matrix,
                            )

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Enhanced model evaluator with comprehensive logging and error handling.
    """
    
    def __init__(
                self,
                model,
                model_name: str
                ):
        """
        Initialize the model evaluator.
        
        Args:
            model: The trained model to evaluate
            model_name: Name of the model for logging purposes
        """
        if model is None:
            raise ValueError("Model cannot be None")
        if not model_name:
            raise ValueError("Model name cannot be empty")
            
        self.model = model
        self.model_name = model_name
        self.evaluation_results = {}
        logger.info(f"ModelEvaluator initialized for {model_name}")

    def evaluate(
                self,
                X_test: Union[pd.DataFrame, np.ndarray],
                Y_test: Union[pd.Series, np.ndarray],
                average: str = 'binary'
                ) -> Dict[str, Any]:
        """
        Evaluate the model on test data with comprehensive logging.
        
        Args:
            X_test: Test features
            Y_test: Test targets
            average: Averaging strategy for multi-class ('binary', 'micro', 'macro', 'weighted')
            
        Returns:
            Dictionary containing evaluation metrics
            
        Raises:
            ValueError: If input data is invalid
            Exception: For any evaluation errors
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"MODEL EVALUATION - {self.model_name}")
        logger.info(f"{'='*80}")
        
        try:
            # Input validation
            if X_test is None or Y_test is None:
                raise ValueError("Test data cannot be None")
                
            if len(X_test) == 0 or len(Y_test) == 0:
                raise ValueError("Test data cannot be empty")
                
            if len(X_test) != len(Y_test):
                raise ValueError(f"Feature and target length mismatch: {len(X_test)} vs {len(Y_test)}")
            
            logger.info(f"Evaluation Configuration:")
            logger.info(f"  • Model: {self.model_name}")
            logger.info(f"  • Test Samples: {len(X_test):,}")
            logger.info(f"  • Features: {X_test.shape[1] if hasattr(X_test, 'shape') else 'Unknown'}")
            
            # Make predictions
            logger.info("Generating predictions...")
            start_time = time.time()
            Y_pred = self.model.predict(X_test)
            prediction_time = time.time() - start_time
            logger.info(f"✓ Predictions generated in {prediction_time:.2f} seconds")
            
            # Calculate metrics
            logger.info("\nCalculating evaluation metrics...")
            
            logger.info("  • Computing confusion matrix...")
            cm = confusion_matrix(Y_test, Y_pred)
            
            logger.info("  • Computing accuracy...")
            accuracy = accuracy_score(Y_test, Y_pred)
            
            logger.info("  • Computing precision...")
            precision = precision_score(Y_test, Y_pred, average=average, zero_division=0)
            
            logger.info("  • Computing recall...")
            recall = recall_score(Y_test, Y_pred, average=average, zero_division=0)
            
            logger.info("  • Computing F1 score...")
            f1 = f1_score(Y_test, Y_pred, average=average, zero_division=0)
            
            # Store results
            self.evaluation_results = {
                'model_name': self.model_name,
                'predictions': Y_pred,
                'confusion_matrix': cm,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'test_samples': len(X_test),
                'prediction_time': prediction_time
            }
            
            # Log results
            logger.info(f"\n{'='*80}")
            logger.info("EVALUATION RESULTS")
            logger.info(f"{'='*80}")
            logger.info(f"Model: {self.model_name}")
            logger.info(f"Test Samples: {len(X_test):,}")
            logger.info(f"\nConfusion Matrix:")
            logger.info(f"\n{cm}")
            logger.info(f"\nPerformance Metrics:")
            logger.info(f"  • Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
            logger.info(f"  • Precision: {precision:.4f} ({precision*100:.2f}%)")
            logger.info(f"  • Recall:    {recall:.4f} ({recall*100:.2f}%)")
            logger.info(f"  • F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
            logger.info(f"\nPrediction Time: {prediction_time:.2f} seconds")
            logger.info(f"{'='*80}\n")
            
            return self.evaluation_results
            
        except ValueError as e:
            logger.error(f"✗ Value error during evaluation: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"✗ Evaluation failed: {str(e)}")
            raise
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get the stored evaluation results.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.evaluation_results:
            logger.warning("No evaluation results available. Run evaluate() first.")
        return self.evaluation_results