import os
import sys
import joblib
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_pipeline import data_pipeline
from typing import Dict, Any, Tuple, Optional
import json
from pathlib import Path


# Add parent directory to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator
from src.model_building import XGBoostModelBuilder, RandomForestModelBuilder

from utils.config import get_model_config, get_data_paths, get_mlflow_config
from utils.mlflow_utils import MLflowTracker, create_mlflow_run_tags
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_model_performance_visualizations(model, X_test: pd.DataFrame, y_test: pd.Series, 
                                          evaluation_results: dict, artifacts_dir: str, model_name: str):
    """Create comprehensive model performance visualizations."""
    try:
        # Create model-specific directory
        model_dir = os.path.join(artifacts_dir, f"model_performance_{model_name}")
        os.makedirs(model_dir, exist_ok=True)
        
        # 1. Confusion Matrix Heatmap
        if 'cm' in evaluation_results:
            plt.figure(figsize=(8, 6))
            sns.heatmap(evaluation_results['cm'], annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Retain', 'Churn'], yticklabels=['Retain', 'Churn'])
            plt.title(f'{model_name} - Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            cm_path = os.path.join(model_dir, f'confusion_matrix_{model_name}.png')
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        
        # 2. Feature Importance (if available)
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(12, 8))
            feature_importance = pd.DataFrame({
                'feature': X_test.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            # Plot top 15 features
            top_features = feature_importance.tail(15)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'{model_name} - Top 15 Feature Importances')
            plt.tight_layout()
            
            importance_path = os.path.join(model_dir, f'feature_importance_{model_name}.png')
            plt.savefig(importance_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save feature importance as JSON
            importance_json_path = os.path.join(model_dir, f'feature_importance_{model_name}.json')
            feature_importance.to_json(importance_json_path, indent=2)
            
        
        # 3. ROC Curve (if probabilities available)
        try:
            from sklearn.metrics import roc_curve, auc
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{model_name} - ROC Curve')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            roc_path = os.path.join(model_dir, f'roc_curve_{model_name}.png')
            plt.savefig(roc_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            
        except Exception as e:
            logger.warning(f"Could not create ROC curve: {str(e)}")
        
        # 4. Prediction Distribution
        try:
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Prediction distribution
            pred_counts = pd.Series(y_pred).value_counts()
            axes[0].bar(['Retain', 'Churn'], [pred_counts.get(0, 0), pred_counts.get(1, 0)])
            axes[0].set_title('Prediction Distribution')
            axes[0].set_ylabel('Count')
            
            # Probability distribution
            axes[1].hist(y_proba, bins=30, alpha=0.7, edgecolor='black')
            axes[1].set_xlabel('Churn Probability')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Churn Probability Distribution')
            
            plt.suptitle(f'{model_name} - Prediction Analysis')
            plt.tight_layout()
            
            pred_dist_path = os.path.join(model_dir, f'prediction_distribution_{model_name}.png')
            plt.savefig(pred_dist_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            
        except Exception as e:
            logger.warning(f"Could not create prediction distribution: {str(e)}")
        
        logger.info(f"✓ Model performance visualizations created for {model_name}")
        
    except Exception as e:
        logger.error(f"✗ Failed to create model performance visualizations: {str(e)}")


def log_model_metadata(model, model_name: str, model_params: dict, training_time: float, artifacts_dir: str):
    """Log comprehensive model metadata."""
    try:
        metadata = {
            'model_name': model_name,
            'model_type': type(model).__name__,
            'model_parameters': model_params,
            'training_time_seconds': training_time,
            'sklearn_version': None,
            'model_size_mb': None,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Try to get sklearn version
        try:
            import sklearn
            metadata['sklearn_version'] = sklearn.__version__
        except:
            pass
        
        # Try to get model size
        try:
            model_path = os.path.join(artifacts_dir, f'temp_{model_name}_model.pkl')
            joblib.dump(model, model_path)
            metadata['model_size_mb'] = os.path.getsize(model_path) / (1024**2)
            os.remove(model_path)  # Clean up temp file
        except:
            pass
        
        # Save metadata
        metadata_path = os.path.join(artifacts_dir, f'model_metadata_{model_name}.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        
        logger.info(f"✓ Model metadata logged for {model_name}")
        
    except Exception as e:
        logger.error(f"✗ Failed to log model metadata: {str(e)}")


def training_pipeline(
                    data_path: str = 'data/raw/CustomerChurnRaw.csv',
                    model_params: Optional[Dict[str, Any]] = None,
                    test_size: float = 0.2, random_state: int = 42,
                    model_path: str = 'artifacts/models/churn_model.pkl',
                    ):
    
    # Initialize MLflow tracker
    mlflow_tracker = None
    
    try:
        # Initialize MLflow
        mlflow_tracker = MLflowTracker()
        mlflow_config = get_mlflow_config()
        tags = create_mlflow_run_tags('model_training', {'model_type': 'XGBoost'})
        mlflow_tracker.start_run(run_name='model_training', tags=tags)
        logger.info("MLflow tracking initialized for training pipeline")
    except Exception as mlflow_error:
        logger.warning(f"MLflow initialization failed: {mlflow_error}. Continuing without MLflow tracking.")
    
    try:
        data_pipeline()
        
        # Create artifacts directory for visualizations
        run_artifacts_dir = os.path.join('artifacts', 'models', 'model_performance_XGBoost')
        os.makedirs(run_artifacts_dir, exist_ok=True)

        # Load training data with logging
        logger.info("Loading training and test datasets...")
        data_paths = get_data_paths()
        X_train = pd.read_csv(data_paths['X_train'])
        Y_train = pd.read_csv(data_paths['Y_train'])
        X_test = pd.read_csv(data_paths['X_test'])
        Y_test = pd.read_csv(data_paths['Y_test'])
        
        logger.info(f"✓ Data loaded - Training: {X_train.shape}, Test: {X_test.shape}")
        

        # Model building and training with timing
        logger.info("Building and training XGBoost model...")
        import time
        training_start_time = time.time()
        
        model_builder = XGBoostModelBuilder(**model_params)
        model = model_builder.build_model()

        trainer = ModelTrainer()
        model, train_score = trainer.train(
                                model=model,
                                X_train=X_train,
                                Y_train=Y_train.squeeze()
                                )
        
        training_end_time = time.time()
        training_time = training_end_time - training_start_time
        logger.info(f"✓ Model training completed in {training_time:.2f} seconds")
        
        # Save model
        trainer.save_model(model, model_path)
        logger.info(f"✓ Model saved to: {model_path}")
        
        
        # Model evaluation with comprehensive logging
        logger.info("Evaluating model performance...")
        evaluator = ModelEvaluator(model, 'XGBoost')
        evaluation_results = evaluator.evaluate(X_test, Y_test.squeeze())
        evaluation_results_cp = evaluation_results.copy()
        
        # Create comprehensive performance visualizations (UNCOMMENTED)
        create_model_performance_visualizations(
            model, X_test, Y_test.squeeze(), evaluation_results, 
            run_artifacts_dir, 'XGBoost'
        )
        
        # Log model metadata
        model_config = get_model_config()['model_params']
        log_model_metadata(model, 'XGBoost', model_config, training_time, run_artifacts_dir)
        
        # Log training metrics (remove confusion matrix and predictions for summary)
        if 'confusion_matrix' in evaluation_results_cp:
            del evaluation_results_cp['confusion_matrix']
        if 'predictions' in evaluation_results_cp:
            del evaluation_results_cp['predictions']
        
        # Add additional training metrics
        evaluation_results_cp.update({
            'train_score': train_score,
            'training_time_seconds': training_time,
            'model_complexity': model.n_estimators if hasattr(model, 'n_estimators') else 0,
            'max_depth': model.max_depth if hasattr(model, 'max_depth') else 0
        })

        
        # Log training summary
        training_summary = {
            'model_type': 'XGBoost',
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features_used': X_train.shape[1],
            'training_time': training_time,
            'model_path': model_path,
            'performance_metrics': evaluation_results_cp,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Save training summary
        summary_path = os.path.join(run_artifacts_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(training_summary, f, indent=2, default=str)
        
        # Log to MLflow
        if mlflow_tracker:
            try:
                # Log model parameters
                mlflow_tracker.log_training_metrics(
                    model=model,
                    training_metrics={
                        'train_score': train_score,
                        'training_time_seconds': training_time,
                        'test_accuracy': evaluation_results.get('accuracy', 0),
                        'test_precision': evaluation_results.get('precision', 0),
                        'test_recall': evaluation_results.get('recall', 0),
                        'test_f1_score': evaluation_results.get('f1_score', 0),
                        'test_samples': len(X_test),
                        'train_samples': len(X_train),
                        'num_features': X_train.shape[1]
                    },
                    model_params=model_config
                )
                
                # Log evaluation results
                mlflow_tracker.log_evaluation_metrics(evaluation_results)
                
                # Log all visualization artifacts
                mlflow_tracker.log_artifacts(run_artifacts_dir, "visualizations")
                
                # Auto-promote model if accuracy >= threshold
                accuracy = evaluation_results.get('accuracy', 0)
                auto_promote_threshold = mlflow_config.get('auto_promote_threshold', 0.8)
                latest_version = mlflow_tracker.get_latest_model_version()
                
                mlflow_tracker.auto_promote_model(
                    accuracy=accuracy,
                    version=latest_version,
                    threshold=auto_promote_threshold
                )
                
                logger.info("Logged all metrics and artifacts to MLflow")
            except Exception as e:
                logger.warning(f"Failed to log to MLflow: {e}")
    
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise
    finally:
        # End MLflow run
        if mlflow_tracker:
            try:
                mlflow_tracker.end_run()
            except Exception as e:
                logger.warning(f"Failed to end MLflow run: {e}")


if __name__ == '__main__':
    model_config = get_model_config()
    model_params = model_config.get('model_params') if model_config else {}
    if model_params is None:
        model_params = {}
    training_pipeline(model_params=model_params)