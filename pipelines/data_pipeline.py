import os
import sys
import pandas as pd
from typing import Dict
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_ingestion import DataIngestorCSV
from src.handle_missing_values import DropMissingValuesStrategy, FillMissingValuesStrategy
from src.feature_encoding import OrdinalEncodingStrategy, NominalEncodingStrategy
from src.feature_scaling import MinMaxScalingStrategy
from src.data_spiltter import SimpleTrainTestSplitStrategy
from src.outlier_detection import OutlierDetector, IQROutlierDetection
from utils.config import get_data_paths, get_columns, get_outlier_config, get_binning_config, get_encoding_config, get_scaling_config, get_splitting_config
from utils.mlflow_utils import MLflowTracker, create_mlflow_run_tags

def data_pipeline(
                    data_path: str='data/raw/CustomerChurnRaw.csv', 
                    target_column: str='Churn', 
                    test_size: float=0.2, 
                    force_rebuild: bool=False
                    ) -> Dict[str, pd.DataFrame]:
    
    # Initialize MLflow tracker
    mlflow_tracker = None
    
    try:
        logger.info("="*80)
        logger.info("Starting Data Pipeline")
        logger.info("="*80)
        
        # Initialize MLflow
        try:
            mlflow_tracker = MLflowTracker()
            tags = create_mlflow_run_tags('data_preprocessing', {'force_rebuild': str(force_rebuild)})
            mlflow_tracker.start_run(run_name='data_preprocessing', tags=tags)
            logger.info("MLflow tracking initialized for data pipeline")
        except Exception as mlflow_error:
            logger.warning(f"MLflow initialization failed: {mlflow_error}. Continuing without MLflow tracking.")
        
        # Load configurations
        data_paths = get_data_paths()
        columns = get_columns()
        outlier_config = get_outlier_config()
        binning_config = get_binning_config()
        encoding_config = get_encoding_config()
        scaling_config = get_scaling_config()
        splitting_config = get_splitting_config()
        logger.info("Loaded all configurations successfully")

        # Define artifact paths
        artifacts_dir = os.path.join(os.path.dirname(__file__), '..', data_paths['data_artifacts_dir'])
        x_train_path = os.path.join(artifacts_dir, 'X_train.csv')
        x_test_path = os.path.join(artifacts_dir, 'X_test.csv')
        y_train_path = os.path.join(artifacts_dir, 'Y_train.csv')
        y_test_path = os.path.join(artifacts_dir, 'Y_test.csv')

        # Check if processed data already exists and force_rebuild is False
        if not force_rebuild and \
           os.path.exists(x_train_path) and \
           os.path.exists(x_test_path) and \
           os.path.exists(y_train_path) and \
           os.path.exists(y_test_path):
            
            logger.info("Found existing processed data. Loading from artifacts...")
            X_train = pd.read_csv(x_train_path)
            X_test = pd.read_csv(x_test_path)
            Y_train = pd.read_csv(y_train_path)
            Y_test = pd.read_csv(y_test_path)
            
            logger.info(f"Loaded splits - X_train: {X_train.shape}, X_test: {X_test.shape}")
            logger.info(f"Y_train: {Y_train.shape}, Y_test: {Y_test.shape}")
            
            return {
                'X_train': X_train,
                'X_test': X_test,
                'Y_train': Y_train,
                'Y_test': Y_test
            }

        # Create artifacts directory
        os.makedirs(data_paths['data_artifacts_dir'], exist_ok=True)
        logger.info(f"Ensured artifacts directory exists: {data_paths['data_artifacts_dir']}")

        # Step 1: Data Ingestion
        logger.info("\n" + "="*80)
        logger.info("Step 1: Data Ingestion")
        logger.info("="*80)
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        ingestor = DataIngestorCSV()
        df = ingestor.ingest(data_path)
        logger.info(f"Loaded data shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")

        # Step 2: Handle Missing Values
        logger.info("\n" + "="*80)
        logger.info("Step 2: Handle Missing Values")
        logger.info("="*80)
        
        # Check initial missing values
        initial_missing = df.isnull().sum().sum()
        logger.info(f"Initial missing values count: {initial_missing}")
        
        if columns['critical_columns']:
            drop_handler = DropMissingValuesStrategy(critical_columns=columns['critical_columns'])
            df = drop_handler.handle(df)
        else:
            logger.info("No critical columns specified for dropping")

        total_charges_handler = FillMissingValuesStrategy(
            method='median',
            relevant_column='TotalCharges'
        )
        df = total_charges_handler.handle(df)
        
        final_missing = df.isnull().sum().sum()
        logger.info(f"Final missing values count: {final_missing}")
        logger.info(f"Data shape after handling missing values: {df.shape}")

        # Step 3: Handle Outliers
        logger.info("\n" + "="*80)
        logger.info("Step 3: Handle Outliers")
        logger.info("="*80)
        
        if columns['outlier_columns']:
            outlier_detector = OutlierDetector(strategy=IQROutlierDetection())
            df = outlier_detector.handle_outliers(df, columns['outlier_columns'])
            logger.info(f"Data shape after outlier removal: {df.shape}")
        else:
            logger.info("No outlier columns specified for detection")

        # Step 4: Feature Binning (skipped for now)
        logger.info("\n" + "="*80)
        logger.info("Step 4: Feature Binning")
        logger.info("="*80)
        logger.info("Skipping feature binning")

        # Step 5: Feature Encoding
        logger.info("\n" + "="*80)
        logger.info("Step 5: Feature Encoding")
        logger.info("="*80)
        
        nominal_strategy = NominalEncodingStrategy(encoding_config['nominal_columns'])
        ordinal_strategy = OrdinalEncodingStrategy(encoding_config['ordinal_mappings'])

        df = nominal_strategy.encode(df)
        df = ordinal_strategy.encode(df)
        
        # Encode target column (Churn)
        if 'target_encoding' in encoding_config and target_column in df.columns:
            target_mapping = encoding_config['target_encoding'].get(target_column, {})
            if target_mapping:
                logger.info(f"Encoding target column '{target_column}' with mapping: {target_mapping}")
                df[target_column] = df[target_column].map(target_mapping)
                
                # Save target encoder
                import json
                encoder_path = os.path.join('artifacts/encode', f"{target_column}_encoder.json")
                with open(encoder_path, "w") as f:
                    json.dump(target_mapping, f)
                logger.info(f"Saved target encoder to {encoder_path}")
                
                # Check for unmapped values
                if df[target_column].isna().any():
                    logger.warning(f"Warning: NaN values found in '{target_column}' after encoding")
                else:
                    logger.info(f"Successfully encoded target column '{target_column}'")
        
        logger.info(f"Data shape after feature encoding: {df.shape}")
        logger.info(f"Sample encoded data:\n{df.head()}")

        # Step 6: Post Processing (Drop customerID)
        logger.info("\n" + "="*80)
        logger.info("Step 6: Post Processing")
        logger.info("="*80)
        
        if 'customerID' in df.columns:
            df = df.drop(columns=['customerID'])
            logger.info("Dropped 'customerID' column")
        else:
            logger.info("'customerID' column not found, skipping drop")
        
        logger.info(f"Data shape after post processing: {df.shape}")

        # Step 7: Data Splitting (BEFORE scaling to prevent data leakage)
        logger.info("\n" + "="*80)
        logger.info("Step 7: Data Splitting")
        logger.info("="*80)
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        
        splitting_strategy = SimpleTrainTestSplitStrategy(
            test_size=splitting_config['test_size'],
            random_state=splitting_config.get('random_state', 42)
        )
        X_train, X_test, Y_train, Y_test = splitting_strategy.split_data(df, target_column)
        
        logger.info(f"X_train shape: {X_train.shape}")
        logger.info(f"X_test shape: {X_test.shape}")
        logger.info(f"Y_train shape: {Y_train.shape}")
        logger.info(f"Y_test shape: {Y_test.shape}")

        # Step 8: Feature Scaling (AFTER splitting to prevent data leakage)
        logger.info("\n" + "="*80)
        logger.info("Step 8: Feature Scaling")
        logger.info("="*80)
        
        minmax_strategy = MinMaxScalingStrategy()
        
        # Fit scaler on training data only
        logger.info("Fitting scaler on training data...")
        X_train = minmax_strategy.scale(X_train, scaling_config['columns_to_scale'])
        
        # Transform test data using the same scaler (no fitting)
        logger.info("Transforming test data with fitted scaler...")
        scaler = minmax_strategy.get_scaler()
        X_test[scaling_config['columns_to_scale']] = scaler.transform(X_test[scaling_config['columns_to_scale']])
        logger.info("Successfully scaled test data")
                # Save scaler for inference
        scaler_path = data_paths.get('scaler', 'artifacts/models/scaler.pkl')
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        import joblib
        joblib.dump(scaler, scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")
        logger.info(f"Sample scaled training data:\n{X_train.head()}")

        # Save processed data
        logger.info("\n" + "="*80)
        logger.info("Saving Processed Data")
        logger.info("="*80)
        
        X_train.to_csv(x_train_path, index=False)
        X_test.to_csv(x_test_path, index=False)
        Y_train.to_csv(y_train_path, index=False)
        Y_test.to_csv(y_test_path, index=False)
        
        logger.info(f"Saved X_train to {x_train_path}")
        logger.info(f"Saved X_test to {x_test_path}")
        logger.info(f"Saved Y_train to {y_train_path}")
        logger.info(f"Saved Y_test to {y_test_path}")

        # Log metrics to MLflow
        if mlflow_tracker:
            try:
                dataset_info = {
                    'total_rows': len(df) + (initial_missing if 'initial_missing' in locals() else 0),
                    'train_rows': len(X_train),
                    'test_rows': len(X_test),
                    'num_features': X_train.shape[1],
                    'missing_values': initial_missing if 'initial_missing' in locals() else 0,
                    'outliers_removed': 0,  # Update if tracked
                    'test_size': splitting_config['test_size'],
                    'random_state': splitting_config.get('random_state', 42),
                    'missing_strategy': 'fill_median',
                    'outlier_method': 'IQR',
                    'encoding_applied': True,
                    'scaling_applied': True,
                    'feature_names': list(X_train.columns)
                }
                mlflow_tracker.log_data_pipeline_metrics(dataset_info)
                logger.info("Logged data pipeline metrics to MLflow")
            except Exception as e:
                logger.warning(f"Failed to log MLflow metrics: {e}")

        logger.info("\n" + "="*80)
        logger.info("Data Pipeline Completed Successfully!")
        logger.info("="*80)
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'Y_train': Y_train,
            'Y_test': Y_test
        }
    
    except FileNotFoundError as e:
        logger.error(f"File not found error: {str(e)}")
        raise
    except KeyError as e:
        logger.error(f"Column error in pipeline: {str(e)}")
        raise
    except ValueError as e:
        logger.error(f"Value error in pipeline: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in data pipeline: {str(e)}")
        raise
    finally:
        # End MLflow run
        if mlflow_tracker:
            try:
                mlflow_tracker.end_run()
            except Exception as e:
                logger.warning(f"Failed to end MLflow run: {e}")

# data_pipeline()