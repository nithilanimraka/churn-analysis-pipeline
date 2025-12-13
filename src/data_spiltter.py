import logging
import pandas as pd
from enum import Enum
from abc import ABC, abstractmethod
from typing import Tuple
from sklearn.model_selection import train_test_split
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')


class DataSplittingStrategy(ABC):
    @abstractmethod
    def split_data(self, df: pd.DataFrame, target_column: str) ->Tuple[pd.
        DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        pass


class SplitType(str, Enum):
    SIMPLE = 'simple' # Simple Splitter I utilized
    STRATIFIED = 'stratified'

class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        logging.info(f"Initialized SimpleTrainTestSplitStrategy with test_size={test_size}, random_state={random_state}")

    def split_data(self, df, target_column):
        try:
            # Validate target column exists
            if target_column not in df.columns:
                raise KeyError(f"Target column '{target_column}' not found in DataFrame")
            
            # Check for missing values
            missing_count = df.isnull().sum().sum()
            if missing_count > 0:
                logging.warning(f"DataFrame contains {missing_count} missing values. Consider handling them before splitting.")
                missing_by_column = df.isnull().sum()
                missing_cols = missing_by_column[missing_by_column > 0]
                for col, count in missing_cols.items():
                    logging.warning(f"  - Column '{col}': {count} missing values")
            
            logging.info(f"Starting train-test split with target column: '{target_column}'")
            logging.info(f"Original dataset shape: {df.shape}")
            
            Y = df[target_column]
            X = df.drop(columns=[target_column])
            
            # Log target distribution
            target_distribution = Y.value_counts()
            logging.info(f"Target column distribution:\n{target_distribution}")
            
            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, 
                test_size=self.test_size, 
                random_state=self.random_state
            )
            
            # Log split results
            logging.info(f"Train set: X_train shape={X_train.shape}, Y_train shape={Y_train.shape}")
            logging.info(f"Test set: X_test shape={X_test.shape}, Y_test shape={Y_test.shape}")
            logging.info(f"Train set size: {len(X_train)} ({(1-self.test_size)*100:.1f}%)")
            logging.info(f"Test set size: {len(X_test)} ({self.test_size*100:.1f}%)")
            
            # Log target distribution in splits
            logging.info(f"Y_train distribution:\n{Y_train.value_counts()}")
            logging.info(f"Y_test distribution:\n{Y_test.value_counts()}")
            
            logging.info("Data splitting completed successfully")
            return X_train, X_test, Y_train, Y_test
            
        except KeyError as e:
            logging.error(f"Column error during data splitting: {str(e)}")
            raise
        except ValueError as e:
            logging.error(f"Value error during data splitting: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error during data splitting: {str(e)}")
            raise