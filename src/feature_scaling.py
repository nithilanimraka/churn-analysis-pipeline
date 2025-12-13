import logging
import pandas as pd
from enum import Enum
from typing import List
from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')


class FeatureScalingStrategy(ABC):

    @abstractmethod
    def scale(self, df: pd.DataFrame, columns_to_scale: List[str]) -> pd.DataFrame:
        pass


class ScalingType(str, Enum):
    MINMAX = 'minmax'
    STANDARD = 'standard'

class MinMaxScalingStrategy(FeatureScalingStrategy):
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.fitted = False
        logging.info("Initialized MinMaxScalingStrategy")

    def scale(self, df, columns_to_scale):
        try:
            # Validate columns exist
            missing_columns = [col for col in columns_to_scale if col not in df.columns]
            if missing_columns:
                raise KeyError(f"Columns not found in DataFrame: {missing_columns}")
            
            # Validate columns are numeric
            non_numeric = [col for col in columns_to_scale if not pd.api.types.is_numeric_dtype(df[col])]
            if non_numeric:
                raise ValueError(f"Non-numeric columns cannot be scaled: {non_numeric}")
            
            logging.info(f"Starting Min-Max scaling for columns: {columns_to_scale}")
            
            # Log original ranges
            for col in columns_to_scale:
                logging.info(f"Column '{col}' - Original range: [{df[col].min():.2f}, {df[col].max():.2f}]")
            
            df[columns_to_scale] = self.scaler.fit_transform(df[columns_to_scale])
            self.fitted = True
            
            logging.info(f"Successfully applied Min-Max scaling to {len(columns_to_scale)} column(s)")
            logging.info("All scaled columns now range from 0 to 1")
            
            return df
            
        except KeyError as e:
            logging.error(f"Column error during Min-Max scaling: {str(e)}")
            raise
        except ValueError as e:
            logging.error(f"Value error during Min-Max scaling: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error during Min-Max scaling: {str(e)}")
            raise
    
    def get_scaler(self):
        if not self.fitted:
            logging.warning("Scaler has not been fitted yet")
        return self.scaler