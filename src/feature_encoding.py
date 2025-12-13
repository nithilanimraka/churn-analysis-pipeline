import logging
import pandas as pd
import os
import json
from enum import Enum
from typing import Dict, List
from abc import ABC, abstractmethod
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')


class FeatureEncodingStrategy(ABC):
    @abstractmethod
    def encode(self, df: pd.DataFrame) ->pd.DataFrame:
        pass


class VariableType(str, Enum):
    NOMINAL = 'nominal'
    ORDINAL = 'ordinal'

class NominalEncodingStrategy(FeatureEncodingStrategy):
    def __init__(self, nominal_columns):
        self.nominal_columns = nominal_columns
        self.encoder_dicts = {}  # To save the encoded stuff in dictionary type
        try:
            os.makedirs('artifacts/encode', exist_ok=True) # To store the encoded stuff in a file in json format
            logging.info(f"Initialized NominalEncodingStrategy for columns: {nominal_columns}")
        except Exception as e:
            logging.error(f"Error creating artifacts/encode directory: {str(e)}")
            raise

    def encode(self, df):
        try:
            for column in self.nominal_columns:
                if column not in df.columns:
                    logging.error(f"Column '{column}' not found in DataFrame")
                    raise KeyError(f"Column '{column}' not found in DataFrame")
                
                unique_values = df[column].unique()
                encoder_dict = {value: i for i, value in enumerate(unique_values)}
                self.encoder_dicts[column] = encoder_dict
                
                logging.info(f"Encoding nominal column '{column}' with {len(unique_values)} unique values: {list(unique_values)}")

                encoder_path = os.path.join('artifacts/encode', f"{column}_encoder.json")
                with open(encoder_path, "w") as f:
                    json.dump(encoder_dict, f)
                logging.info(f"Saved encoder for '{column}' to {encoder_path}")

                df[column] = df[column].map(encoder_dict)
                
                # Check for NaN values after mapping (indicates unseen values)
                if df[column].isna().any():
                    logging.warning(f"Warning: NaN values found in '{column}' after encoding. This may indicate unseen values.")
                else:
                    logging.info(f"Successfully encoded column '{column}'")
                    
            return df
        except KeyError as e:
            logging.error(f"Column error during nominal encoding: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Error during nominal encoding: {str(e)}")
            raise
    
    def get_encoder_dicts(self):
        return self.encoder_dicts
    
class OrdinalEncodingStrategy(FeatureEncodingStrategy):
    def __init__(self, ordinal_mappings):
        self.ordinal_mappings = ordinal_mappings
        logging.info(f"Initialized OrdinalEncodingStrategy for columns: {list(ordinal_mappings.keys())}")

    def encode(self, df):
        try:
            for column, mapping in self.ordinal_mappings.items():
                if column not in df.columns:
                    logging.error(f"Column '{column}' not found in DataFrame")
                    raise KeyError(f"Column '{column}' not found in DataFrame")
                
                logging.info(f"Encoding ordinal column '{column}' with mapping: {mapping}")
                df[column] = df[column].map(mapping)
                
                # Check for NaN values after mapping (indicates unmapped values)
                if df[column].isna().any():
                    logging.warning(f"Warning: NaN values found in '{column}' after encoding. Check if all values are in the mapping.")
                else:
                    logging.info(f"Successfully encoded ordinal variable '{column}' with {len(mapping)} categories")
                    
            return df
        except KeyError as e:
            logging.error(f"Column error during ordinal encoding: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Error during ordinal encoding: {str(e)}")
            raise