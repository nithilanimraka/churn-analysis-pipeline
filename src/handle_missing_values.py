import logging
import pandas as pd
from enum import Enum
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel
from abc import ABC, abstractmethod
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()


class MissingValueHandlingStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) ->pd.DataFrame:
        pass

class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, critical_columns=[]):
        self.critical_columns = critical_columns 
        logging.info(f"Dropping rows with missing values in critical columns: {self.critical_columns}")

    def handle(self, df):
        try:
            df_cleaned = df.dropna(subset=self.critical_columns)
            n_dropped = len(df) - len(df_cleaned)
            logging.info(f"{n_dropped} row(s) have been dropped due to missing values in {self.critical_columns}")
            return df_cleaned
        except KeyError as e:
            logging.error(f"Column not found in DataFrame: {e}")
            raise
        except Exception as e:
            logging.error(f"Error dropping missing values: {str(e)}")
            raise


class FillMissingValuesStrategy(MissingValueHandlingStrategy):

    def __init__(
                self, 
                method='median', 
                fill_value=None, 
                relevant_column=None, 
                is_custom_imputer=False,
                custom_imputer=None
                ):
        self.method = method
        self.fill_value = fill_value
        self.relevant_column = relevant_column
        self.is_custom_imputer = is_custom_imputer
        self.custom_imputer = custom_imputer

    def handle(self, df):
        try:
            if self.is_custom_imputer:
                logging.info(f"Using custom imputer for column {self.relevant_column}")
                return self.custom_imputer.impute(df)
            
            if self.relevant_column not in df.columns:
                raise KeyError(f"Column '{self.relevant_column}' not found in DataFrame")
            
            initial_missing = df[self.relevant_column].isna().sum()
            logging.info(f"Handling {initial_missing} missing values in column '{self.relevant_column}' using method '{self.method}'")
            
            # Convert column to numeric, coercing errors to NaN
            df[self.relevant_column] = pd.to_numeric(df[self.relevant_column], errors='coerce')
            
            # Fill missing values based on method
            if self.fill_value is not None:
                df[self.relevant_column] = df[self.relevant_column].fillna(self.fill_value)
                logging.info(f"Filled missing values in '{self.relevant_column}' with custom value: {self.fill_value}")
            elif self.method == 'mean':
                fill_val = df[self.relevant_column].mean()
                df[self.relevant_column] = df[self.relevant_column].fillna(fill_val)
                logging.info(f"Filled missing values in '{self.relevant_column}' with mean: {fill_val:.2f}")
            elif self.method == 'median':
                fill_val = df[self.relevant_column].median()
                df[self.relevant_column] = df[self.relevant_column].fillna(fill_val)
                logging.info(f"Filled missing values in '{self.relevant_column}' with median: {fill_val:.2f}")
            elif self.method == 'mode':
                fill_val = df[self.relevant_column].mode()[0]
                df[self.relevant_column] = df[self.relevant_column].fillna(fill_val)
                logging.info(f"Filled missing values in '{self.relevant_column}' with mode: {fill_val:.2f}")
            else:
                raise ValueError(f"Invalid method: '{self.method}'. Choose from 'mean', 'median', or 'mode'")
            
            final_missing = df[self.relevant_column].isna().sum()
            logging.info(f"Successfully filled missing values. Remaining missing values: {final_missing}")
            return df
            
        except KeyError as e:
            logging.error(f"Column error: {str(e)}")
            raise
        except ValueError as e:
            logging.error(f"Value error: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Error filling missing values in column '{self.relevant_column}': {str(e)}")
            raise