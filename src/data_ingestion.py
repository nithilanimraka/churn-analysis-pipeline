import os
import pandas as pd
from abc import ABC, abstractmethod
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path_or_link: str) ->pd.DataFrame:
        pass

class DataIngestorCSV(DataIngestor):
    def ingest(self, file_path_or_link):
        try:
            logger.info(f"Starting CSV ingestion from: {file_path_or_link}")
            df = pd.read_csv(file_path_or_link)
            logger.info(f"Successfully ingested CSV with shape: {df.shape}")
            return df
        except FileNotFoundError as e:
            logger.error(f"File not found: {file_path_or_link}")
            raise
        except pd.errors.EmptyDataError as e:
            logger.error(f"Empty CSV file: {file_path_or_link}")
            raise
        except Exception as e:
            logger.error(f"Error ingesting CSV from {file_path_or_link}: {str(e)}")
            raise
    
class DataIngestorExcel(DataIngestor):
    def ingest(self, file_path_or_link):
        try:
            logger.info(f"Starting Excel ingestion from: {file_path_or_link}")
            df = pd.read_excel(file_path_or_link)
            logger.info(f"Successfully ingested Excel with shape: {df.shape}")
            return df
        except FileNotFoundError as e:
            logger.error(f"File not found: {file_path_or_link}")
            raise
        except Exception as e:
            logger.error(f"Error ingesting Excel from {file_path_or_link}: {str(e)}")
            raise