import os
import pandas as pd
from src.utils.logger import logger

class DataIngestion:
    def __init__(self):
        self.raw_data_path = os.path.join("dataset", "raw")
        self.train_path = os.path.join(self.raw_data_path, "train.csv")
        self.store_path = os.path.join(self.raw_data_path, "store.csv")
        self.merged_raw_path = os.path.join(self.raw_data_path, "merged_raw.csv")

    def initiate_data_ingestion(self) -> str:
        """It reads the raw data, combines it, and saves it as a single file."""
        try:
            logger.info("The data ingestion process has begun.")
            
            df_train = pd.read_csv(self.train_path, low_memory=False)
            df_store = pd.read_csv(self.store_path)
            logger.info(f"Raw data read successfully. Train size: {df_train.shape}, Store size: {df_store.shape}")

     
            df_merged = pd.merge(df_train, df_store, on='Store', how='left')
            logger.info(f"Raw data combined successfully. New size: {df_merged.shape}")

            df_merged.to_csv(self.merged_raw_path, index=False)
            logger.info(f"Combined raw data saved: {self.merged_raw_path}")

            return self.merged_raw_path

        except Exception as e:
            logger.error(f"An error occurred during the data ingestion process: {e}")
            raise e