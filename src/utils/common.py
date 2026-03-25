import os
import yaml
import joblib
from src.utils.logger import logger # Kendi logger'ımızı içeri alıyoruz

def read_config(config_path="config.yaml"):
    """
    It reads the YAML file and returns it as a dictionary.
    """
    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            logger.info(f"The configuration file was read successfully: {config_path}")
        return config
    except Exception as e:
        logger.error(f"The configuration file could not be read: {e}")
        raise e

def save_object(file_path, obj):
    """
    Python saves objects (such as ML models) as .pkl files.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(obj, file_path)
        logger.info(f"Object successfully registered: {file_path}")
    except Exception as e:
        logger.error(f"An error occurred while saving the object: {e}")
        raise e

def load_object(file_path):
    """
    Loads saved .pkl objects into memory.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        obj = joblib.load(file_path)
        logger.info(f"Object successfully loaded: {file_path}")
        return obj
    except Exception as e:
        logger.error(f"An error occurred while loading the object: {file_path} - {e}")
        raise e