from src.utils.logger import logger
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainingPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logger.info("====== TRAINING PIPELINE STARTED ======")
            
            ingestion = DataIngestion()
            raw_merged_path = ingestion.initiate_data_ingestion()
            
            transformation = DataTransformation()
            processed_data_path = transformation.transform_training_data(raw_data_path=raw_merged_path)
            
            
            trainer = ModelTrainer()
            model_path = trainer.initiate_model_training(processed_data_path=processed_data_path)
            
            logger.info("====== TRAINING PIPELINE SUCCESSFULLY COMPLETED ======")
            
        except Exception as e:
            logger.error(f"The training pipeline failed: {e}")
            raise e


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()