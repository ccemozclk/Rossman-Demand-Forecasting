import os
from src.utils.logger import logger
from src.utils.common import load_object
from src.components.data_transformation import DataTransformation

class InferencePipeline:
    def __init__(self):
        
        self.model_path = os.path.join("artifacts", "rossmann_model.pkl")

    def predict(self, store: int, date_str: str, promo: int, state_holiday: str, school_holiday: int):
        try:
            logger.info(f"--- New Prediction Request Received --- [Store: {store}, Date: {date_str}]")

            
            data_transformer = DataTransformation()
            input_matrix = data_transformer.transform_inference_data(
                store=store,
                date_str=date_str,
                promo=promo,
                state_holiday=state_holiday,
                school_holiday=school_holiday
            )
            
            
            model = load_object(self.model_path)
            
            
            prediction = model.predict(input_matrix)
            predicted_sales = round(float(prediction[0]), 2)
            
            logger.info(f"Prediction completed successfully. Expected Sales: {predicted_sales} Euro")
            
            return predicted_sales

        except Exception as e:
            logger.error(f"Critical error occurred in the inference pipeline: {e}")
            raise e