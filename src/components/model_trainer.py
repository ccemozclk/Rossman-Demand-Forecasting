import os
import json
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from src.utils.logger import logger
from src.utils.common import save_object



class ModelTrainer:

    def __init__(self):
        self.model_save_path = os.path.join("artifacts", "rossmann_model.pkl")
        self.metrics_save_path = os.path.join("artifacts", "metrics.json")
        
        self.best_params = {
            'n_estimators': 796,
            'learning_rate': 0.078227,
            'num_leaves': 97,
            'max_depth': 14,
            'min_child_samples': 21,
            'subsample': 0.728,
            'colsample_bytree': 0.901,
            'random_state': 42,
            'verbose': -1
        }

    def rmspe(self, y_true, y_pred):
        mask = y_true != 0
        y_true_masked = y_true[mask]
        y_pred_masked = y_pred[mask]
        error = np.sqrt(np.mean(((y_true_masked - y_pred_masked) / y_true_masked) ** 2))
        return error
    

    def initiate_model_training(self, processed_data_path: str):
        try:
            logger.info("Model training process has started.")
            
            
            df = pd.read_csv(processed_data_path)
            
            
            if 'Customers' in df.columns:
                df = df.drop(columns=['Customers'])
                
            y = df['Sales']
            X = df.drop(columns=['Sales'])

            val_condition = (X['Year'] == 2015) & (X['Month'] >= 6)

            X_train = X[~val_condition]
            y_train = y[~val_condition]
            X_val = X[val_condition]
            y_val = y[val_condition]

            logger.info("The model is being trained for validation...")
            val_model = LGBMRegressor(**self.best_params)
            val_model.fit(X_train, y_train)

            preds = val_model.predict(X_val)
            model_rmspe = self.rmspe(y_val.values, preds)
            logger.info(f"Calculated Dynamic RMSPE Score: %{model_rmspe*100:.2f}")


            logger.info("For the live production environment, the model is trained with ALL data...")
            final_model = LGBMRegressor(**self.best_params)
            final_model.fit(X, y)
            save_object(self.model_save_path, final_model)
            
            os.makedirs(os.path.dirname(self.metrics_save_path), exist_ok=True)
            with open(self.metrics_save_path, "w") as f:
                json.dump({"rmspe": float(model_rmspe)}, f)
                
            logger.info("The model and metrics (metrics.json) have been successfully saved.")
            return self.model_save_path

        except Exception as e:
            logger.error(f"An error occurred during model training: {e}")
            raise e