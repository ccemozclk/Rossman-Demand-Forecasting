import numpy as np
import datetime
from src.utils.logger import logger
import pandas as pd
import os
import re

class DataTransformation:
    def __init__(self):
        self.expected_features = [
            'Store', 'DayOfWeek', 'Open', 'Promo', 'SchoolHoliday', 'Assortment',
            'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear',
            'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'Year', 'Month', 'Day',
            'WeekOfYear', 'Sales_Lag_1', 'Sales_Lag_7', 'Sales_Lag_42',
            'Sales_Roll_Mean_7', 'Sales_Roll_Mean_30', 'StoreType_b', 'StoreType_c',
            'StoreType_d', 'StateHoliday_a', 'StateHoliday_b', 'StateHoliday_c',
            'PromoInterval_Jan_Apr_Jul_Oct', 'PromoInterval_Mar_Jun_Sept_Dec', 'PromoInterval_None'
        ]

    def transform_training_data(self, raw_data_path: str):
        """
        It takes the raw training data, cleans it, performs feature engineering, and saves it to the processed folder.
        """
        import re
        try:
            logger.info("The transformation process for educational data has begun.")
            
            
            df = pd.read_csv(raw_data_path, low_memory=False)
            
            df['Date'] = pd.to_datetime(df['Date'])
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['Day'] = df['Date'].dt.day
            df['WeekOfYear'] = df['Date'].dt.isocalendar().week

            df['Promo2SinceWeek'] = df['Promo2SinceWeek'].fillna(0)
            df['Promo2SinceYear'] = df['Promo2SinceYear'].fillna(0)
            df['PromoInterval'] = df['PromoInterval'].fillna('None')
            df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].fillna(0)
            df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].fillna(0)
            max_distance = df['CompetitionDistance'].max()
            df['CompetitionDistance'] = df['CompetitionDistance'].fillna(max_distance)

            df = df[(df['Open'] == 1) & (df['Sales'] > 0)]
            df.sort_values(['Store', 'Date'], ascending=[True, True], inplace=True)
            
            df['Sales_Lag_1'] = df.groupby('Store')['Sales'].shift(1)
            df['Sales_Lag_7'] = df.groupby('Store')['Sales'].shift(7)
            df['Sales_Lag_42'] = df.groupby('Store')['Sales'].shift(42)
            df['Sales_Roll_Mean_7'] = df.groupby('Store')['Sales'].transform(
                    lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()
                )
            df['Sales_Roll_Mean_30'] = df.groupby('Store')['Sales'].transform(
                    lambda x: x.shift(1).rolling(window=30, min_periods=1).mean()
                )
            df.dropna(inplace=True)

            df.drop(columns=['Date'], inplace=True, errors='ignore')

            if df['Assortment'].dtype == 'object':
                df['Assortment'] = df['Assortment'].map({'a': 1, 'b': 2, 'c': 3})

            df = pd.get_dummies(df, columns=['StoreType', 'StateHoliday', 'PromoInterval'])
            df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '_', x))

            for col in self.expected_features:
                if col not in df.columns:
                    df[col] = 0.0

            final_columns = self.expected_features + ['Sales']
            df = df[final_columns]

            for col in df.columns:
                if df[col].dtype == 'bool':
                    df[col] = df[col].astype(int)
            
            processed_dir = os.path.join("dataset", "processed")
            os.makedirs(processed_dir, exist_ok=True)
            processed_data_path = os.path.join(processed_dir, "processed_train.csv")
            df.to_csv(processed_data_path, index=False)
            
            logger.info(f"Processed training data has been saved: {processed_data_path}")
            return processed_data_path

        except Exception as e:
            logger.error(f"Error occurred while transforming training data: {e}")
            raise e

    def _get_store_metadata(self, store_id: int):
        """
        Mock database query: Retrieves static store information.
        """
        logger.info(f"Retrieving static data from the database: Store {store_id}")
        return {
            'Assortment': 1, 
            'CompetitionDistance': 1270.0,
            'CompetitionOpenSinceMonth': 9.0,
            'CompetitionOpenSinceYear': 2008.0,
            'Promo2': 0,
            'Promo2SinceWeek': 0.0,
            'Promo2SinceYear': 0.0,
            'StoreType': 'c',
            'PromoInterval': 'None'
        }

    def _get_historical_sales(self, store_id: int, target_date: str):
        """
        Mock database query: Retrieves historical sales and moving averages.
        """
        logger.info(f"Retrieving historical sales and moving averages from the database: Store {store_id}, Target Date {target_date}")
        return {
            'Sales_Lag_1': 4500.0,
            'Sales_Lag_7': 5200.0,
            'Sales_Lag_42': 4800.0,
            'Sales_Roll_Mean_7': 4950.0,
            'Sales_Roll_Mean_30': 5100.0
        }

    def transform_inference_data(self, store: int, date_str: str, promo: int, state_holiday: str, school_holiday: int) -> np.ndarray:
        """Retrieves 5 basic inputs from the user and transforms them into a 30-column matrix."""
        try:
            logger.info("The inference process has begun.")
            
            
            date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            
            store_meta = self._get_store_metadata(store)
            hist_sales = self._get_historical_sales(store, date_str)
            
            
            feature_dict = {col: 0.0 for col in self.expected_features}
            
            
            feature_dict['Store'] = store
            feature_dict['DayOfWeek'] = date_obj.isoweekday()
            feature_dict['Open'] = 1.0 
            feature_dict['Promo'] = promo
            feature_dict['SchoolHoliday'] = school_holiday
            
            
            feature_dict['Year'] = date_obj.year
            feature_dict['Month'] = date_obj.month
            feature_dict['Day'] = date_obj.day
            feature_dict['WeekOfYear'] = date_obj.isocalendar().week
            
            
            feature_dict['Assortment'] = store_meta['Assortment']
            feature_dict['CompetitionDistance'] = store_meta['CompetitionDistance']
            feature_dict['CompetitionOpenSinceMonth'] = store_meta['CompetitionOpenSinceMonth']
            feature_dict['CompetitionOpenSinceYear'] = store_meta['CompetitionOpenSinceYear']
            feature_dict['Promo2'] = store_meta['Promo2']
            feature_dict['Promo2SinceWeek'] = store_meta['Promo2SinceWeek']
            feature_dict['Promo2SinceYear'] = store_meta['Promo2SinceYear']
            
            
            for key, val in hist_sales.items():
                feature_dict[key] = val
                
            
            st_type = store_meta['StoreType']
            if st_type in ['b', 'c', 'd']:
                feature_dict[f'StoreType_{st_type}'] = 1.0
                
            if state_holiday in ['a', 'b', 'c']:
                feature_dict[f'StateHoliday_{state_holiday}'] = 1.0
                
            pi_type = store_meta['PromoInterval']
            if pi_type == 'Jan,Apr,Jul,Oct':
                feature_dict['PromoInterval_Jan_Apr_Jul_Oct'] = 1.0
            elif pi_type == 'Mar,Jun,Sept,Dec':
                feature_dict['PromoInterval_Mar_Jun_Sept_Dec'] = 1.0
            elif pi_type == 'None':
                feature_dict['PromoInterval_None'] = 1.0

            
            input_array = np.array([feature_dict[col] for col in self.expected_features]).reshape(1, -1)
            
            logger.info(f"The transformation was successfully completed. The resulting matrix size is: {input_array.shape}")
            return input_array
            
        except Exception as e:
            logger.error(f"An error occurred during the transformation process: {e}")
            raise e
        
    