from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.pipelines.inference_pipeline import InferencePipeline

app = FastAPI(title="Smart Rossmann Prediction API", version="2.0")

class UserInput(BaseModel):
    Store: int
    Date: str 
    Promo: int
    StateHoliday: str
    SchoolHoliday: int

@app.get("/")
def read_root():
    return {"message": "Smart Rossmann Prediction API is active. Use /predict for forecasting."}

@app.post("/predict")
def predict_sales(user_data: UserInput):
    try:
        
        pipeline = InferencePipeline()
        
        
        prediction = pipeline.predict(
            store=user_data.Store,
            date_str=user_data.Date,
            promo=user_data.Promo,
            state_holiday=user_data.StateHoliday,
            school_holiday=user_data.SchoolHoliday
        )
        
        return {
            "Store_ID": user_data.Store,
            "Date": user_data.Date,
            "Is_Promo": bool(user_data.Promo),
            "Predicted_Sales_Euro": prediction
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error occurred during prediction: {e}")