from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import pickle
import pandas as pd
from starlette import status
import numpy as np

app = FastAPI()

# Model setting
model = "model_best.pkl"
target_encoder = "target_encoder.pkl"
preprocessor = "preprocessor.pkl"

# Class input
class Input(BaseModel):
    Gender: str
    Age: float             
    Height: float
    Weight: float
    family_history_with_overweight: str
    FAVC: str
    FCVC: float
    NCP: float
    CAEC: str
    SMOKE: str
    CH2O: float
    SCC: str
    FAF: float
    TUE: float
    CALC: str
    MTRANS: str

    class Config:
        json_schema_extra = {
            "example": {
                "Gender": "Male",
                "Age": 23,
                "Height": 1.75,
                "Weight": 70.0,
                "family_history_with_overweight": "yes",
                "FAVC": "no",
                "FCVC": 2.0,
                "NCP": 3.0,
                "CAEC": "Sometimes",
                "SMOKE": "no",
                "CH2O": 3.0,
                "SCC": "yes",
                "FAF": 1.0,
                "TUE": 2.0,
                "CALC": "no",
                "MTRANS": "Public_Transportation"
            }
        }

# Class prediksi   
class Prediction(BaseModel):
    prediction: str
    probability: float
    
# Load model pickle
with open(model, 'rb') as f:
    model = pickle.load(f)
  
with open(target_encoder, 'rb') as f:
    target_encoder = pickle.load(f)
    
with open(preprocessor, 'rb') as f:
    preprocessor = pickle.load(f)
      

@app.post("/predict", status_code=status.HTTP_200_OK)
async def predict(input: Input):
    try:
        data = pd.DataFrame([input.dict()])
        encoded = preprocessor.transform(data)
        
        # Prediction
        prediction_num = model.predict(encoded)
        prediction = target_encoder.inverse_transform(prediction_num)[0] # Convert to string prediction
        proba = model.predict_proba(encoded)
        max_proba = np.max(proba, axis=1)[0] # Hitung probabilitasnya

        return Prediction(prediction=prediction, probability=max_proba)
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )