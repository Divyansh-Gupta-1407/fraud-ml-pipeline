from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import mlflow.pyfunc
import mlflow
import os

os.makedirs("mlruns", exist_ok=True)
os.makedirs("mlartifacts", exist_ok=True)

MODEL_NAME = "fraud_detection_xgb"
MODEL_STAGE = "Production"
SCALER_PATH = "data/processed/scaler.pkl"

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://host.docker.internal:5000"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

print("Loading resources...")

scaler = joblib.load(SCALER_PATH)

# MODEL_PATH = (
#     "mlruns/0/models/m-195b68f592034b2cb1a51d18562c100e/artifacts"
# )

# model = mlflow.pyfunc.load_model(MODEL_PATH)

MODEL_PATH = "artifacts/model"
model = mlflow.pyfunc.load_model(MODEL_PATH)

print("✅ Model loaded successfully!")


# 3. Define Input Schema
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

app = FastAPI(title="Fraud Detection API")

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running!"}

@app.post("/predict")
def predict(input_data: Transaction):
    # Convert input to DataFrame
    data_dict = input_data.dict()
    df = pd.DataFrame([data_dict])
    
    # --- CRITICAL: Apply the same Preprocessing as Training ---
    # We must scale 'Time' and 'Amount' using the loaded scaler
    cols_to_scale = ['Time', 'Amount']
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    
    # Reorder columns to match training input exactly (just in case)
    # df = df[model.metadata.signature.inputs.input_names()] 

    # Make Prediction
    prediction = model.predict(df)[0]
    result = "FRAUD" if prediction == 1 else "LEGIT"

    return {
        "prediction": int(prediction),
        "result": result
    }