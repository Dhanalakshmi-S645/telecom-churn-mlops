from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
import joblib
import numpy as np
import pandas as pd
import os

app = FastAPI(
    title="Telecom Churn Prediction API",
    description="MLOps pipeline for predicting customer churn in telecom. Built with XGBoost, MLflow, and Evidently AI.",
    version="1.0.0"
)

# Load model and preprocessing artifacts
try:
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    encoders = joblib.load("models/encoders.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    model_name = joblib.load("models/best_model_name.pkl")
except Exception as e:
    model = scaler = encoders = feature_names = None
    model_name = "Not loaded"


class CustomerData(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: float
    TotalCharges: float


class PredictionResponse(BaseModel):
    churn_prediction: str
    churn_probability: float
    risk_level: str
    model_used: str
    recommendation: str


@app.get("/")
def root():
    return {
        "message": "Telecom Churn Prediction API",
        "model": model_name,
        "status": "running",
        "endpoints": ["/predict", "/health", "/docs"]
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": model_name
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")

    data = customer.dict()
    df = pd.DataFrame([data])

    # Feature engineering (must match preprocess.py)
    df["tenure_years"] = df["tenure"] / 12
    df["charges_per_month_ratio"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["has_streaming"] = (
        (df["StreamingTV"] == "Yes") | (df["StreamingMovies"] == "Yes")
    ).astype(int)
    df["num_services"] = (
        (df["PhoneService"] == "Yes").astype(int) +
        (df["MultipleLines"] == "Yes").astype(int) +
        (df["InternetService"] != "No").astype(int) +
        (df["OnlineSecurity"] == "Yes").astype(int) +
        (df["OnlineBackup"] == "Yes").astype(int) +
        (df["DeviceProtection"] == "Yes").astype(int) +
        (df["TechSupport"] == "Yes").astype(int) +
        (df["StreamingTV"] == "Yes").astype(int) +
        (df["StreamingMovies"] == "Yes").astype(int)
    )

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        if col in encoders:
            le = encoders[col]
            try:
                df[col] = le.transform(df[col])
            except ValueError:
                df[col] = 0

    df = df[feature_names]
    df_scaled = pd.DataFrame(scaler.transform(df), columns=feature_names)

    prob = model.predict_proba(df_scaled)[0][1]
    prediction = int(prob >= 0.5)

    if prob < 0.3:
        risk = "Low"
        recommendation = "Customer is stable. Consider upsell opportunities."
    elif prob < 0.6:
        risk = "Medium"
        recommendation = "Monitor this customer. Consider loyalty offers or discounts."
    else:
        risk = "High"
        recommendation = "Immediate retention action needed. Offer contract upgrade or special discount."

    return PredictionResponse(
        churn_prediction="Will Churn" if prediction == 1 else "Will Stay",
        churn_probability=round(float(prob), 4),
        risk_level=risk,
        model_used=model_name,
        recommendation=recommendation
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
