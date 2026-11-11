import os
import pickle
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ml_models = {}
MODEL_PATH = "DiaScreen AI.bin"
THRESHOLD = 0.2

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at: {MODEL_PATH}")
    
    with open(MODEL_PATH, "rb") as f_in:
        dv, model = pickle.load(f_in)
        ml_models["dv"] = dv
        ml_models["model"] = model
    print("✅ Model & Vectorizer loaded successfully into memory.")
    yield
    ml_models.clear()

app = FastAPI(
    title="DiaScreen AI API",
    description="Early Clinical Diabetes Screening REST Service",
    version="1.0.0",
    lifespan=lifespan,
)

class PatientInput(BaseModel):
    age: int = Field(..., ge=1, le=120, example=52)
    gender: Literal["male", "female"] = Field(..., example="male")
    smoking_status: Literal["never", "former", "current"] = Field(..., example="former")
    alcohol_consumption_per_week: float = Field(..., ge=0, example=2.0)
    physical_activity_minutes_per_week: float = Field(..., ge=0, example=60.0)
    diet_score: float = Field(..., ge=0, le=10, example=5.0)
    sleep_hours_per_day: float = Field(..., ge=0, le=24, example=6.0)
    screen_time_hours_per_day: float = Field(..., ge=0, le=24, example=4.0)
    family_history_diabetes: Literal[0, 1] = Field(..., example=1)
    hypertension_history: Literal[0, 1] = Field(..., example=1)
    cardiovascular_history: Literal[0, 1] = Field(..., example=0)
    bmi: float = Field(..., gt=10, lt=80, example=29.4)
    waist_to_hip_ratio: float = Field(..., gt=0.3, lt=2.0, example=0.96)
    diabetes_risk_score: float = Field(..., example=6.0)
    systolic_bp: float = Field(..., example=135.0)
    diastolic_bp: float = Field(..., example=85.0)
    heart_rate: float = Field(..., example=78.0)
    glucose_fasting: float = Field(..., example=118.0)
    glucose_postprandial: float = Field(..., example=155.0)
    hba1c: float = Field(..., example=6.1)
    insulin_level: float = Field(..., example=14.5)
    cholesterol_total: float = Field(..., example=210.0)
    hdl_cholesterol: float = Field(..., example=42.0)
    ldl_cholesterol: float = Field(..., example=135.0)
    triglycerides: float = Field(..., example=175.0)

class PredictionOutput(BaseModel):
    probability: float
    threshold: float
    prediction: int
    risk_level: str
    recommendation: str

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "DiaScreen AI"}

@app.post("/predict", response_model=PredictionOutput)
def predict_diabetes(patient: PatientInput):
    dv = ml_models.get("dv")
    model = ml_models.get("model")

    if dv is None or model is None:
        raise HTTPException(status_code=503, detail="Model is not initialized.")

    patient_dict = patient.model_dump()
    X = dv.transform([patient_dict])

    if isinstance(model, xgb.Booster):
        features = dv.get_feature_names_out().tolist()
        dmatrix = xgb.DMatrix(X, feature_names=features)
        prob = float(model.predict(dmatrix)[0])
    else:
        prob = float(model.predict_proba(X)[0, 1])

    is_positive = int(prob >= THRESHOLD)

    return PredictionOutput(
        probability=round(prob, 4),
        threshold=THRESHOLD,
        prediction=is_positive,
        risk_level="HIGH RISK" if is_positive else "LOW RISK",
        recommendation=(
            "Urgent referral for clinical examination"
            if is_positive
            else "Current indicators are safe."
        ),
    )