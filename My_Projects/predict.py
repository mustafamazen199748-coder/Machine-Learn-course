import os
import pickle
from contextlib import asynccontextmanager
from typing import Literal

import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = "DiaScreen AI.bin"
THRESHOLD = 0.2
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f_in:
        dv, model = pickle.load(f_in)
        ml_models["dv"] = dv
        ml_models["model"] = model
    print("✅ DiaScreen AI (XGBoost + DictVectorizer) loaded successfully.")
    yield
    ml_models.clear()


app = FastAPI(
    title="DiaScreen AI API",
    description="Early Clinical Diabetes Screening Service powered by XGBoost",
    version="1.0.0",
    lifespan=lifespan,
)


class PatientInput(BaseModel):
    # Categorical & Binary Features
    gender: Literal["male", "female", "other"] = Field(..., example="male")
    smoking_status: Literal["never", "former", "current"] = Field(..., example="former")
    cardiovascular_history: Literal[0, 1] = Field(..., example=0)
    hypertension_history: Literal[0, 1] = Field(..., example=1)
    family_history_diabetes: Literal[0, 1] = Field(..., example=1)

    # Numerical Clinical & Lifestyle Features
    age: int = Field(..., ge=1, le=120, example=54)
    alcohol_consumption_per_week: float = Field(..., ge=0, example=1.0)
    physical_activity_minutes_per_week: float = Field(..., ge=0, example=70.0)
    diet_score: float = Field(..., ge=0, le=10, example=6.5)
    sleep_hours_per_day: float = Field(..., ge=0, le=24, example=6.5)
    screen_time_hours_per_day: float = Field(..., ge=0, le=24, example=6.0)
    bmi: float = Field(..., gt=10, lt=80, example=29.4)
    waist_to_hip_ratio: float = Field(..., gt=0.3, lt=2.0, example=0.92)
    systolic_bp: float = Field(..., example=135.0)
    diastolic_bp: float = Field(..., example=85.0)
    heart_rate: float = Field(..., example=74.0)
    cholesterol_total: float = Field(..., example=215.0)
    hdl_cholesterol: float = Field(..., example=45.0)
    ldl_cholesterol: float = Field(..., example=135.0)
    triglycerides: float = Field(..., example=165.0)
    glucose_fasting: float = Field(..., example=132.0)
    glucose_postprandial: float = Field(..., example=198.0)
    insulin_level: float = Field(..., example=12.5)
    hba1c: float = Field(..., example=7.4)
    diabetes_risk_score: float = Field(..., example=41.2)


class PredictionOutput(BaseModel):
    probability: float
    threshold: float
    prediction: int
    risk_level: str
    recommendation: str


@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "XGBoost", "service": "DiaScreen AI"}


@app.post("/predict", response_model=PredictionOutput)
def predict(patient: PatientInput):
    dv = ml_models.get("dv")
    model = ml_models.get("model")

    if dv is None or model is None:
        raise HTTPException(status_code=503, detail="Model pipeline is not initialized.")

    patient_dict = patient.model_dump()
    patient_dict["gender"] = patient_dict["gender"].lower().replace(" ", "_")
    patient_dict["smoking_status"] = patient_dict["smoking_status"].lower().replace(" ", "_")

    X = dv.transform([patient_dict])
    dmatrix = xgb.DMatrix(X)
    prob = float(model.predict(dmatrix)[0])
    is_positive = int(prob > THRESHOLD)

    return PredictionOutput(
        probability=round(prob, 4),
        threshold=THRESHOLD,
        prediction=is_positive,
        risk_level="HIGH RISK" if is_positive else "LOW RISK",
        recommendation=(
            "Urgent referral for clinical and laboratory examinations."
            if is_positive
            else "Current indicators are within the safe range."
        ),
    )
