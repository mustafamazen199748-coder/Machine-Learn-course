import pickle
from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict
import uvicorn

class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gender: Literal["male", "female"]
    seniorcitizen: Literal[0, 1]
    partner: Literal["yes", "no"]
    dependents: Literal["yes", "no"]
    phoneservice: Literal["yes", "no"]
    multiplelines: Literal["no", "yes", "no_phone_service"]
    internetservice: Literal["dsl", "fiber_optic", "no"]
    onlinesecurity: Literal["no", "yes", "no_internet_service"]
    onlinebackup: Literal["no", "yes", "no_internet_service"]
    deviceprotection: Literal["no", "yes", "no_internet_service"]
    techsupport: Literal["no", "yes", "no_internet_service"]
    streamingtv: Literal["no", "yes", "no_internet_service"]
    streamingmovies: Literal["no", "yes", "no_internet_service"]
    contract: Literal["month-to-month", "one_year", "two_year"]
    paperlessbilling: Literal["yes", "no"]
    paymentmethod: Literal[
        "electronic_check",
        "mailed_check",
        "bank_transfer_(automatic)",
        "credit_card_(automatic)",
    ]
    tenure: int = Field(..., ge=0)
    monthlycharges: float = Field(..., ge=0.0)
    totalcharges: float = Field(..., ge=0.0)


class PredictResponse(BaseModel):
    churn_probability: float
    churn: bool


app = FastAPI(title="Customer Churn Prediction API")

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


def predict_single(customer_data: dict) -> float:
    X = dv.transform([customer_data])
    prob = model.predict_proba(X)[0, 1]
    return float(prob)


@app.post("/predict", response_model=PredictResponse)
def predict(customer: Customer) -> PredictResponse:
    customer_dict = customer.model_dump()
    prob = predict_single(customer_dict)

    return PredictResponse(
        churn_probability=round(prob, 4),
        churn=bool(prob >= 0.5)
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)