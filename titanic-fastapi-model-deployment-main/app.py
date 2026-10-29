
import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal

# Load the model trained in Task 1
model = joblib.load("model.pkl")

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="ML model serving using FastAPI",
    version="1.0.0"
)

# Validate incoming passenger details
class Passenger(BaseModel):
    Pclass: int = Field(ge=1, le=3)
    Sex: Literal["male", "female"]
    Age: float | None = Field(default=None, ge=0, le=100)
    SibSp: int = Field(ge=0)
    Parch: int = Field(ge=0)
    Fare: float = Field(ge=0)
    Embarked: Literal["C", "Q", "S"] | None = None

# GET endpoint: check API status
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }

# POST endpoint: predict passenger survival
@app.post("/predict")
def predict(passenger: Passenger):
    passenger_data = pd.DataFrame(
        [passenger.model_dump()]
    )

    prediction = int(model.predict(passenger_data)[0])

    return {
        "prediction": prediction,
        "label": (
            "Survived"
            if prediction == 1
            else "Did not survive"
        )
    }
