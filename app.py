from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib


# Create FastAPI application
app = FastAPI(
    title="Titanic Survival Prediction API"
)


# Load the trained model
model = joblib.load("model.pkl")


# Define input data with validation
class Passenger(BaseModel):
    Pclass: int = Field(..., ge=1, le=3)
    Sex: str
    Age: float = Field(..., ge=0)
    SibSp: int = Field(..., ge=0)
    Parch: int = Field(..., ge=0)
    Fare: float = Field(..., ge=0)
    Embarked: str


# Health check endpoint
@app.get("/health")
def health():
    return {
        "status": "API is running"
    }


# Prediction endpoint
@app.post("/predict")
def predict(passenger: Passenger):

    data = pd.DataFrame([{
        "Pclass": passenger.Pclass,
        "Sex": passenger.Sex,
        "Age": passenger.Age,
        "SibSp": passenger.SibSp,
        "Parch": passenger.Parch,
        "Fare": passenger.Fare,
        "Embarked": passenger.Embarked
    }])

    prediction = model.predict(data)[0]

    if prediction == 1:
        result = "Survived"
    else:
        result = "Did Not Survive"

    return {
        "prediction": int(prediction),
        "result": result
    }