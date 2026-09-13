"""
api.py
======
A simple FastAPI backend exposing a POST /predict endpoint.

This is the piece that would eventually sit between your real backend
(receiving data from the ESP32 gateway over the internet) and the
trained ML model. For now, you can test it manually as described in
the README, using curl, Postman, or the interactive docs FastAPI
generates automatically at http://127.0.0.1:8000/docs

WHAT /predict DOES
-------------------
1. Receives sensor JSON in the request body.
2. Validates it (FastAPI + Pydantic reject malformed/missing fields
   automatically, with a clear error message).
3. Builds the same engineered features used during training (via the
   shared RealtimeAnalyzer, which also keeps short per-node history so
   delta_/rollmean_ features are meaningful across repeated calls).
4. Runs the trained model to get a prediction and per-class probabilities.
5. Returns a clean JSON response.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from realtime_analyzer import RealtimeAnalyzer

app = FastAPI(
    title="Mine Subsidence Monitoring - ML Prediction API",
    description=(
        "Prototype API for classifying underground coal mine sensor "
        "readings as NORMAL, WARNING, or CRITICAL. Student project (SIH "
        "2026) - NOT validated for real mine safety decisions."
    ),
    version="0.1.0",
)

# The analyzer is created ONCE when the server starts (loads the model
# once, and keeps per-node history in memory across requests - this is
# what lets delta_/rollmean_ features work properly across a stream of
# real /predict calls for the same node).
analyzer = RealtimeAnalyzer()


class SensorReading(BaseModel):
    """
    Defines exactly what a valid incoming request must look like.
    FastAPI + Pydantic automatically validate incoming JSON against
    this and return a helpful error if something is missing or the
    wrong type - you get this for free, no manual checking needed.
    """
    node_id: str = Field(..., example="NODE_01")
    timestamp: Optional[str] = Field(None, example="2026-09-08T10:30:00")
    distance: float = Field(..., description="Distance sensor reading, e.g. cm")
    tilt_x: float = Field(..., description="Tilt on X axis, degrees")
    tilt_y: float = Field(..., description="Tilt on Y axis, degrees")
    vibration: float = Field(..., description="Vibration magnitude")
    temperature: float = Field(..., description="Temperature, Celsius")
    humidity: float = Field(..., ge=0, le=100, description="Relative humidity, %")


class PredictionResponse(BaseModel):
    node_id: str
    timestamp: Optional[str]
    prediction: str
    probabilities: dict
    highest_probability: float


@app.get("/")
def root():
    return {
        "message": "Mine Subsidence Monitoring ML API is running.",
        "docs": "/docs",
        "predict_endpoint": "POST /predict",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading):
    try:
        result = analyzer.process_reading(reading.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {
        "node_id": result["node_id"],
        "timestamp": result["timestamp"],
        "prediction": result["prediction"].upper(),
        "probabilities": result["probabilities"],
        "highest_probability": result["highest_probability"],
    }


# To run this server:
#   uvicorn api:app --reload
# (see README.md for full instructions)
