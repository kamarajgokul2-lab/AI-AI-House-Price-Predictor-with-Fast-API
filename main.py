"""
Estima.AI — House Price Prediction API
========================================
A minimal FastAPI backend matching the contract expected by the
Estima.AI frontend.

Run:
    pip install fastapi uvicorn scikit-learn joblib --break-system-packages
    uvicorn main:app --reload --port 8000

Endpoint:
    POST /predict
    Request:  {"area": 1200}
    Response: {"area": 1200, "predicted_price": 2400000}
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Estima.AI Prediction API", version="1.0.0")

# Allow the frontend (any origin in dev) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    area: float = Field(..., gt=0, le=50000, description="Built-up area in square feet")


class PredictResponse(BaseModel):
    area: float
    predicted_price: float


# Simple linear coefficient — replace with a loaded model
# (e.g. joblib.load("model.pkl")) for a real deployment.
PRICE_PER_SQFT = 2000
BASE_PRICE = 50_000


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    if payload.area <= 0:
        raise HTTPException(status_code=400, detail="Area must be greater than 0")

    predicted_price = BASE_PRICE + payload.area * PRICE_PER_SQFT

    return PredictResponse(area=payload.area, predicted_price=predicted_price)


@app.get("/health")
def health():
    return {"status": "ok"}
