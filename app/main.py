"""
app/main.py
-----------
FastAPI service for credit card fraud detection.

Artifacts (autoencoder.pt, weights.npy, scaler.pkl, config.json) are loaded
once at startup and shared across all requests.

Run locally:
    uvicorn app.main:app --reload

Run via Docker:
    docker-compose up
"""

import json
import pickle
from contextlib import asynccontextmanager

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ── Model definition (must match autoencoder.pt exactly) ──────────────────────
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int = 30):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20), nn.ReLU(),
            nn.Linear(20, 10),        nn.ReLU(),
            nn.Linear(10, 5),         nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(5, 10),         nn.ReLU(),
            nn.Linear(10, 20),        nn.ReLU(),
            nn.Linear(20, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ── Global artifact store ──────────────────────────────────────────────────────
artifacts: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all ML artifacts once when the server starts."""
    with open("config.json") as f:
        cfg = json.load(f)

    model = Autoencoder(input_dim=cfg["input_dim"])
    model.load_state_dict(torch.load(cfg["model_path"], weights_only=True))
    model.eval()

    with open(cfg["scaler_path"], "rb") as f:
        scalers = pickle.load(f)  # {"amount": StandardScaler, "time": StandardScaler}

    weights = np.load(cfg["weights_path"])

    artifacts["model"]         = model
    artifacts["scalers"]       = scalers
    artifacts["weights"]       = weights
    artifacts["threshold"]     = cfg["threshold"]
    artifacts["feature_order"] = cfg["feature_order"]

    print(f"Model loaded. Threshold = {cfg['threshold']:.6f}")
    yield
    artifacts.clear()


app = FastAPI(
    title="Credit Card Fraud Detection API",
    description=(
        "Autoencoder-based anomaly detection for credit card transactions. "
        "The model was trained only on normal transactions; fraudulent ones "
        "produce a higher weighted reconstruction error. "
        "POST 30 transaction features to `/predict` and receive an anomaly score."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Pydantic schemas ───────────────────────────────────────────────────────────
class TransactionFeatures(BaseModel):
    Time: float
    V1:   float
    V2:   float
    V3:   float
    V4:   float
    V5:   float
    V6:   float
    V7:   float
    V8:   float
    V9:   float
    V10:  float
    V11:  float
    V12:  float
    V13:  float
    V14:  float
    V15:  float
    V16:  float
    V17:  float
    V18:  float
    V19:  float
    V20:  float
    V21:  float
    V22:  float
    V23:  float
    V24:  float
    V25:  float
    V26:  float
    V27:  float
    V28:  float
    Amount: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "Time": 0.0,
                "V1": -1.3598, "V2": -0.0728, "V3": 2.5363,
                "V4": 1.3782,  "V5": -0.3383, "V6": 0.4624,
                "V7": 0.2396,  "V8": 0.0987,  "V9": 0.3638,
                "V10": 0.0908, "V11": -0.5516, "V12": -0.6178,
                "V13": -0.9913, "V14": -0.3112, "V15": 1.4682,
                "V16": -0.4704, "V17": 0.2079, "V18": 0.0258,
                "V19": 0.4031, "V20": 0.2514, "V21": -0.0183,
                "V22": 0.2778, "V23": -0.1105, "V24": 0.0669,
                "V25": 0.1285, "V26": -0.1891, "V27": 0.1336,
                "V28": -0.0210, "Amount": 149.62,
            }
        }
    }


class PredictionResponse(BaseModel):
    anomaly_score: float = Field(..., description="Weighted MSE reconstruction error")
    is_fraud:      bool  = Field(..., description="True if anomaly_score >= threshold")
    threshold:     float = Field(..., description="Decision threshold used for classification")


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Monitoring"])
def health():
    """Returns service status. Used by Docker healthcheck."""
    return {"status": "ok", "model": "autoencoder"}


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(transaction: TransactionFeatures):
    """
    Predict whether a credit card transaction is fraudulent.

    - **Time** and **Amount** are raw values; they will be standardized internally.
    - **V1–V28** are the PCA-transformed features from the original dataset; pass them as-is.
    """
    try:
        scalers       = artifacts["scalers"]
        model         = artifacts["model"]
        weights       = artifacts["weights"]
        threshold     = artifacts["threshold"]
        feature_order = artifacts["feature_order"]

        raw = transaction.model_dump()

        # Scale Time and Amount with their respective saved scalers
        raw["Amount"] = float(scalers["amount"].transform([[raw["Amount"]]])[0, 0])
        raw["Time"]   = float(scalers["time"].transform([[raw["Time"]]])[0, 0])

        # Assemble feature vector in the correct order (shape: [1, 30])
        x = np.array([[raw[feat] for feat in feature_order]], dtype=np.float32)

        # Forward pass through autoencoder
        with torch.no_grad():
            reconstruction = model(torch.FloatTensor(x)).numpy()

        # Weighted MSE anomaly score
        score = float(
            np.average((x - reconstruction) ** 2, axis=1, weights=weights)[0]
        )

        return PredictionResponse(
            anomaly_score=round(score, 6),
            is_fraud=score >= threshold,
            threshold=round(threshold, 6),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
