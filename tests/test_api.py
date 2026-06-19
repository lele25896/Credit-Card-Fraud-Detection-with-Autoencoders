"""
tests/test_api.py
-----------------
Test suite for the fraud-detection API using FastAPI's TestClient.

Run from the repo root (where config.json + artifacts live):
    pip install -r requirements-dev.txt
    pytest -v

The two sample transactions are REAL rows from creditcard.csv:
- NORMAL: first row of the dataset (Class=0)
- FRAUD:  first fraudulent row of the dataset (Class=1, Time=406)
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    """Context-managed client so the lifespan startup loads the artifacts."""
    with TestClient(app) as c:
        yield c


# Real dataset rows (Class label removed) ------------------------------------
NORMAL_TX = {
    "Time": 0.0, "V1": -1.359807, "V2": -0.072781, "V3": 2.536347, "V4": 1.378155,
    "V5": -0.338321, "V6": 0.462388, "V7": 0.239599, "V8": 0.098698, "V9": 0.363787,
    "V10": 0.090794, "V11": -0.5516, "V12": -0.617801, "V13": -0.99139, "V14": -0.311169,
    "V15": 1.468177, "V16": -0.470401, "V17": 0.207971, "V18": 0.025791, "V19": 0.403993,
    "V20": 0.251412, "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
    "V25": 0.128539, "V26": -0.189115, "V27": 0.133558, "V28": -0.021053, "Amount": 149.62,
}

FRAUD_TX = {
    "Time": 406.0, "V1": -2.312227, "V2": 1.951992, "V3": -1.609851, "V4": 3.997906,
    "V5": -0.522188, "V6": -1.426545, "V7": -2.537387, "V8": 1.391657, "V9": -2.770089,
    "V10": -2.772272, "V11": 3.202033, "V12": -2.899907, "V13": -0.595222, "V14": -4.289254,
    "V15": 0.389724, "V16": -1.140747, "V17": -2.830056, "V18": -0.016822, "V19": 0.416956,
    "V20": 0.126911, "V21": 0.517232, "V22": -0.035049, "V23": -0.465211, "V24": 0.320198,
    "V25": 0.044519, "V26": 0.17784, "V27": 0.261145, "V28": -0.143276, "Amount": 0.0,
}


def test_health(client):
    """GET /health returns 200 and the expected payload."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model": "autoencoder"}


def test_predict_schema(client):
    """POST /predict returns 200 with all expected fields and correct types."""
    r = client.post("/predict", json=NORMAL_TX)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"anomaly_score", "is_fraud", "threshold"}
    assert isinstance(body["anomaly_score"], (int, float))
    assert isinstance(body["is_fraud"], bool)
    assert body["threshold"] > 0


def test_normal_transaction_is_not_fraud(client):
    """A legitimate transaction stays below the threshold."""
    r = client.post("/predict", json=NORMAL_TX)
    assert r.status_code == 200
    body = r.json()
    assert body["is_fraud"] is False
    assert body["anomaly_score"] < body["threshold"]


def test_fraud_transaction_is_detected(client):
    """A known fraudulent transaction is flagged."""
    r = client.post("/predict", json=FRAUD_TX)
    assert r.status_code == 200
    body = r.json()
    assert body["is_fraud"] is True
    assert body["anomaly_score"] >= body["threshold"]


def test_fraud_scores_higher_than_normal(client):
    """Sanity check: fraud anomaly score exceeds the normal one."""
    normal = client.post("/predict", json=NORMAL_TX).json()["anomaly_score"]
    fraud = client.post("/predict", json=FRAUD_TX).json()["anomaly_score"]
    assert fraud > normal


def test_predict_rejects_missing_field(client):
    """An incomplete payload is rejected with 422 (validation error)."""
    incomplete = {k: v for k, v in NORMAL_TX.items() if k != "Amount"}
    r = client.post("/predict", json=incomplete)
    assert r.status_code == 422
