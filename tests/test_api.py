"""
tests/test_api.py
-----------------
Test suite for the fraud-detection API using FastAPI's TestClient.

Run from the repo root (where config.json + artifacts live):
    pip install -r requirements-dev.txt
    pytest -v

The two sample transactions are REAL rows from creditcard.csv:
- NORMAL: first row of the dataset (Class=0)
- FRAUD:  a clearly anomalous fraud (Class=1, Time=100924) — high reconstruction
          error, well above threshold. (The dataset's first fraud is borderline
          and not reliably flagged, so it's a poor test fixture.)
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
    "Time": 100924.0, "V1": -23.984747, "V2": 16.697832, "V3": -22.209875, "V4": 9.584969,
    "V5": -16.230439, "V6": 2.596333, "V7": -33.239328, "V8": -21.560039, "V9": -10.842526,
    "V10": -19.836149, "V11": 3.223233, "V12": -10.895134, "V13": -1.523452, "V14": 0.116303,
    "V15": -3.098805, "V16": -7.606425, "V17": -18.108261, "V18": -7.511866, "V19": -1.243285,
    "V20": 5.804551, "V21": -12.615023, "V22": 5.774087, "V23": 2.750221, "V24": 0.513411,
    "V25": -1.608804, "V26": -0.459624, "V27": -4.626127, "V28": -0.334561, "Amount": 1.0,
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
