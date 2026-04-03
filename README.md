# Credit Card Fraud Detection with Autoencoders

Unsupervised anomaly detection on the [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data) dataset using a PyTorch autoencoder.

---

## Approach

The key idea is to train an autoencoder **exclusively on normal transactions**. The model learns to reconstruct legitimate patterns. At inference time, fraudulent transactions — never seen during training — produce a higher reconstruction error, which serves as the anomaly score.

This is a fully **unsupervised** approach: no fraud labels are used during training.

### Weighted Anomaly Score

Plain MSE treats all 30 features equally. A weighted variant is computed by assigning each feature a weight proportional to how much its reconstruction error differs between frauds and normal transactions:

```
w_i  =  max(0,  MSE_fraud_i  -  MSE_normal_i)

score(x)  =  Σ  w_i · (x_i - x̂_i)²
```

Weights are estimated on a held-out **validation set** to avoid data leakage on the test set.

---

## Results

Evaluated on a held-out test set (20% of the dataset, stratified). Threshold selected to maximise F1 on the fraud class.

| Model | F1 (Fraud) | Precision | Recall | ROC-AUC | Avg Precision |
|---|---|---|---|---|---|
| AE — MSE baseline | 0.61 | 0.60 | 0.63 | 0.941 | 0.571 |
| AE — Weighted MSE | **0.80** | **0.79** | **0.82** | 0.936 | **0.682** |

The weighted score raises F1 from 0.61 to **0.80** with no retraining — only a change in how the anomaly score is computed.

---

## Architecture

```
Encoder:  30 → 20 → 10 → 5  (ReLU activations)
Decoder:   5 → 10 → 20 → 30
```

No Dropout or BatchNorm. For anomaly detection, the model must reconstruct normal transactions as precisely as possible — regularisation adds noise to reconstructions and weakens the anomaly signal.

---

## REST API

The model is deployed as a public REST API built with **FastAPI** and served via **Docker** on Render.

**Live endpoint:** `https://fraud-detection-api-cccu.onrender.com/docs`

### Predict endpoint

```
POST /predict
```

Request body (JSON):
```json
{
  "Time": 0.0,
  "V1": -1.3598, "V2": -0.0728, "...",
  "V28": -0.021,
  "Amount": 149.62
}
```

Response:
```json
{
  "anomaly_score": 0.034521,
  "is_fraud": false,
  "threshold": 11.979988
}
```

`Time` and `Amount` are raw values — the API standardises them internally. `V1–V28` are passed as-is (already PCA-scaled in the dataset).

### Run locally with Docker

```bash
# 1. Generate scaler and threshold artifacts (requires creditcard.csv)
python save_artifacts.py

# 2. Build and start
docker compose up --build

# 3. Open interactive docs
http://localhost:8000/docs
```

---

## Project Structure

```
├── pytorch_model.ipynb   # main notebook
├── autoencoder.pt        # saved model weights
├── weights.npy           # saved feature weights
├── scaler.pkl            # fitted StandardScalers for Time and Amount
├── config.json           # threshold and feature order
├── save_artifacts.py     # generates scaler.pkl and config.json
├── requirements.txt      # pinned dependencies
├── Dockerfile            # CPU-only PyTorch image
├── docker-compose.yml    # single-command deploy
└── app/
    └── main.py           # FastAPI application
```

> `creditcard.csv` is not included in this repository due to file size. Download it from Kaggle (see Setup).

---

## Setup

### 1. Dataset

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data) and place it in the project folder.

### 2. Dependencies

```bash
pip install torch numpy pandas scikit-learn matplotlib
```

Or with conda:

```bash
conda install pytorch numpy pandas scikit-learn matplotlib -c pytorch
```

### 3. Run

Open `pytorch_model.ipynb` and run all cells.  
On the first run the model trains (~1 min) and saves `autoencoder.pt` and `weights.npy`.  
On subsequent runs the saved files are loaded automatically — set `FORCE_RETRAIN = True` to retrain from scratch.

---

## Notebook Structure

| Section | Description |
|---|---|
| Dataset | Load and inspect the data |
| Preprocessing | Standardise `Amount` and `Time`, three-way train/val/test split |
| Autoencoder Architecture | Model definition |
| Training | Train on normal transactions only, save/load checkpoint |
| Anomaly Score | MSE baseline → Weighted MSE |
| Evaluation | Classification report, confusion matrix |
| Explainability | Per-feature reconstruction error, permutation feature importance |

---

## Dataset

- **Source:** [ULB Machine Learning Group](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)
- 284,807 transactions over two days
- 492 frauds (0.17%) — highly imbalanced
- Features V1-V28: PCA components (anonymised); `Time` and `Amount` in original scale
