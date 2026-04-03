"""
save_artifacts.py
-----------------
Run once from the project root (requires creditcard.csv):

    python save_artifacts.py

Produces:
    scaler.pkl   — {"amount": StandardScaler, "time": StandardScaler}
    config.json  — threshold, feature_order, and validation metrics
"""

import json
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Reproducibility (must match the notebook) ─────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)


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


# ── 1. Load raw data ───────────────────────────────────────────────────────────
print("Loading creditcard.csv ...")
df = pd.read_csv("creditcard.csv")
print(f"  {len(df):,} rows, {df['Class'].sum()} frauds")

# ── 2. Fit scalers on raw columns BEFORE any transformation ───────────────────
# NOTE: the notebook reuses one StandardScaler instance, calling fit_transform
# on Amount first then on Time — which overwrites the Amount fit.
# We use two separate scalers, each fitted independently on the raw column.
scaler_amount = StandardScaler()
scaler_time   = StandardScaler()
scaler_amount.fit(df[["Amount"]])
scaler_time.fit(df[["Time"]])

# ── 3. Apply the same transforms the notebook applied ─────────────────────────
df["Amount"] = scaler_amount.transform(df[["Amount"]])
df["Time"]   = scaler_time.transform(df[["Time"]])

# ── 4. Reproduce the exact same splits as the notebook ────────────────────────
feature_order = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
X = df[feature_order].values
y = df["Class"].values

X_tv, X_test, y_tv, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=0.15, stratify=y_tv, random_state=42
)

print(f"  Train: {X_train.shape[0]:,} | Val: {X_val.shape[0]:,} | Test: {X_test.shape[0]:,}")
print(f"  Val frauds: {y_val.sum()}")

# ── 5. Load the trained model ──────────────────────────────────────────────────
model = Autoencoder(input_dim=30)
model.load_state_dict(torch.load("autoencoder.pt", weights_only=True))
model.eval()
print("Model loaded.")

# ── 6. Load pre-computed feature weights ──────────────────────────────────────
weights = np.load("weights.npy")

# ── 7. Compute anomaly scores on the validation set ───────────────────────────
X_val_t = torch.FloatTensor(X_val)
with torch.no_grad():
    val_rec = model(X_val_t).numpy()

w_err_val = np.average((X_val - val_rec) ** 2, axis=1, weights=weights)

# ── 8. Find optimal threshold (maximise F1 on fraud class) ────────────────────
prec, rec, thr = precision_recall_curve(y_val, w_err_val)
f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-8)
best_idx  = int(np.argmax(f1))
threshold = float(thr[best_idx])

print(f"\nOptimal threshold : {threshold:.6f}")
print(f"Val F1            : {f1[best_idx]:.4f}")
print(f"Val Precision     : {prec[best_idx]:.4f}")
print(f"Val Recall        : {rec[best_idx]:.4f}")

# ── 9. Save scalers ────────────────────────────────────────────────────────────
scalers = {"amount": scaler_amount, "time": scaler_time}
with open("scaler.pkl", "wb") as f:
    pickle.dump(scalers, f)
print("\nSaved scaler.pkl")

# ── 10. Save config ────────────────────────────────────────────────────────────
config = {
    "threshold":     threshold,
    "model_path":    "autoencoder.pt",
    "weights_path":  "weights.npy",
    "scaler_path":   "scaler.pkl",
    "input_dim":     30,
    "feature_order": feature_order,
    "val_f1":        round(float(f1[best_idx]), 4),
    "val_precision": round(float(prec[best_idx]), 4),
    "val_recall":    round(float(rec[best_idx]), 4),
}
with open("config.json", "w") as f:
    json.dump(config, f, indent=2)
print("Saved config.json")
print("\nDone. You can now build the Docker image.")
