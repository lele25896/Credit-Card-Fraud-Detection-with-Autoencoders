"""
dashboard.py
------------
Streamlit dashboard for the Credit Card Fraud Detection API.

Run locally:
    pip install streamlit requests pandas
    streamlit run dashboard.py
"""

import io

import pandas as pd
import requests
import streamlit as st

API_URL = "https://fraud-detection-api-cccu.onrender.com"

# ── Preset transactions ────────────────────────────────────────────────────────
PRESETS = {
    "Normal transaction (example)": {
        "Time": 0.0, "V1": -1.3598, "V2": -0.0728, "V3": 2.5363,
        "V4": 1.3782, "V5": -0.3383, "V6": 0.4624, "V7": 0.2396,
        "V8": 0.0987, "V9": 0.3638, "V10": 0.0908, "V11": -0.5516,
        "V12": -0.6178, "V13": -0.9913, "V14": -0.3112, "V15": 1.4682,
        "V16": -0.4704, "V17": 0.2079, "V18": 0.0258, "V19": 0.4031,
        "V20": 0.2514, "V21": -0.0183, "V22": 0.2778, "V23": -0.1105,
        "V24": 0.0669, "V25": 0.1285, "V26": -0.1891, "V27": 0.1336,
        "V28": -0.0210, "Amount": 149.62,
    },
    "Fraudulent transaction (example)": {
        "Time": 406.0, "V1": -3.0435, "V2": -3.1572, "V3": 1.0877,
        "V4": 2.2884, "V5": 4.5077, "V6": -2.1398, "V7": -5.1676,
        "V8": -2.7359, "V9": -7.8966, "V10": -9.6022, "V11": 5.6274,
        "V12": -11.8817, "V13": 1.3466, "V14": -17.8026, "V15": -3.2815,
        "V16": -8.3687, "V17": -20.5001, "V18": -5.8702, "V19": -0.2208,
        "V20": 0.5587, "V21": -0.0804, "V22": -0.3441, "V23": 0.3527,
        "V24": -0.1875, "V25": 0.6025, "V26": 0.3382, "V27": 0.3419,
        "V28": 0.2700, "Amount": 390.0,
    },
}

FEATURE_ORDER = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
)

st.title("Credit Card Fraud Detection")
st.caption(
    "Autoencoder-based anomaly detection · "
    f"[API docs]({API_URL}/docs) · F1: 0.80 · ROC-AUC: 0.94"
)

# ── Helper ─────────────────────────────────────────────────────────────────────
def call_predict(features: dict) -> dict | None:
    try:
        r = requests.post(f"{API_URL}/predict", json=features, timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        st.error("Request timed out. The API may be starting up (cold start ~30s). Try again.")
    except requests.exceptions.RequestException as e:
        st.error(f"API error: {e}")
    return None


def show_result(result: dict):
    col1, col2, col3 = st.columns(3)
    col1.metric("Anomaly Score", f"{result['anomaly_score']:.4f}")
    col2.metric("Threshold", f"{result['threshold']:.4f}")
    if result["is_fraud"]:
        col3.error("FRAUD DETECTED", icon="🚨")
        st.progress(min(result["anomaly_score"] / (result["threshold"] * 3), 1.0))
    else:
        col3.success("Normal transaction", icon="✅")
        st.progress(min(result["anomaly_score"] / (result["threshold"] * 3), 1.0))


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Single Transaction", "Batch Analysis"])

# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — Single transaction
# ────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Analyze a single transaction")

    preset_name = st.selectbox("Load a preset", list(PRESETS.keys()))
    preset = PRESETS[preset_name]

    # Key discriminative features as sliders
    st.markdown("**Key features** (most influential on fraud detection):")
    c1, c2, c3 = st.columns(3)
    v17 = c1.slider("V17 (weight: 9.64)", -15.0, 15.0, float(preset["V17"]), 0.01)
    v12 = c2.slider("V12 (weight: 3.58)", -15.0, 15.0, float(preset["V12"]), 0.01)
    v14 = c3.slider("V14 (weight: 3.30)", -15.0, 15.0, float(preset["V14"]), 0.01)

    c4, c5, c6 = st.columns(3)
    v16 = c4.slider("V16 (weight: 2.79)", -15.0, 15.0, float(preset["V16"]), 0.01)
    v10 = c5.slider("V10 (weight: 2.71)", -15.0, 15.0, float(preset["V10"]), 0.01)
    amount = c6.number_input("Amount (€)", min_value=0.0, value=float(preset["Amount"]), step=1.0)

    # Remaining features hidden in expander
    remaining = {k: v for k, v in preset.items()
                 if k not in ("V17", "V12", "V14", "V16", "V10", "Amount")}
    with st.expander("Other features (V1–V11, V13, V15, V18–V28, Time)"):
        cols = st.columns(5)
        inputs = {}
        for i, (k, v) in enumerate(remaining.items()):
            inputs[k] = cols[i % 5].number_input(k, value=float(v), format="%.4f", key=f"inp_{k}")

    # Build full feature dict
    features = {
        **inputs,
        "V17": v17, "V12": v12, "V14": v14,
        "V16": v16, "V10": v10, "Amount": amount,
    }

    if st.button("Analyze transaction", type="primary"):
        with st.spinner("Calling API..."):
            result = call_predict(features)
        if result:
            show_result(result)

# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — Batch analysis
# ────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Analyze a batch of transactions")
    st.markdown(
        "Upload a CSV file with columns: `Time`, `V1`–`V28`, `Amount`. "
        "A `Class` column (0/1) is optional — if present it will be used to show ground truth."
    )

    # Download template
    template_df = pd.DataFrame([PRESETS["Normal transaction (example)"],
                                 PRESETS["Fraudulent transaction (example)"]])
    template_csv = template_df.to_csv(index=False)
    st.download_button(
        "Download CSV template",
        data=template_csv,
        file_name="transactions_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload transactions CSV", type="csv")

    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(f"**{len(df)} transactions loaded.** Preview:")
        st.dataframe(df.head(), use_container_width=True)

        missing = [c for c in FEATURE_ORDER if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            if st.button("Analyze all transactions", type="primary"):
                results = []
                progress = st.progress(0)
                status = st.empty()

                for i, row in df.iterrows():
                    features = {col: float(row[col]) for col in FEATURE_ORDER}
                    result = call_predict(features)
                    if result:
                        results.append({
                            "#": i,
                            "Amount (€)": row["Amount"],
                            **({"True Label": "Fraud" if row["Class"] == 1 else "Normal"}
                               if "Class" in df.columns else {}),
                            "Anomaly Score": round(result["anomaly_score"], 4),
                            "Prediction": "🚨 Fraud" if result["is_fraud"] else "✅ Normal",
                            "is_fraud": result["is_fraud"],
                        })
                    progress.progress((i + 1) / len(df))
                    status.text(f"Processed {i + 1}/{len(df)} transactions...")

                progress.empty()
                status.empty()

                if results:
                    results_df = pd.DataFrame(results)
                    n_fraud = results_df["is_fraud"].sum()

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total transactions", len(results_df))
                    m2.metric("Flagged as fraud", int(n_fraud))
                    m3.metric("Fraud rate", f"{n_fraud / len(results_df) * 100:.1f}%")

                    st.markdown("**Results** (sorted by anomaly score):")
                    display_df = (
                        results_df
                        .drop(columns=["is_fraud"])
                        .sort_values("Anomaly Score", ascending=False)
                        .reset_index(drop=True)
                    )
                    st.dataframe(
                        display_df.style.apply(
                            lambda row: ["background-color: #ffcccc" if "Fraud" in str(row.get("Prediction", "")) else "" for _ in row],
                            axis=1,
                        ),
                        use_container_width=True,
                    )

                    # Download results
                    csv_out = display_df.to_csv(index=False)
                    st.download_button(
                        "Download results as CSV",
                        data=csv_out,
                        file_name="fraud_analysis_results.csv",
                        mime="text/csv",
                    )
