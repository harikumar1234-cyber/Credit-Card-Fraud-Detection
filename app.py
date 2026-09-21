import streamlit as st #type:ignore
import pandas as pd #type:ignore
import numpy as np #type:ignore
import pickle

st.set_page_config(page_title="Fraud Detector", page_icon="💳", layout="centered")

# -----------------------------
# Load model + scaler
# -----------------------------
@st.cache_resource
def load_artifacts():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("model_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return model, scaler, meta

model, scaler, meta = load_artifacts()
feature_cols = meta["feature_columns"]

st.title("💳 Credit Card Fraud Detector")


st.markdown(
    f"**Model in use:** `{meta['best_model_name']}`  |  "
    f"**Test F1-score:** `{meta['f1']:.4f}`"
)
st.info(
    "This dataset's V1–V28 features are PCA-anonymized (real bank data, "
    "anonymized for privacy) so they can't be entered manually in a "
    "meaningful way. Use a real transaction row from the test set below, "
    "or upload a CSV with the same columns."
)

st.divider()

# -----------------------------
# Option 1: Try a sample transaction
# -----------------------------
st.subheader("Option 1: Try a sample transaction")

@st.cache_data
def load_sample_rows():
    df = pd.read_csv("creditcard.csv")
    df["scaled_amount"] = scaler.fit_transform(df["Amount"].values.reshape(-1, 1))
    df["scaled_time"] = scaler.fit_transform(df["Time"].values.reshape(-1, 1))
    df = df.drop(["Time", "Amount"], axis=1)
    fraud_sample = df[df["Class"] == 1].sample(3, random_state=1)
    legit_sample = df[df["Class"] == 0].sample(3, random_state=1)
    return pd.concat([fraud_sample, legit_sample]).reset_index(drop=True)

try:
    samples = load_sample_rows()
    sample_idx = st.selectbox(
        "Pick a sample transaction:",
        options=list(range(len(samples))),
        format_func=lambda i: f"Sample {i+1} (true label: {'FRAUD' if samples.loc[i,'Class']==1 else 'Legit'})"
    )
    if st.button("Analyze Sample", type="primary"):
        row = samples.loc[sample_idx, feature_cols]
        pred = model.predict([row.values])[0]
        proba = model.predict_proba([row.values])[0][pred]

        st.divider()
        if pred == 1:
            st.error(f"🚩 Prediction: **FRAUD** (confidence: {proba*100:.1f}%)")
        else:
            st.success(f"✅ Prediction: **Legitimate** (confidence: {proba*100:.1f}%)")

        true_label = "FRAUD" if samples.loc[sample_idx, "Class"] == 1 else "Legit"
        st.caption(f"True label for this sample: **{true_label}**")
except FileNotFoundError:
    st.warning("creditcard.csv not found in this folder — sample mode needs it. "
               "Manual CSV upload (below) still works.")

st.divider()

# -----------------------------
# Option 2: Upload CSV
# -----------------------------
st.subheader("Option 2: Upload transactions CSV")
st.caption("CSV must have the same columns as the training data (Time, V1-V28, Amount).")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is not None:
    new_df = pd.read_csv(uploaded)
    try:
        new_df["scaled_amount"] = scaler.fit_transform(new_df["Amount"].values.reshape(-1, 1))
        new_df["scaled_time"] = scaler.fit_transform(new_df["Time"].values.reshape(-1, 1))
        X_new = new_df[feature_cols]
        preds = model.predict(X_new)
        probs = model.predict_proba(X_new)[:, 1]

        result_df = new_df.copy()
        result_df["Prediction"] = np.where(preds == 1, "FRAUD", "Legit")
        result_df["Fraud_Probability"] = probs

        st.dataframe(result_df[["Prediction", "Fraud_Probability"]])
        st.metric("Flagged as Fraud", int(preds.sum()))
    except KeyError as e:
        st.error(f"Missing expected column: {e}")

st.divider()
st.caption(
    "Note: Precision/recall trade-off matters here — this model favors "
    "catching more fraud (recall) even at the cost of some false alarms, "
    "which is often the right choice in real fraud systems."
)
