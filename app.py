import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="UPI Fraud Detector", layout="centered", page_icon="🔍")

# --- CONSTANTS ---
MODEL_PATH = "fraud_detection_nn.keras"
SCALER_PATH = "scaler.pkl"

CATEGORY_LABELS = {
    0: "Entertainment", 1: "Food Dining", 2: "Gas Transport", 3: "Grocery NET",
    4: "Grocery POS", 5: "Health Fitness", 6: "Home", 7: "Kids Pets",
    8: "Miscellaneous NET", 9: "Miscellaneous POS", 10: "Personal Care",
    11: "Shopping NET", 12: "Shopping POS", 13: "Travel"
}

# --- LOADERS ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)

model = load_model()
scaler = load_scaler()

# --- HEADER ---
st.markdown("""
    <h1 style='text-align: center;'>🔍 UPI Fraud Detection</h1>
    <p style='text-align: center; color: grey;'>
        Predict whether a UPI transaction is fraudulent using a trained neural network.
    </p>
    <hr style="margin: 20px 0;">
""", unsafe_allow_html=True)

# --- FORM ---
with st.form("fraud_form"):
    st.subheader("🧾 Transaction Details")

    col1, col2 = st.columns(2)
    with col1:
        transaction_date = st.date_input("Transaction Date", datetime.today())
        age = st.slider("User Age", 18, 100)
        category_label = st.selectbox("Transaction Category", list(CATEGORY_LABELS.values()))
    with col2:
        trans_hour = st.slider("Hour of Transaction", 0, 23)
        trans_amount = st.number_input("Amount (₹)", min_value=0.0, step=0.5)
        upi_number = st.text_input("UPI Number (digits only)", value="1234567890")

    state = st.number_input("State Code", 0, 50, 10)
    zip_code = st.number_input("ZIP Code", 100, 999999, 400001)

    submitted = st.form_submit_button("🚨 Predict Fraud")

    if submitted:
        try:
            category = [k for k, v in CATEGORY_LABELS.items() if v == category_label][0]
            input_data = np.array([[trans_hour,
                                    transaction_date.day,
                                    transaction_date.month,
                                    transaction_date.year,
                                    category,
                                    age,
                                    trans_amount,
                                    state,
                                    zip_code,
                                    float(upi_number)]])
            
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled, verbose=0)[0][0]
            is_fraud = prediction >= 0.5

            st.markdown("---")
            st.metric("🔎 Prediction Score", f"{prediction:.4f}", help="Probability of fraud (≥ 0.5 is considered fraud)")
            if is_fraud:
                st.error("🔴 FRAUD DETECTED — Transaction flagged for review.")
            else:
                st.success("🟢 Transaction is Safe — No fraud detected.")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
