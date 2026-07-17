import streamlit as st
import pandas as pd
import joblib
import os
from geopy.distance import geodesic

st.set_page_config(page_title="💳 Credit Card Fraud Detector", layout="wide")

# ---------- Custom CSS ----------
st.markdown("""
    <style>
    body { background-color: #f6f9fc; }
    .main-title {
        text-align: center;
        color: #2c3e50;
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .sub-title {
        text-align: center;
        color: #7f8c8d;
        font-size: 18px;
        margin-bottom: 35px;
    }
    .card {
        background-color: white;
        padding: 25px 40px;
        border-radius: 20px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 10px;
        color: #2e7d32;
        font-weight: bold;
        text-align: center;
    }
    .error-box {
        background-color: #ffebee;
        padding: 15px;
        border-radius: 10px;
        color: #c62828;
        font-weight: bold;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Title ----------
st.markdown("<div class='main-title'>💳 Credit Card Fraud Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Enter a Credit Card Number to Auto-Fill Transaction Details and Predict Fraud</div>", unsafe_allow_html=True)

# ---------- Load Model ----------
model = None
MODEL_FILES = ["fraud_detection_model.jb", "model.pkl", "fraud_model.pkl"]
for f in MODEL_FILES:
    if os.path.exists(f):
        model = joblib.load(f)
        st.sidebar.success(f"✅ Loaded model: {f}")
        break
if model is None:
    st.sidebar.error("❌ No model file found. Please place your model file in this folder.")
    st.stop()

# ---------- Load Label Encoders ----------
encoders = joblib.load("label_encoder.jb")

# ---------- Load Data ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset_small.csv")

data = pd.read_csv(DATA_PATH)

data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
data['hour'] = data['trans_date_trans_time'].dt.hour
data['day'] = data['trans_date_trans_time'].dt.day
data['month'] = data['trans_date_trans_time'].dt.month
data['cc_num_str'] = data['cc_num'].astype(str).str.replace(r'\.0$', '', regex=True)
data["distance"] = data.apply(
    lambda row: geodesic(
        (row["lat"], row["long"]),
        (row["merch_lat"], row["merch_long"])
    ).km,
    axis=1
)
# ---------- User Input ----------
st.markdown("<div class='card'>", unsafe_allow_html=True)
cc_input = st.text_input("🔢 Enter Credit Card Number (or last few digits)", placeholder="Example: 4922710831011201")
st.markdown("</div>", unsafe_allow_html=True)

if cc_input:
    matches = data[data['cc_num_str'].str.endswith(cc_input.strip())]
    if matches.empty:
        st.error("❌ No matching transactions found.")
    else:
        st.write(f"**Found {len(matches)} transaction(s)** with card ending `{cc_input}`.")
        matches_display = matches[['trans_date_trans_time', 'merchant', 'category', 'amt', 'is_fraud', 'city', 'trans_num']].reset_index(drop=True)
        st.dataframe(matches_display)

        selected_idx = st.selectbox("Select the transaction row to auto-fill", matches_display.index)
        chosen = matches.iloc[selected_idx]

        # ---------- Autofill ----------
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🧾 Transaction Details")
        merchant = chosen["merchant"]
        category = chosen["category"]
        gender = chosen["gender"]

        st.text_input("Merchant", merchant, disabled=True)
        st.text_input("Category", category, disabled=True)
        st.text_input("Gender", gender, disabled=True)
        amt = float(chosen["amt"])
        hour = int(chosen["hour"])
        day = int(chosen["day"])
        month = int(chosen["month"])

        st.text_input("Amount ($)", f"{amt}", disabled=True)
        st.text_input("Hour", str(hour), disabled=True)
        st.text_input("Day", str(day), disabled=True)
        st.text_input("Month", str(month), disabled=True)
        distance = float(chosen["distance"])

        st.text_input("Distance (km)",value=f"{distance:.2f}",disabled=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ---------- Predict ----------
        if st.button("🚀 Predict Fraud Status"):
            X = pd.DataFrame([{
                'merchant': merchant,
                'category': category,
                'amt': amt,
                'cc_num': int(chosen['cc_num']),
                'hour': hour,
                'day': day,
                'month': month,
                'gender': gender,
                'distance': distance
            }])

           

            # --- Encode Categorical Columns using saved encoders ---
            for col in ["merchant", "category", "gender"]:
                X[col] = encoders[col].transform([X[col].iloc[0]])[0]
            try:
                y_pred = int(model.predict(X)[0])
                y_prob = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else None
                actual = 'FRAUD' if chosen['is_fraud'] == 1 else 'LEGITIMATE'

                if y_pred == 1:
                    st.markdown(f"<div class='error-box'>🚨 Prediction: FRAUD DETECTED (Prob={y_prob:.3f})</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='success-box'>✅ Prediction: LEGITIMATE (Prob={y_prob:.3f})</div>", unsafe_allow_html=True)

                st.info(f"**Actual Label (from dataset):** {actual}")
            except Exception as e:
                st.error(f"⚠️ Prediction failed: {e}")
