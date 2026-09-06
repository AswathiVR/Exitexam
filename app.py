from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from model_utils import MODEL_COLUMNS, load_data, make_models


st.set_page_config(page_title="Telco Churn Dashboard", page_icon="📊", layout="wide")
BASE_DIR = Path(__file__).parent
MODEL_FILE = BASE_DIR / "forest_model.joblib"


@st.cache_data
def get_data():
    return load_data()


@st.cache_resource
def get_model():
    try:
        if MODEL_FILE.exists():
            return joblib.load(MODEL_FILE)
    except Exception:
        st.warning("The saved model could not be loaded, so the model is being rebuilt.")

    model = make_models()["forest"]
    model.fit(load_data()[MODEL_COLUMNS], load_data()["Churn"])
    return model


data = get_data()
model = get_model()
features = data[MODEL_COLUMNS]
probabilities = model.predict_proba(features)[:, 1]

st.title("Telco Customer Churn Dashboard")
st.write("Explore churn patterns and estimate retention risk using the saved Random Forest model.")

first, second, third = st.columns(3)
first.metric("Overall churn rate", f"{data['Churn'].mean():.2%}")
second.metric("Customers", f"{len(data):,}")
third.metric("Average predicted risk", f"{probabilities.mean():.2%}")

st.subheader("Churn by contract")
contract_table = data.groupby("Contract")["Churn"].agg(Customers="count", Churn_Rate="mean")
contract_table["Churn_Rate"] *= 100
st.dataframe(contract_table.style.format({"Churn_Rate": "{:.2f}%"}), use_container_width=True)

figure, axis = plt.subplots()
contract_table["Churn_Rate"].plot(kind="bar", ax=axis, color="#1976D2")
axis.set_title("Churn Rate by Contract Type")
axis.set_xlabel("Contract")
axis.set_ylabel("Churn rate (%)")
axis.tick_params(axis="x", rotation=0)
st.pyplot(figure)
plt.close(figure)

st.subheader("Customer risk calculator")
st.write("Enter service and billing information. Demographic fields are not used by the saved model.")

with st.form("risk_form"):
    left, right = st.columns(2)
    with left:
        tenure = st.number_input("Tenure (months)", 0, 72, 12)
        monthly_charges = st.number_input("Monthly charges ($)", 0.0, 200.0, 70.0)
        total_charges = st.number_input("Total charges ($)", 0.0, 10000.0, 840.0)
        contract = st.selectbox("Contract", sorted(data["Contract"].unique()))
        internet_service = st.selectbox("Internet service", sorted(data["InternetService"].unique()))
        payment_method = st.selectbox("Payment method", sorted(data["PaymentMethod"].unique()))
    with right:
        phone_service = st.selectbox("Phone service", sorted(data["PhoneService"].unique()))
        multiple_lines = st.selectbox("Multiple lines", sorted(data["MultipleLines"].unique()))
        paperless_billing = st.selectbox("Paperless billing", sorted(data["PaperlessBilling"].unique()))
        online_security = st.selectbox("Online security", sorted(data["OnlineSecurity"].unique()))
        tech_support = st.selectbox("Tech support", sorted(data["TechSupport"].unique()))
    submitted = st.form_submit_button("Calculate churn risk")

if submitted:
    customer = pd.DataFrame([{
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": "No internet service" if internet_service == "No" else "No",
        "DeviceProtection": "No internet service" if internet_service == "No" else "No",
        "TechSupport": tech_support,
        "StreamingTV": "No internet service" if internet_service == "No" else "No",
        "StreamingMovies": "No internet service" if internet_service == "No" else "No",
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }])[MODEL_COLUMNS]
    risk = model.predict_proba(customer)[0, 1]
    st.metric("Predicted churn risk", f"{risk:.2%}")
    if risk >= 0.60:
        st.warning("High risk: consider an approved retention action.")
    else:
        st.success("Lower risk: continue normal customer care and monitoring.")

st.caption("Decision-support demonstration only. Do not use predictions as the sole basis for customer treatment.")