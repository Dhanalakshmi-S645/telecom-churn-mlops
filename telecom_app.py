import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os

st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📡",
    layout="wide"
)

# Load artifacts
@st.cache_resource
def load_model():
    try:
        model = joblib.load("models/best_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        encoders = joblib.load("models/encoders.pkl")
        feature_names = joblib.load("models/feature_names.pkl")
        model_name = joblib.load("models/best_model_name.pkl")
        return model, scaler, encoders, feature_names, model_name
    except:
        return None, None, None, None, "Not loaded"

model, scaler, encoders, feature_names, model_name = load_model()

# Header
st.markdown("""
<div style='background: linear-gradient(90deg, #1F4E79, #2E86C1); padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
    <h1 style='color: white; margin: 0;'>📡 Telecom Churn Prediction System</h1>
    <p style='color: #AED6F1; margin: 5px 0 0 0;'>End-to-End MLOps Pipeline | Model: <b>{}</b></p>
</div>
""".format(model_name), unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Predict Churn", "📊 Model Comparison", "🏗️ Architecture"])

with tab1:
    st.subheader("Enter Customer Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Demographics**")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    with col2:
        st.markdown("**📞 Services**")
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multi_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    with col3:
        st.markdown("**💳 Billing**")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
        total = st.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure * monthly))

    predict_btn = st.button("🔮 Predict Churn Risk", use_container_width=True, type="primary")

    if predict_btn and model is not None:
        data = {
            "gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
            "MultipleLines": multi_lines, "InternetService": internet,
            "OnlineSecurity": security, "OnlineBackup": backup,
            "DeviceProtection": device, "TechSupport": tech,
            "StreamingTV": tv, "StreamingMovies": movies,
            "Contract": contract, "PaperlessBilling": billing,
            "PaymentMethod": payment, "MonthlyCharges": monthly,
            "TotalCharges": total
        }

        df = pd.DataFrame([data])
        df["tenure_years"] = df["tenure"] / 12
        df["charges_per_month_ratio"] = df["TotalCharges"] / (df["tenure"] + 1)
        df["has_streaming"] = ((df["StreamingTV"] == "Yes") | (df["StreamingMovies"] == "Yes")).astype(int)
        df["num_services"] = (
            (df["PhoneService"] == "Yes").astype(int) +
            (df["MultipleLines"] == "Yes").astype(int) +
            (df["InternetService"] != "No").astype(int) +
            (df["OnlineSecurity"] == "Yes").astype(int) +
            (df["OnlineBackup"] == "Yes").astype(int) +
            (df["DeviceProtection"] == "Yes").astype(int) +
            (df["TechSupport"] == "Yes").astype(int) +
            (df["StreamingTV"] == "Yes").astype(int) +
            (df["StreamingMovies"] == "Yes").astype(int)
        )

        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        for col in cat_cols:
            if col in encoders:
                try:
                    df[col] = encoders[col].transform(df[col])
                except:
                    df[col] = 0

        df = df[feature_names]
        df_scaled = pd.DataFrame(scaler.transform(df), columns=feature_names)
        prob = model.predict_proba(df_scaled)[0][1]

        st.markdown("---")
        r1, r2, r3 = st.columns(3)

        if prob < 0.3:
            risk_color = "#27AE60"
            risk_label = "🟢 LOW RISK"
            recommendation = "Customer is stable. Great candidate for upsell campaigns."
        elif prob < 0.6:
            risk_color = "#F39C12"
            risk_label = "🟡 MEDIUM RISK"
            recommendation = "Monitor closely. Consider offering a loyalty discount or plan upgrade."
        else:
            risk_color = "#E74C3C"
            risk_label = "🔴 HIGH RISK"
            recommendation = "Immediate action needed! Offer contract incentive or personalized retention offer."

        with r1:
            st.metric("Churn Probability", f"{prob:.1%}")
        with r2:
            st.metric("Risk Level", risk_label)
        with r3:
            st.metric("Prediction", "Will Churn" if prob >= 0.5 else "Will Stay")

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Churn Risk Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": risk_color},
                "steps": [
                    {"range": [0, 30], "color": "#D5F5E3"},
                    {"range": [30, 60], "color": "#FDEBD0"},
                    {"range": [60, 100], "color": "#FADBD8"}
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 50}
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        st.info(f"💡 **Recommendation:** {recommendation}")

    elif predict_btn and model is None:
        st.error("Model not loaded. Please run `python src/train.py` first.")

with tab2:
    st.subheader("📊 Model Comparison Results")
    try:
        results_df = pd.read_csv("reports/model_comparison.csv")
        st.dataframe(results_df.style.highlight_max(
            subset=["auc_roc", "f1_score", "accuracy"],
            color="#D5F5E3"
        ), use_container_width=True)

        fig = px.bar(
            results_df.melt(id_vars="model", value_vars=["auc_roc", "f1_score", "precision", "recall"]),
            x="model", y="value", color="variable", barmode="group",
            title="Model Performance Comparison",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.info("Run `python src/train.py` first to generate model comparison data.")

with tab3:
    st.subheader("🏗️ MLOps Pipeline Architecture")
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │                  TELECOM CHURN MLOPS PIPELINE                   │
    └─────────────────────────────────────────────────────────────────┘
    
    📦 DATA LAYER                   🔧 TRAINING LAYER
    ┌──────────────┐                ┌──────────────────────────────┐
    │ IBM Telco    │ → preprocess   │  6 Models Trained:           │
    │ Dataset      │   .py          │  • Logistic Regression       │
    │ 7,043 rows   │                │  • Decision Tree             │
    │ 21 features  │                │  • Random Forest             │
    └──────────────┘                │  • Gradient Boosting         │
                                    │  • XGBoost ← Best            │
    📊 TRACKING LAYER               │  • LightGBM                  │
    ┌──────────────┐                └──────────────────────────────┘
    │   MLflow     │
    │  Experiment  │                🔍 MONITORING LAYER
    │  Tracking    │                ┌──────────────────────────────┐
    │  6 runs      │                │  Evidently AI Reports:       │
    └──────────────┘                │  • Data Drift Detection      │
                                    │  • Model Performance         │
    🚀 SERVING LAYER                │  • Data Quality              │
    ┌──────────────┐                └──────────────────────────────┘
    │  FastAPI     │
    │  /predict    │
    │  /health     │
    │  /docs       │
    └──────────────┘
    
    🖥️  UI LAYER: Streamlit → Deployed on HuggingFace Spaces
    ```
    """)
    st.markdown("""
    **Tech Stack:** Python • Scikit-learn • XGBoost • LightGBM • MLflow • Evidently AI • FastAPI • Streamlit • Plotly
    
    **GitHub:** [github.com/Dhanalakshmi-S645/telecom-churn-mlops](https://github.com/Dhanalakshmi-S645)
    """)
