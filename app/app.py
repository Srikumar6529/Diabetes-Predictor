import streamlit as st
import numpy as np
import joblib
import os
import plotly.graph_objects as go

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Healthcare Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

# -----------------------------
# Custom Styling
# -----------------------------
st.markdown("""
<style>
.main {padding: 1rem;}
h1 {color: #1f77b4;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model, Scaler, Threshold
# -----------------------------
@st.cache_resource
def load_artifacts():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = os.path.join(BASE_DIR, "model")

    model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    threshold = joblib.load(os.path.join(MODEL_DIR, "threshold.pkl"))

    return model, scaler, threshold


model, scaler, threshold = load_artifacts()

# -----------------------------
# Header
# -----------------------------
st.title("🏥 Healthcare Risk Prediction System")
st.markdown("### AI-powered decision support for early diabetes detection")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.title("⚙️ Patient Information")

st.sidebar.subheader("Demographics")
age = st.sidebar.slider("Age", 10, 100, 30)
pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 0)

st.sidebar.subheader("Medical Measurements")
glucose = st.sidebar.slider("Glucose", 0, 200, 120)
bp = st.sidebar.slider("Blood Pressure", 0, 140, 70)
skin = st.sidebar.slider("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.slider("Insulin", 0, 900, 80)
bmi = st.sidebar.number_input("BMI", 10.0, 50.0, 25.0)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🔮 Predict", use_container_width=True)

# -----------------------------
# Prediction Logic
# -----------------------------
if predict_btn:

    # Prepare input
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict probability
    prob = model.predict_proba(input_scaled)[0][1]
    prediction = 1 if prob >= threshold else 0

    prob_percent = prob * 100

    # -----------------------------
    # Results Section
    # -----------------------------
    st.markdown("---")
    st.header("🎯 Prediction Results")

    col1, col2 = st.columns([2, 1])

    # LEFT SIDE
    with col1:

        if prob >= 0.7:
            st.error("🔴 HIGH RISK: Immediate medical attention recommended")
        elif prob >= threshold:
            st.warning("🟠 MODERATE RISK: Monitor and consult a doctor")
        else:
            st.success("🟢 LOW RISK")

        st.subheader("Risk Probability")
        st.metric("Diabetes Risk", f"{prob_percent:.1f}%")

    # RIGHT SIDE (Gauge)
    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_percent,
            number={'suffix': "%"},
            title={'text': "Risk Level"},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Risk Factor Analysis
    # -----------------------------
    st.markdown("---")
    st.subheader("⚠️ Risk Factor Analysis")

    risks = []
    positives = []

    if glucose > 125:
        risks.append("🔴 High Glucose (>125)")
    else:
        positives.append("🟢 Healthy Glucose")

    if bmi > 30:
        risks.append("🔴 High BMI (>30)")
    elif 18.5 <= bmi <= 25:
        positives.append("🟢 Healthy BMI")

    if age > 45:
        risks.append("🟡 Age Risk (>45)")

    if bp > 80:
        risks.append("🔴 High Blood Pressure")
    else:
        positives.append("🟢 Normal Blood Pressure")

    if dpf > 0.5:
        risks.append("🟡 Genetic Risk")

    if risks:
        st.warning("**Risk Factors:**")
        for r in risks:
            st.write(f"- {r}")

    if positives:
        st.success("**Positive Indicators:**")
        for p in positives:
            st.write(f"- {p}")

    # -----------------------------
    # Recommendations
    # -----------------------------
    st.markdown("---")
    st.subheader("💡 Recommendations")

    if prediction == 1:
        st.error("""
        - Consult a healthcare professional
        - Monitor glucose regularly
        - Consider lifestyle changes
        """)
    else:
        st.success("""
        - Maintain healthy diet
        - Exercise regularly
        - Periodic health check-ups
        """)

    # -----------------------------
    # Disclaimer
    # -----------------------------
    st.markdown("---")
    st.warning("""
    ⚠️ This tool is for educational purposes only and does not replace professional medical advice.
    """)

# -----------------------------
# Initial Screen
# -----------------------------
else:
    st.info("👈 Enter patient data in the sidebar and click Predict")

    col1, col2, col3 = st.columns(3)
    col1.metric("Model", "Random Forest")
    col2.metric("ROC-AUC", "~0.83")
    col3.metric("Focus", "High Recall (Healthcare)")