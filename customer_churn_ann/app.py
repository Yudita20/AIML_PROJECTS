
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch import nn
import streamlit.components.v1 as components


# --------------------------------------------------
# 1. PATHS AND MODEL ARCHITECTURE
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"


class ChurnANN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


@st.cache_resource
def load_assets():
    model_file = ART / "churn_ann.pt"
    prep_file = ART / "preprocessor.joblib"

    if not model_file.exists() or not prep_file.exists():
        return None, None, None

    ckpt = torch.load(model_file, map_location="cpu")

    model = ChurnANN(ckpt["input_dim"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, joblib.load(prep_file), ckpt.get("threshold", 0.5)


# --------------------------------------------------
# 2. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ChurnSense | Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# --------------------------------------------------
# 3. CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #F7FAFA;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1150px;
}

h1, h2, h3 {
    color: #134E4A !important;
}

.stMarkdown h2,
.stMarkdown h3 {
    color: #134E4A !important;
    font-weight: 700;
}

div[data-testid="stForm"] {
    background-color: #FFFFFF;
    padding: 25px;
    border: 1px solid #DCEAE7;
    border-radius: 16px;
}

.stSelectbox label,
.stNumberInput label,
.stSlider label {
    color: #334155 !important;
    font-weight: 600 !important;
}

div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    color: #1F2937 !important;
    border-color: #CBD5E1 !important;
}

.stNumberInput input {
    background-color: #FFFFFF !important;
    color: #1F2937 !important;
}

/* Header */

.main-header {
    background: #FFFFFF;
    padding: 30px;
    border: 1px solid #DCEAE7;
    border-radius: 16px;
    margin-bottom: 28px;
}

.brand {
    color: #0F766E;
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -1px;
}

.tagline {
    color: #0F766E;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    margin-top: 4px;
}

.main-header h1 {
    color: #134E4A !important;
    font-size: 32px;
    margin-top: 25px;
    margin-bottom: 8px;
}

.main-header p {
    color: #64748B;
    font-size: 15px;
}

.tech-label {
    display: inline-block;
    background: #E6F4F1;
    color: #0F766E;
    padding: 7px 12px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 700;
    margin-top: 8px;
}

/* Submit button */

div[data-testid="stFormSubmitButton"] > button {
    background-color: #0F766E !important;
    color: white !important;
    border: none !important;
    border-radius: 9px !important;
    padding: 0.75rem 1.2rem !important;
    font-weight: 700 !important;
    width: 100% !important;
    min-height: 48px;
}

div[data-testid="stFormSubmitButton"] > button:hover {
    background-color: #115E59 !important;
    color: white !important;
}

/* Prediction cards */

div[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #DCEAE7;
    padding: 18px;
    border-radius: 12px;
}

/* Prediction result cards */
div[data-testid="stMetric"] {
    background-color: #FFFFFF !important;
    border: 1px solid #DCEAE7;
    padding: 22px;
    border-radius: 14px;
}

/* Metric labels */
div[data-testid="stMetricLabel"] {
    color: #475569 !important;
}

/* Metric values */
div[data-testid="stMetricValue"] {
    color: #134E4A !important;
    font-weight: 750 !important;
}

/* Metric delta text */
div[data-testid="stMetricDelta"] {
    font-weight: 600 !important;
}

/* Success message */
div[data-testid="stAlert"][kind="success"] {
    background-color: #E6F4F1 !important;
    border: 1px solid #A7D9CE !important;
    color: #115E59 !important;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 4. APPLICATION HEADER
# --------------------------------------------------

st.markdown("""
<div class="main-header">
    <div class="brand">ChurnSense</div>
    <div class="tagline">CUSTOMER CHURN INTELLIGENCE</div>
    <h1>Customer Churn Prediction</h1>
    <p>
        Identify customers at risk of leaving using
        an Artificial Neural Network.
    </p>
    <div class="tech-label">POWERED BY PYTORCH ANN</div>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 5. LOAD MODEL AND PREPROCESSOR
# --------------------------------------------------

model, preprocessor, threshold = load_assets()

if model is None:
    st.warning(
        "Model artifacts not found. "
        "Run python train.py first."
    )
    st.stop()


# --------------------------------------------------
# 6. CUSTOMER INPUT FORM
# --------------------------------------------------

st.markdown("## Customer Details")
st.caption(
    "Enter the customer's information below to assess churn risk."
)

with st.form("churn_form"):

    # PERSONAL INFORMATION

    st.markdown("### 1. Personal Information")
    st.caption("Basic details about the customer.")

    c1, c2 = st.columns(2)

    with c1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", [0, 1])

    with c2:
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])

    st.divider()

    # ACCOUNT INFORMATION

    st.markdown("### 2. Account Information")
    st.caption("Customer tenure and account preferences.")

    c1, c2 = st.columns(2)

    with c1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

    with c2:
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    st.divider()

    # SERVICE INFORMATION

    st.markdown("### 3. Service Information")
    st.caption("Telecom services subscribed by the customer.")

    c1, c2 = st.columns(2)

    with c1:
        phone = st.selectbox("Phone service", ["Yes", "No"])
        lines = st.selectbox(
            "Multiple lines",
            ["No", "Yes", "No phone service"]
        )
        internet = st.selectbox(
            "Internet service",
            ["DSL", "Fiber optic", "No"]
        )
        security = st.selectbox(
            "Online security",
            ["Yes", "No", "No internet service"]
        )
        backup = st.selectbox(
            "Online backup",
            ["Yes", "No", "No internet service"]
        )

    with c2:
        protection = st.selectbox(
            "Device protection",
            ["Yes", "No", "No internet service"]
        )
        support = st.selectbox(
            "Tech support",
            ["Yes", "No", "No internet service"]
        )
        tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )
        movies = st.selectbox(
            "Streaming movies",
            ["Yes", "No", "No internet service"]
        )

    st.divider()

    # BILLING INFORMATION

    st.markdown("### 4. Billing Information")
    st.caption("Monthly and total customer charges.")

    c1, c2 = st.columns(2)

    with c1:
        monthly = st.number_input(
            "Monthly charges", 0.0, 200.0, 70.0
        )

    with c2:
        total = st.number_input(
            "Total charges", 0.0, 10000.0, 840.0
        )

    submit = st.form_submit_button(
        "Predict Customer Churn",
        type="primary",
        use_container_width=True
    )


# --------------------------------------------------
# 7. PREDICTION LOGIC
# --------------------------------------------------

if submit:

    row = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": lines,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    try:
        features = preprocessor.transform(row).astype(np.float32)

        with torch.no_grad():
            probability = torch.sigmoid(
                model(torch.tensor(features))
            ).item()

        # DISPLAY RESULTS

        st.markdown('<div id="prediction-results"></div>',
                    unsafe_allow_html=True)

        st.markdown("---")

        with st.container(border=True):

            st.markdown("### Prediction Results")
            st.caption("ANN-based customer churn assessment")

            result1, result2 = st.columns(2)

            with result1:
                st.metric(
                    label="Churn Probability",
                    value=f"{probability:.1%}"
                )

            with result2:
                if probability >= threshold:
                    st.metric(
                        label="Customer Risk",
                        value="High Risk",
                        delta="Likely to churn",
                        delta_color="inverse"
                    )
                else:
                    st.metric(
                        label="Customer Risk",
                        value="Low Risk",
                        delta="Likely to stay",
                        delta_color="normal"
                    )

            st.divider()

            if probability >= threshold:
                st.error(
                    "This customer is predicted to have "
                    "a higher likelihood of churning."
                )
            else:
                st.success(
                    "This customer is predicted to have "
                    "a lower likelihood of churning."
                )

            st.caption(
                f"Decision threshold: {threshold:.2f} | "
                "Predictions are estimates, not guarantees."
            )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

    components.html(
        """
        <script>
            window.parent.document
                .getElementById("prediction-results")
                ?.scrollIntoView({
                    behavior: "smooth",
                    block: "start"
                });
        </script>
        """,
        height=0
    )
