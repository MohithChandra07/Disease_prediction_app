import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title="Disease Prediction App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #ffffff !important; }
    [data-testid="stAppViewContainer"] { background-color: #ffffff !important; }
    [data-testid="stMain"] { background-color: #ffffff !important; }
    .main .block-container { background-color: #ffffff !important; padding: 2rem 3rem; }

    [data-testid="stSidebar"] { background-color: #f8faff !important; border-right: 1px solid #e8f0fe; }
    [data-testid="stSidebar"] * { color: #1a1a2e !important; }

    #MainMenu, footer, header { visibility: hidden; }

    h1, h2, h3, p, label, div { color: #1a1a2e; }

    .app-header {
        background: #ffffff;
        border: 1px solid #e8f0fe;
        border-left: 5px solid #1a73e8;
        border-radius: 12px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 2px 8px rgba(26,115,232,0.08);
    }
    .app-header h1 {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a1a2e !important;
        margin: 0 0 0.3rem 0;
    }
    .app-header p {
        font-size: 0.95rem;
        color: #5f6368 !important;
        margin: 0;
    }

    .section-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #1a73e8 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 1.6rem 0 0.8rem;
        padding-bottom: 0.5rem;
        border-bottom: 1.5px solid #e8f0fe;
    }

    .stButton > button {
        background: #1a73e8 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.65rem 2rem !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 2px 8px rgba(26,115,232,0.25) !important;
    }
    .stButton > button:hover {
        background: #1557b0 !important;
        box-shadow: 0 4px 16px rgba(26,115,232,0.35) !important;
    }

    /* POSITIVE = green, NEGATIVE = red */
    .result-positive {
        background: #f6fef8;
        border: 1.5px solid #34a853;
        border-left: 5px solid #34a853;
        border-radius: 12px;
        padding: 1.4rem 1.8rem;
    }
    .result-negative {
        background: #fff8f6;
        border: 1.5px solid #ea4335;
        border-left: 5px solid #ea4335;
        border-radius: 12px;
        padding: 1.4rem 1.8rem;
    }
    .result-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
    }
    .result-label-pos { color: #34a853 !important; }
    .result-label-neg { color: #ea4335 !important; }
    .result-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #1a1a2e !important;
        margin: 0;
    }
    .result-sub {
        font-size: 0.88rem;
        color: #5f6368 !important;
        margin-top: 0.2rem;
    }

    .conf-card {
        background: #f8faff;
        border: 1px solid #e8f0fe;
        border-radius: 12px;
        padding: 1.4rem 1.8rem;
    }
    .conf-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #1a73e8 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
    }
    .conf-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e !important;
    }
    .bar-bg {
        background: #e8f0fe;
        border-radius: 999px;
        height: 8px;
        margin-top: 0.6rem;
        overflow: hidden;
    }
    .bar-fill-pos { height: 8px; border-radius: 999px; background: #34a853; }
    .bar-fill-neg { height: 8px; border-radius: 999px; background: #ea4335; }

    .stSlider label { color: #1a1a2e !important; font-size: 0.9rem !important; }
    .stSelectbox label { color: #1a1a2e !important; font-size: 0.9rem !important; }

    /* Light dropdowns */
    .stSelectbox > div > div {
        background-color: #f8faff !important;
        border: 1px solid #e8f0fe !important;
        border-radius: 8px !important;
        color: #1a1a2e !important;
    }
    .stSelectbox > div > div > div { color: #1a1a2e !important; }

    .disclaimer {
        background: #fef9e7;
        border: 1px solid #f9ca24;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        font-size: 0.78rem;
        color: #7d6608 !important;
        margin-top: 1rem;
        line-height: 1.5;
    }

    hr { border: none; border-top: 1px solid #e8f0fe; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_model(disease):
    model    = joblib.load(f'{BASE}/models/{disease}_model.pkl')
    scaler   = joblib.load(f'{BASE}/models/scaler_{disease}.pkl')
    features = joblib.load(f'{BASE}/models/features_{disease}.pkl')
    return model, scaler, features

def predict(model, scaler, features, user_input):
    df     = pd.DataFrame([user_input], columns=features)
    scaled = scaler.transform(df)
    pred   = model.predict(scaled)[0]
    proba  = model.predict_proba(scaled)[0][1]
    return pred, proba, scaled, features

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🩺 Disease Predictor")
    st.markdown("---")
    disease = st.selectbox(
        "Select Disease",
        ["Diabetes", "Heart Disease", "Liver Disease"]
    )
    st.markdown("---")
    st.markdown("**Models used:**")
    st.markdown("- Diabetes → XGBoost (AUC 0.801)")
    st.markdown("- Heart → Logistic Reg (AUC 0.895)")
    st.markdown("- Liver → Logistic Reg (AUC 0.852)")
    st.markdown("""
    <div class='disclaimer'>
    ⚠️ For educational purposes only. Not a substitute for professional medical advice.
    </div>
    """, unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────
icons = {"Diabetes": "💉", "Heart Disease": "🫀", "Liver Disease": "🤍"}
st.markdown(f"""
<div class='app-header'>
    <h1>{icons[disease]} {disease} Prediction</h1>
    <p>Enter patient clinical data and click <strong>Run Prediction</strong> to get an AI-powered risk assessment with explainability.</p>
</div>
""", unsafe_allow_html=True)

# ── Inputs ─────────────────────────────────────────────────
user_input = {}

if disease == "Diabetes":
    st.markdown("<p class='section-label'>Patient Clinical Data</p>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        user_input['Pregnancies']              = st.slider("Pregnancies", 0, 17, 3)
        user_input['Glucose']                  = st.slider("Glucose Level (mg/dL)", 0, 200, 120)
        user_input['BloodPressure']            = st.slider("Blood Pressure (mmHg)", 0, 122, 70)
        user_input['SkinThickness']            = st.slider("Skin Thickness (mm)", 0, 99, 20)
    with col2:
        user_input['Insulin']                  = st.slider("Insulin (μU/mL)", 0, 846, 79)
        user_input['BMI']                      = st.slider("BMI", 0.0, 67.1, 32.0)
        user_input['DiabetesPedigreeFunction'] = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
        user_input['Age']                      = st.slider("Age (years)", 21, 81, 33)
    model_key = 'diabetes'

elif disease == "Heart Disease":
    st.markdown("<p class='section-label'>Patient Clinical Data</p>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        user_input['age']      = st.slider("Age", 20, 80, 50)
        user_input['sex']      = st.selectbox("Sex", [0,1], format_func=lambda x: "Female" if x==0 else "Male")
        user_input['cp']       = st.slider("Chest Pain Type (0–3)", 0, 3, 1)
        user_input['trestbps'] = st.slider("Resting Blood Pressure", 90, 200, 130)
        user_input['chol']     = st.slider("Cholesterol (mg/dL)", 100, 600, 250)
        user_input['fbs']      = st.selectbox("Fasting Blood Sugar > 120", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
        user_input['restecg']  = st.slider("Resting ECG (0–2)", 0, 2, 1)
    with col2:
        user_input['thalach']  = st.slider("Max Heart Rate Achieved", 70, 210, 150)
        user_input['exang']    = st.selectbox("Exercise Induced Angina", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
        user_input['oldpeak']  = st.slider("ST Depression (oldpeak)", 0.0, 6.2, 1.0)
        user_input['slope']    = st.slider("Slope of ST Segment (0–2)", 0, 2, 1)
        user_input['ca']       = st.slider("Major Vessels Coloured (0–3)", 0, 3, 0)
        user_input['thal']     = st.slider("Thalassemia (0–3)", 0, 3, 2)
    model_key = 'heart'

else:
    st.markdown("<p class='section-label'>Patient Clinical Data</p>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        user_input['Age']                        = st.slider("Age", 4, 90, 35)
        user_input['Gender']                     = st.selectbox("Gender", [0,1], format_func=lambda x: "Female" if x==0 else "Male")
        user_input['Total_Bilirubin']            = st.slider("Total Bilirubin", 0.4, 75.0, 1.0)
        user_input['Direct_Bilirubin']           = st.slider("Direct Bilirubin", 0.1, 19.7, 0.3)
        user_input['Alkaline_Phosphotase']       = st.slider("Alkaline Phosphotase", 63, 2110, 200)
    with col2:
        user_input['Alamine_Aminotransferase']   = st.slider("Alamine Aminotransferase", 10, 2000, 35)
        user_input['Aspartate_Aminotransferase'] = st.slider("Aspartate Aminotransferase", 10, 4929, 40)
        user_input['Total_Protiens']             = st.slider("Total Proteins", 2.7, 9.6, 6.8)
        user_input['Albumin']                    = st.slider("Albumin", 0.9, 5.5, 3.5)
        user_input['Albumin_Globulin_Ratio']     = st.slider("Albumin / Globulin Ratio", 0.3, 2.8, 1.0)
    model_key = 'liver'

# ── Predict ────────────────────────────────────────────────
st.markdown("---")
if st.button("🔍 Run Prediction", use_container_width=True):
    with st.spinner("Analysing patient data..."):
        model, scaler, features = load_model(model_key)
        pred, proba, scaled, features = predict(model, scaler, features, user_input)

    st.markdown("---")
    col1, col2 = st.columns([3, 2])

    with col1:
        if pred == 1:
            st.markdown(f"""
            <div class='result-positive'>
                <p class='result-label result-label-pos'>Prediction Result</p>
                <p class='result-value'>POSITIVE</p>
                <p class='result-sub'>{disease} detected — elevated risk identified</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-negative'>
                <p class='result-label result-label-neg'>Prediction Result</p>
                <p class='result-value'>NEGATIVE</p>
                <p class='result-sub'>No {disease} detected — low risk profile</p>
            </div>""", unsafe_allow_html=True)

    with col2:
        # POSITIVE = green bar, NEGATIVE = red bar
        fill_class = "bar-fill-pos" if pred == 1 else "bar-fill-neg"
        st.markdown(f"""
        <div class='conf-card'>
            <p class='conf-label'>Confidence Score</p>
            <p class='conf-value'>{proba*100:.1f}%</p>
            <div class='bar-bg'>
                <div class='{fill_class}' style='width:{proba*100:.1f}%'></div>
            </div>
        </div>""", unsafe_allow_html=True)

    # SHAP
    st.markdown("---")
    st.markdown("<p class='section-label'>Feature Importance — Why this prediction?</p>", unsafe_allow_html=True)
    try:
        explainer   = shap.Explainer(model, pd.DataFrame(scaled, columns=features))
        shap_values = explainer(pd.DataFrame(scaled, columns=features))
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('white')
        shap.plots.bar(shap_values[0], show=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    except Exception:
        fi = pd.Series(dict(zip(features, np.abs(scaled[0])))).sort_values(ascending=True).tail(8)
        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        fi.plot(kind='barh', ax=ax, color='#1a73e8', edgecolor='white')
        ax.set_title('Top Contributing Features', fontweight='bold', color='#1a1a2e')
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(colors='#1a1a2e')
        st.pyplot(fig)
        plt.close()