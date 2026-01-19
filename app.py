import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Page Configuration ---
st.set_page_config(
    page_title="Diabetes Detection System",
    page_icon="🩺",
    layout="centered"
)

# --- Load the Model ---
@st.cache_resource
def load_model():
    try:
        # Load the dictionary containing model and scaler
        data = joblib.load('diabetes_model_final.pkl')
        
        # specific handling if you saved just the model or a dict
        if isinstance(data, dict):
            return data['model'], data.get('scaler')
        else:
            return data, None
    except FileNotFoundError:
        st.error("Model file not found. Please run the notebook to generate 'diabetes_model_data.pkl'.")
        return None, None

model, scaler = load_model()

# --- App Title and Description ---
st.title("🩺 Diabetes Detection System")
st.markdown("""
    Enter the patient's diagnostic measures below to predict the likelihood of diabetes.
    This system uses a machine learning model trained on historical medical data.
""")

st.divider()

# --- Input Form ---
# We use columns to organize the inputs neatly
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1, value=0, help="Number of times pregnant")
    glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=100, help="Plasma glucose concentration (2 hours in an oral glucose tolerance test)")
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=140, value=70, help="Diastolic blood pressure (mm Hg)")
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20, help="Triceps skin fold thickness (mm)")

with col2:
    insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=79, help="2-Hour serum insulin (mu U/ml)")
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f", help="Body mass index (weight in kg/(height in m)^2)")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, format="%.3f", help="Diabetes pedigree function")
    age = st.number_input("Age", min_value=0, max_value=120, step=1, value=30, help="Age in years")

# --- Prediction Logic ---
if st.button("Analyze Result", type="primary", use_container_width=True):
    if model is not None:
        # Create a dataframe for the input to match training format
        input_data = pd.DataFrame([[
            pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age
        ]], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

        # Scale the data if a scaler was loaded
        if scaler:
            input_data_processed = scaler.transform(input_data)
        else:
            input_data_processed = input_data

        # Make prediction
        prediction = model.predict(input_data_processed)
        probability = model.predict_proba(input_data_processed)[0][1]

        st.divider()
        
        # Display Results
        if prediction[0] == 1:
            st.error(f"**Prediction: Diabetic**")
            st.warning(f"The model predicts a **{probability:.1%}** probability of diabetes based on the input parameters.")
            st.info("⚠️ Recommendation: Please consult a healthcare professional for further diagnosis.")
        else:
            st.success(f"**Prediction: Non-Diabetic**")
            st.markdown(f"The model predicts a **{probability:.1%}** probability of diabetes.")
            st.balloons()