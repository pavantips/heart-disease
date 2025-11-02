import streamlit as st
import joblib
import pandas as pd

# TODO: Load your saved model (hint: joblib.load())
# TODO: Load your saved feature columns

model = joblib.load('heart_disease_model.pkl')
feature_columns = joblib.load('feature_columns.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🏥 Heart Disease Prediction App")
st.write("Enter patient information to predict diabetes risk")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", ["Male", "Female"])  # ← User sees this
    chest_pain = st.selectbox("Chest Pain Type", ["Asymptomatic", "Atypical Angina", "Non-Anginal Pain", "Typical Angina"])
    resting_bp = st.number_input("Resting Blood Pressure", min_value=0, max_value=200, value=120)
    cholesterol = st.number_input("Cholesterol", min_value=0, max_value=600, value=200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])

with col2:
    resting_ecg = st.selectbox("Resting ECG", ["LVH", "Normal", "ST"])
    max_hr = st.number_input("Max Heart Rate", min_value=0, max_value=220, value=150)
    exercise_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    st_slope = st.selectbox("ST Slope", ["Down", "Flat", "Up"])

if st.button("Predict Heart Disease Risk"):
    # MAP user-friendly inputs to encoded values
    sex_map = {'Male': 1, 'Female': 0}
    chest_pain_map = {'Asymptomatic': 1000, 'Atypical Angina': 1001, 'Non-Anginal Pain': 1002, 'Typical Angina': 1003}
    resting_ecg_map = {'LVH': 10, 'Normal': 11, 'ST': 12}
    exercise_angina_map = {'No': 100, 'Yes': 101}
    st_slope_map = {'Down': 2000, 'Flat': 2001, 'Up': 2002}
    fasting_bs_map = {'No': 0, 'Yes': 1}
    
    # Create dataframe with ENCODED values
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex_map[sex]],  # ← Convert Male/Female to 1/0
        'ChestPainType': [chest_pain_map[chest_pain]],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs_map[fasting_bs]],
        'RestingECG': [resting_ecg_map[resting_ecg]],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina_map[exercise_angina]],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope_map[st_slope]]
    })
    
    # Scale the data (IMPORTANT!)
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    
    # Display result
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
        st.write("Please consult a cardiologist immediately.")
    else:
        st.success("✅ Low Risk of Heart Disease")
        st.write("Continue maintaining a healthy lifestyle!")
    
