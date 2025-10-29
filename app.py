import streamlit as st
import joblib
import pandas as pd

# TODO: Load your saved model (hint: joblib.load())
# TODO: Load your saved feature columns

model = joblib.load('diabetes_test.pkl')
feature_columns = joblib.load('feature_columns.pkl')

st.title("🏥 Diabetes Prediction App")
st.write("Enter patient information to predict diabetes risk")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=120)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=80)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    age = st.number_input("Age", min_value=0, max_value=120, value=30)

if st.button("Predict Diabetes Risk"):
    # Create dataframe with EXACT column names from training
    input_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree],
        'Age': [age]
    })
    
    # Make prediction
    prediction = model.predict(input_data)
    prediction_value = prediction[0]

    # Display result
    if prediction_value == 1:
        st.error("⚠️ Diabetes Detected - High Risk")
        st.write("Please consult with a healthcare professional.")
    else:
        st.success("✅ No Diabetes Detected - Low Risk")
        st.write("Maintain a healthy lifestyle!")
    
