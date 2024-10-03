import streamlit as st
import numpy as np
import joblib

# Load the trained model (replace 'heart_disease_model.pkl' with your model file)
model = joblib.load(open('model.pkl', 'rb'))

# Function to make predictions
def predict_heart_disease(features):
    # Reshape input if necessary
 features = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
 prediction = model.predict(features)
 return prediction

# Streamlit app
st.title("Heart Disease Prediction")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=25)
sex = st.selectbox("Sex (1 = male; 0 = female)", [0, 1])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (in mm Hg)", min_value=50, max_value=200, value=120)
chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)", [0, 1])
restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina (1 = yes; 0 = no)", [0, 1])
oldpeak = st.number_input("Oldpeak (ST depression induced by exercise)", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope of the Peak Exercise ST Segment (0-2)", [0, 1, 2])
ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy (0-3)", min_value=0, max_value=3, value=0)
thal = st.selectbox("Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)", [1, 2, 3])

# Prediction button
if st.button("Predict"):
    features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    prediction = predict_heart_disease(features)

    if prediction == 1:
        st.write("The model predicts that you are at risk of heart disease.")
    else:
        st.write("The model predicts that you are not at risk of heart disease.")
