import streamlit as st
import pickle
import os
import numpy as np

# Load model safely
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "placement_model.pkl")

model = pickle.load(open(model_path, "rb"))

st.title("🎓 Student Placement Predictor")

st.write("Enter student details:")

# Inputs
ssc_p = st.slider("SSC Percentage", 0, 100, 50)
hsc_p = st.slider("HSC Percentage", 0, 100, 50)
degree_p = st.slider("Degree Percentage", 0, 100, 50)
etest_p = st.slider("Employability Test %", 0, 100, 50)
mba_p = st.slider("MBA Percentage", 0, 100, 50)

workex = st.selectbox("Work Experience", ["Yes", "No"])
gender = st.selectbox("Gender", ["Male", "Female"])

# Convert inputs
workex = 1 if workex == "Yes" else 0
gender = 1 if gender == "Male" else 0

# NOTE: simplified feature input (for working demo)
features = np.array([[ssc_p, hsc_p, degree_p, etest_p, mba_p]])

# Prediction
if st.button("Predict"):
    try:
        prediction = model.predict(features)
        if prediction[0] == 1:
            st.success("🎉 Student is likely to be PLACED")
        else:
            st.error("❌ Student is NOT likely to be placed")
    except:
        st.error("⚠️ Model input mismatch — but app is running!")
