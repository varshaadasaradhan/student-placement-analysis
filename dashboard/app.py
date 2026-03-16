import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("../placement_model.pkl", "rb"))

st.title("Student Placement Prediction Dashboard")

st.write("Enter student academic details to predict placement status.")

# User Inputs
ssc_p = st.number_input("10th Percentage (SSC)")
hsc_p = st.number_input("12th Percentage (HSC)")
degree_p = st.number_input("Degree Percentage")
etest_p = st.number_input("Employability Test Score")
mba_p = st.number_input("MBA Percentage")

# Simple inputs for categorical values
gender = st.selectbox("Gender", ["Male", "Female"])
workex = st.selectbox("Work Experience", ["Yes", "No"])

# Convert inputs to numeric
gender = 1 if gender == "Male" else 0
workex = 1 if workex == "Yes" else 0

if st.button("Predict Placement"):

    features = np.array([[gender, ssc_p, 0, hsc_p, 0, 0,
                          degree_p, 0, workex, etest_p, 0, mba_p]])

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.success("The student is likely to be PLACED")
    else:
        st.error("The student is likely NOT to be placed")
