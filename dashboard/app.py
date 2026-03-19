import streamlit as st
import pickle
import os
import pandas as pd

# Load model + columns
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "placement_model.pkl")

model, columns = pickle.load(open(model_path, "rb"))

st.title("🎓 Student Placement Predictor")

st.write("Enter student details:")

# Inputs
gender = st.selectbox("Gender", ["M", "F"])
ssc_p = st.slider("SSC %", 0, 100, 50)
ssc_b = st.selectbox("SSC Board", ["Central", "Others"])

hsc_p = st.slider("HSC %", 0, 100, 50)
hsc_b = st.selectbox("HSC Board", ["Central", "Others"])
hsc_s = st.selectbox("HSC Stream", ["Commerce", "Science", "Arts"])

degree_p = st.slider("Degree %", 0, 100, 50)
degree_t = st.selectbox("Degree Type", ["Sci&Tech", "Comm&Mgmt", "Others"])

workex = st.selectbox("Work Experience", ["Yes", "No"])

etest_p = st.slider("Employability Test %", 0, 100, 50)

specialisation = st.selectbox("Specialisation", ["Mkt&HR", "Mkt&Fin"])

mba_p = st.slider("MBA %", 0, 100, 50)

# Create input dictionary
input_dict = {
    "gender": gender,
    "ssc_p": ssc_p,
    "ssc_b": ssc_b,
    "hsc_p": hsc_p,
    "hsc_b": hsc_b,
    "hsc_s": hsc_s,
    "degree_p": degree_p,
    "degree_t": degree_t,
    "workex": workex,
    "etest_p": etest_p,
    "specialisation": specialisation,
    "mba_p": mba_p
}

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Apply same encoding
input_df = pd.get_dummies(input_df)

# Match training columns
input_df = input_df.reindex(columns=columns, fill_value=0)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.success("🎉 Student is likely to be PLACED")
    else:
        st.error("❌ Student is NOT likely to be placed")
