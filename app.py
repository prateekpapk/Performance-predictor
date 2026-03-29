import streamlit as st
import numpy as np
import joblib
import warnings
import pandas as pd
from Model.model_loader import load_model
from Utils.preprocess import prepare_input
warnings.filterwarnings("ignore")

model = load_model()
st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🎓 Student Performance Predictor</h1>", unsafe_allow_html=True)
st.markdown("### 📊 Enter student details")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.slider("📘 Study Hours per Day", 0.0, 12.0, 2.0)
    attendance = st.slider("📅 Attendance (%)", 0.0, 100.0, 80.0)
    sleep_hours = st.slider("😴 Sleep Hours", 0.0, 12.0, 7.0)

with col2:
    mental_health = st.slider("🧠 Mental Health Rating (1-10)", 1, 10, 5)
    part_time_job = st.radio("💼 Part-Time Job", ["No", "Yes"])



if st.button("🚀 Predict Exam Score"):

    input_data = prepare_input(
        study_hours,
        attendance,
        mental_health,
        sleep_hours,
        part_time_job
    )

    raw_prediction = model.predict(input_data)[0]


    prediction = max(0, min(100, raw_prediction))

    st.success(f"🎯 Predicted Exam Score : {prediction:.2f}")

    st.subheader("📊 Input Feature Visualization")
    df = pd.DataFrame({
        "Feature": ["Study", "Attendance", "Mental Health", "Sleep"],
        "Value": [study_hours, attendance/10, mental_health, sleep_hours]
    })
    st.bar_chart(df.set_index("Feature"))

    st.subheader("📈 Score Level")
    st.progress(int(prediction))
  

st.markdown("---")
st.info("""
📌 Model Details:
- Algorithm: Best model selected using GridSearchCV  
- Compared Models: Linear Regression, Decision Tree, Random Forest  
- Metrics Used: RMSE, R² Score  
- Type: Supervised Learning  
""")