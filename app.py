#!/usr/bin/env python
# coding: utf-8

# In[9]:


import streamlit as st
import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load your pre-trained model
model = load_model('hospitalmodel.h5')

# Description and Title
st.title("Hospital Patient Readmission Risk Prediction and Bed Occupancy Forecasting")
st.markdown("""
Welcome to the **Healthcare Resource Management Platform**! This application combines **machine learning** and **time series forecasting** to enable predictive healthcare management. 

### Key Features:
- **Predict Patient Readmission Risk**: Enter specific patient details to assess their likelihood of readmission.
- **Forecast Bed Occupancy**: Use ARIMA forecasting to predict future bed occupancy levels, allowing for effective resource allocation.
""")

# Sidebar for ARIMA Parameters
st.sidebar.header("ARIMA Model Parameters")
p = st.sidebar.number_input("AR (p):", min_value=0, value=1, step=1)
d = st.sidebar.number_input("I (d):", min_value=0, value=1, step=1)
q = st.sidebar.number_input("MA (q):", min_value=0, value=1, step=1)
forecast_days = st.sidebar.slider("Days to Forecast:", min_value=1, max_value=60, value=30)

# Patient Details for Readmission Prediction
st.subheader("Enter Patient Details for Readmission Prediction")
patient_details = {
    "admission_type": st.selectbox("Admission Type", ["EMERGENCY", "URGENT", "ELECTIVE"]),
    "admission_location": st.selectbox("Admission Location", ["CLINIC REFERRAL", "EMERGENCY ROOM", "TRANSFER"]),
    "discharge_location": st.selectbox("Discharge Location", ["HOME", "SNF", "REHAB"]),
    "insurance": st.selectbox("Insurance", ["Medicare", "Private", "Medicaid", "Self Pay"]),
    "marital_status": st.selectbox("Marital Status", ["SINGLE", "MARRIED", "DIVORCED", "WIDOWED"]),
    "ethnicity": st.selectbox("Ethnicity", ["WHITE", "BLACK", "HISPANIC", "ASIAN"]),
    "diagnosis": st.selectbox("Diagnosis", ["Diagnosis A", "Diagnosis B", "Diagnosis C"]),
    "has_chartevents_data": st.selectbox("Has Chartevents Data", [0, 1]),
    "edregtime_numeric": st.number_input("ED Registration Time (numeric)", value=0),
    "edouttime_numeric": st.number_input("ED Out Time (numeric)", value=0),
    "los": st.number_input("Length of Stay (days)", value=1)
}

# Placeholder: Prediction function for patient readmission risk
def predict_readmission(patient_details):
    # Placeholder preprocessing and prediction code (replace with actual code)
    return np.random.choice([0, 1])

if st.button("Predict Readmission Risk"):
    readmission_risk = predict_readmission(patient_details)
    st.write(f"Predicted Readmission Risk: {'High' if readmission_risk == 1 else 'Low'}")

# Bed Occupancy Forecasting Section
st.subheader("Forecast Bed Occupancy")

# Placeholder occupancy time series
dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
occupancy_data = np.random.poisson(lam=20, size=100)
occupancy_series = pd.Series(occupancy_data, index=dates)

try:
    arima_model = ARIMA(occupancy_series, order=(p, d, q))
    arima_fit = arima_model.fit()
    forecast = arima_fit.forecast(steps=forecast_days)

    # Display forecast with custom aesthetics
    st.markdown("### Forecasted Bed Occupancy")
    st.line_chart(forecast)
    st.write(f"Forecast for the next {forecast_days} days is shown above.")
except Exception as e:
    st.error(f"An error occurred during ARIMA forecasting: {e}")

# Project Overview and Additional Insights
st.markdown("""
### Project Overview
- **Patient-Centric Approach**: Evaluate individual readmission risks for better follow-up care.
- **Proactive Resource Allocation**: Forecast bed demand to aid in resource management.
- **User-Friendly and Dynamic**: Easily adaptable to changing data, with interactive settings for quick, accurate results.

By integrating these features, hospitals can enhance patient outcomes and operational efficiency, staying prepared for varying levels of demand.
""")


# In[ ]:




