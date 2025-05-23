import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import joblib

st.title("Predictive Maintenance Dashboard")

# Load data
data = pd.read_csv('C:/Users/KIIT/Downloads/turbine_sensor_data.csv', parse_dates=['timestamp'])

# Load model and scaler
model = load_model('C:/Users/KIIT/PycharmProjects/Predictive Maintenance Dashboard/autoencoder_model.h5', compile=False)
scaler = joblib.load("C:/Users/KIIT/PycharmProjects/Predictive Maintenance Dashboard/scaler.pkl")

# Preprocess data
features = data[['temperature', 'vibration', 'pressure']]
scaled = scaler.transform(features)
recon = model.predict(scaled)
mse = np.mean(np.power(scaled - recon, 2), axis=1)

threshold = np.percentile(mse, 95)
data['anomaly'] = mse > threshold
data['mse'] = mse

# Show line chart and anomalies
st.line_chart(data[['timestamp', 'mse']].set_index('timestamp'))
st.dataframe(data[data['anomaly'] == True])

# --- New Feature: Timestamp Checker ---
st.subheader("Check Maintenance Status by Timestamp")

user_input = st.text_input("Enter a timestamp (YYYY-MM-DD HH:MM:SS):")
if st.button("Check Status"):
    try:
        user_time = pd.to_datetime(user_input)

        # Find the closest timestamp instead of exact match
        closest_time = data.iloc[(data['timestamp'] - user_time).abs().argsort()[:1]]

        if not closest_time.empty:
            if closest_time['anomaly'].values[0]:
                st.error(f"⚠️ Maintenance Required at {closest_time['timestamp'].values[0]}")
            else:
                st.success(f"✅ Normal Operation at {closest_time['timestamp'].values[0]}")
        else:
            st.warning("⏰ Timestamp not found in the dataset.")
    except Exception as e:
        st.warning("Invalid timestamp format. Please enter in YYYY-MM-DD HH:MM:SS format.")
