import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

Data_ml = joblib.load("data_ml.pkl")
y_test = joblib.load("y_test.pkl")
rf_predictions = joblib.load("rf_pred.pkl")
xgb_predictions = joblib.load("xgb_pred.pkl")

lr_r2 = joblib.load("lr_r2.pkl")
rf_r2 = joblib.load("rf_r2.pkl")
xgb_r2 = joblib.load("xgb_r2.pkl")

lr_rmse = joblib.load("lr_rmse.pkl")
rf_rmse = joblib.load("rf_rmse.pkl")
xgb_rmse = joblib.load("xgb_rmse.pkl")

st.set_page_config(page_title="Energy Analytics Dashboard", layout="wide")

# Load data
df = Data_ml.copy()

# Sidebar filters
st.sidebar.header("Filters")
country = st.sidebar.selectbox("Select Country", ["Ireland", "France", "Germany"])
df_filtered = df[df["Geopolitical entity (reporting)"] == country]

# Title
st.title("National Energy Consumption Analytics")
st.markdown("**Comparative Energy Analysis and Machine Learning Insights**")

# Section 1: Data Distribution
st.subheader("Energy Consumption Distribution")
fig, ax = plt.subplots()
ax.hist(df_filtered["OBS_VALUE"], bins=40, color='skyblue', edgecolor='black')
ax.set_title(f"{country} Energy Consumption Distribution")
ax.set_xlabel("Energy Consumption")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# Section 2: Time Series Trend
st.subheader("Energy Consumption Over Time")
fig, ax = plt.subplots()
ax.plot(df_filtered["TIME_PERIOD"], df_filtered["OBS_VALUE"], marker='o', linestyle='-')
ax.set_title(f"{country} Energy Consumption Trend")
ax.set_xlabel("Year")
ax.set_ylabel("Energy Consumption")
plt.xticks(rotation=45)
st.pyplot(fig)

# Section 3: Model Performance
st.subheader("Machine Learning Model Performance")
performance_df = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest", "XGBoost"],
    "R² Score": [lr_r2, rf_r2, xgb_r2],
    "RMSE": [lr_rmse, rf_rmse, xgb_rmse]
})
st.dataframe(performance_df, use_container_width=True)

# Section 4: Actual vs Predicted
st.subheader("Actual vs Predicted Energy Consumption")
model_choice = st.selectbox("Select Model", ["Random Forest", "XGBoost"])
y_pred = rf_predictions if model_choice == "Random Forest" else xgb_predictions

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, color='blue', alpha=0.6)
ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
ax.set_title(f"{model_choice}: Actual vs Predicted")
ax.set_xlabel("Actual Values")
ax.set_ylabel("Predicted Values")
st.pyplot(fig)
