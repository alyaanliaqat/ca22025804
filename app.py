import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import joblib

# Load data
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

# Load dataframe
df = Data_ml.copy()

# Sidebar filter
st.sidebar.header("Filters")
country = st.sidebar.selectbox("Select Country", ["Ireland", "France", "Germany"])
df_filtered = df[df["Geopolitical entity (reporting)"] == country]

# Title
st.title("National Energy Consumption Analytics")
st.markdown("**Comparative Energy Analysis and Machine Learning Insights**")

# Section 1: Data Distribution
st.subheader("Energy Consumption Distribution")
fig_dist = px.histogram(
    df_filtered,
    x="OBS_VALUE",
    nbins=40,
    title=f"{country} Energy Consumption Distribution",
    labels={"OBS_VALUE": "Energy Consumption"}
)
st.plotly_chart(fig_dist, use_container_width=True)

# Section 2: Time Series Trend (Aggregated by Year)
st.subheader("Energy Consumption Over Time")
df_yearly = df_filtered.groupby("TIME_PERIOD")["OBS_VALUE"].sum().reset_index()  # sum by year

fig_ts = px.line(
    df_yearly,
    x="TIME_PERIOD",
    y="OBS_VALUE",
    title=f"{country} Energy Consumption Trend (Yearly Total)",
    labels={"OBS_VALUE": "Energy Consumption", "TIME_PERIOD": "Year"}
)
st.plotly_chart(fig_ts, use_container_width=True)

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

fig_pred = px.scatter(
    x=y_test,
    y=y_pred,
    labels={"x": "Actual Values", "y": "Predicted Values"},
    title=f"{model_choice}: Actual vs Predicted"
)
fig_pred.add_shape(
    type="line",
    x0=min(y_test), y0=min(y_test),
    x1=max(y_test), y1=max(y_test),
    line=dict(dash="dash")
)
st.plotly_chart(fig_pred, use_container_width=True)
