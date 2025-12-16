import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model and scaler
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("credit_model.pkl")
        scaler = joblib.load("credit_scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_artifacts()

st.title("Credit Score Prediction")
st.write("Enter the details below to predict credit risk.")

if model is None or scaler is None:
    st.error("Model or scaler file not found. Please run 'train_model.py' first.")
    st.stop()

# Input fields based on the dataset features
monthly_income = st.number_input("Monthly Income", min_value=0.0, value=5000.0)
debt_ratio = st.number_input("Debt Ratio", min_value=0.0, value=0.3)
num_dependents = st.number_input("Number of Dependents", min_value=0, value=0)
num_90_days_late = st.number_input("Number of Times 90Days Late", min_value=0, value=0)
num_60_89_days_late = st.number_input("Number of Time 60-89 Days Past Due Not Worse", min_value=0, value=0)
num_30_59_days_late = st.number_input("Number of Time 30-59 Days Past Due Not Worse", min_value=0, value=0)

# Create input dataframe matching the feature engineering in training
input_data = {
    "MonthlyIncome": [monthly_income],
    "NumberOfDependents": [num_dependents],
    "NumberOfTimes90DaysLate": [num_90_days_late],
    "NumberOfTime30-59DaysPastDueNotWorse": [num_30_59_days_late],
    "NumberOfTime60-89DaysPastDueNotWorse": [num_60_89_days_late],
    "DebtRatio": [debt_ratio]
}
df = pd.DataFrame(input_data)

# Feature Engineering (Must match train_model.py)
df["IncomeToDebt"] = df["MonthlyIncome"] / (df["DebtRatio"] + 1)
df["LatePaymentScore"] = (
    df["NumberOfTimes90DaysLate"] * 3 +
    df["NumberOfTime60-89DaysPastDueNotWorse"] * 2 +
    df["NumberOfTime30-59DaysPastDueNotWorse"]
)
df["HighRisk"] = (df["LatePaymentScore"] >= 5).astype(int)

# Prepare columns for prediction (make sure order matches training X)
# Based on training code: X = df.drop("SeriousDlqin2yrs", axis=1)
# The columns in df after feature engineering in training (before dropping target):
# [RevolvingUtilizationOfUnsecuredLines, age, NumberOfTime30-59DaysPastDueNotWorse, DebtRatio, MonthlyIncome, NumberOfOpenCreditLinesAndLoans, NumberOfTimes90DaysLate, NumberOfRealEstateLoansOrLines, NumberOfTime60-89DaysPastDueNotWorse, NumberOfDependents, IncomeToDebt, LatePaymentScore, HighRisk]
# Wait, the user's original code drops "Unnamed: 0".
# It also has other columns I missed in the input above: RevolvingUtilizationOfUnsecuredLines, age, NumberOfOpenCreditLinesAndLoans, NumberOfRealEstateLoansOrLines.
# I need to add these inputs.

st.sidebar.header("Additional Information")
revolving_util = st.sidebar.number_input("Revolving Utilization of Unsecured Lines", min_value=0.0, value=0.0)
age = st.sidebar.number_input("Age", min_value=0, value=30)
num_open_credit_lines = st.sidebar.number_input("Number of Open Credit Lines and Loans", min_value=0, value=5)
num_real_estate_loans = st.sidebar.number_input("Number of Real Estate Loans or Lines", min_value=0, value=0)

df["RevolvingUtilizationOfUnsecuredLines"] = [revolving_util]
df["age"] = [age]
df["NumberOfOpenCreditLinesAndLoans"] = [num_open_credit_lines]
df["NumberRealEstateLoansOrLines"] = [num_real_estate_loans]

# Reorder columns to match training set implicitly.
# The training script drops "SeriousDlqin2yrs" and "Unnamed: 0".
# The order of columns in 'df' (after reads and drops) is important for the Scaler and Model (?)
# RandomForest handles column names if passed a dataframe, but Scaler is numpy array based usually.
# So I must ensure column order is EXACTLY the same as X_train.
# Original Columns in cs-training.csv:
# Unnamed: 0, SeriousDlqin2yrs, RevolvingUtilizationOfUnsecuredLines, age, NumberOfTime30-59DaysPastDueNotWorse, DebtRatio, MonthlyIncome, NumberOfOpenCreditLinesAndLoans, NumberOfTimes90DaysLate, NumberOfRealEstateLoansOrLines, NumberOfTime60-89DaysPastDueNotWorse, NumberOfDependents
# After cleaning in script:
# Drops "Unnamed: 0".
# Adds "IncomeToDebt", "LatePaymentScore", "HighRisk".
# Drops "SeriousDlqin2yrs".
# Resulting X columns order:
# RevolvingUtilizationOfUnsecuredLines, age, NumberOfTime30-59DaysPastDueNotWorse, DebtRatio, MonthlyIncome, NumberOfOpenCreditLinesAndLoans, NumberOfTimes90DaysLate, NumberOfRealEstateLoansOrLines, NumberOfTime60-89DaysPastDueNotWorse, NumberOfDependents, IncomeToDebt, LatePaymentScore, HighRisk

feature_order = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
    "IncomeToDebt",
    "LatePaymentScore",
    "HighRisk"
]

df_final = df[feature_order]

if st.button("Predict"):
    try:
        # Scale
        X_scaled = scaler.transform(df_final)
        
        # Predict
        # We use probability with a custom threshold of 0.35
        probability = model.predict_proba(X_scaled)[0][1]
        
        if probability > 0.35:
            st.error(f"Loan Rejected: High Credit Risk! (Probability: {probability:.2f} > 0.35)")
        else:
            st.success(f"Loan Approved: Low Credit Risk. (Probability: {probability:.2f} <= 0.35)")
            
    except Exception as e:
        st.error(f"Error during prediction: {e}")
