import pandas as pd
import numpy as np
import kagglehub
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

def train():
    print("Downloading dataset...")
    path = kagglehub.dataset_download("lihxlhx/give-me-some-credit")
    print("Path to dataset files:", path)
    
    # Check if file exists in the downloaded path
    csv_path = os.path.join(path, "cs-training.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print("Loading data...")
    df = pd.read_csv(csv_path)
    df = df.drop("Unnamed: 0", axis=1)
    
    # Handle missing values
    df["MonthlyIncome"].fillna(df["MonthlyIncome"].median(), inplace=True)
    df["NumberOfDependents"].fillna(df["NumberOfDependents"].median(), inplace=True)
    
    # Clip outliers
    df["NumberOfTimes90DaysLate"] = df["NumberOfTimes90DaysLate"].clip(0, 10)
    df["NumberOfTime30-59DaysPastDueNotWorse"] = df["NumberOfTime30-59DaysPastDueNotWorse"].clip(0, 10)
    df["NumberOfTime60-89DaysPastDueNotWorse"] = df["NumberOfTime60-89DaysPastDueNotWorse"].clip(0, 10)

    # Feature Engineering
    df["IncomeToDebt"] = df["MonthlyIncome"] / (df["DebtRatio"] + 1)
    df["LatePaymentScore"] = (
        df["NumberOfTimes90DaysLate"] * 3 +
        df["NumberOfTime60-89DaysPastDueNotWorse"] * 2 +
        df["NumberOfTime30-59DaysPastDueNotWorse"]
    )
    df["HighRisk"] = (df["LatePaymentScore"] >= 5).astype(int)

    X = df.drop("SeriousDlqin2yrs", axis=1)
    y = df["SeriousDlqin2yrs"]

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training Random Forest Classifier...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    )
    rf.fit(X_train_scaled, y_train)

    # Evaluation
    pred = rf.predict(X_test_scaled)
    prob = rf.predict_proba(X_test_scaled)[:, 1]

    print("\nModel Evaluation:")
    print(classification_report(y_test, pred))
    print("ROC-AUC:", roc_auc_score(y_test, prob))

    # Save artifacts
    joblib.dump(rf, "credit_model.pkl")
    joblib.dump(scaler, "credit_scaler.pkl")
    print("\nSaved credit_model.pkl and credit_scaler.pkl")

if __name__ == "__main__":
    train()