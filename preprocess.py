import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

def load_and_clean(filepath="data/Telco-Customer-Churn.csv"):
    df = pd.read_csv(filepath)

    # Fix TotalCharges (has spaces, should be numeric)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Drop customerID (not useful)
    df.drop(columns=["customerID"], inplace=True)

    # Binary encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def feature_engineer(df):
    # New features
    df["tenure_years"] = df["tenure"] / 12
    df["charges_per_month_ratio"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["has_streaming"] = (
        (df["StreamingTV"] == "Yes") | (df["StreamingMovies"] == "Yes")
    ).astype(int)
    df["num_services"] = (
        (df["PhoneService"] == "Yes").astype(int) +
        (df["MultipleLines"] == "Yes").astype(int) +
        (df["InternetService"] != "No").astype(int) +
        (df["OnlineSecurity"] == "Yes").astype(int) +
        (df["OnlineBackup"] == "Yes").astype(int) +
        (df["DeviceProtection"] == "Yes").astype(int) +
        (df["TechSupport"] == "Yes").astype(int) +
        (df["StreamingTV"] == "Yes").astype(int) +
        (df["StreamingMovies"] == "Yes").astype(int)
    )
    return df


def encode_and_scale(df):
    df = df.copy()
    label_cols = df.select_dtypes(include=["object"]).columns.tolist()

    encoders = {}
    for col in label_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(encoders, "models/encoders.pkl")
    joblib.dump(list(X.columns), "models/feature_names.pkl")

    return X_scaled, y


def get_data_splits(filepath="data/Telco-Customer-Churn.csv"):
    df = load_and_clean(filepath)
    df = feature_engineer(df)
    X, y = encode_and_scale(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✅ Data ready: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    print(f"   Churn rate: {y.mean():.1%}")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    get_data_splits()