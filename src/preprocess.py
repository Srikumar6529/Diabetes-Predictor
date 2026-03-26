# src/preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Columns with invalid zero values
INVALID_ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_data(path):
    df = pd.read_csv(path)
    return df


def handle_missing_values(df):
    """
    Replace biologically invalid zero values with median.
    """
    for col in INVALID_ZERO_COLS:
        df[col] = df[col].replace(0, df[col].median())
    return df


def split_data(df):
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    return train_test_split(X, y, test_size=0.2, random_state=42)


def scale_data(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def preprocess_pipeline(path):
    df = load_data(path)
    df = handle_missing_values(df)

    X_train, X_test, y_train, y_test = split_data(df)
    X_train, X_test, scaler = scale_data(X_train, X_test)

    return X_train, X_test, y_train, y_test, scaler