# src/train.py

import os
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
)

from sklearn.model_selection import GridSearchCV

from preprocess import preprocess_pipeline


# -----------------------------
# Custom Prediction (Threshold)
# -----------------------------
def predict_with_threshold(model, X, threshold=0.5):
    probs = model.predict_proba(X)[:, 1]
    return (probs >= threshold).astype(int), probs


# -----------------------------
# Evaluation Function
# -----------------------------
def evaluate_model(name, model, X_test, y_test, threshold=0.5):
    y_pred, y_prob = predict_with_threshold(model, X_test, threshold)

    metrics = {
        "Model": name,
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    }

    return metrics, y_pred


# -----------------------------
# Main Training Pipeline
# -----------------------------
def main():
    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline("Data\diabetes.csv")

    results = []

    # ------------------------
    # 1. Logistic Regression
    # ------------------------
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train, y_train)
    metrics, _ = evaluate_model("Logistic Regression", lr, X_test, y_test)
    results.append(metrics)

    # ------------------------
    # 2. Random Forest
    # ------------------------
    rf = RandomForestClassifier(class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)
    metrics, _ = evaluate_model("Random Forest", rf, X_test, y_test)
    results.append(metrics)

    # ------------------------
    # 3. SVM
    # ------------------------
    svm = SVC(probability=True, class_weight="balanced")
    svm.fit(X_train, y_train)
    metrics, _ = evaluate_model("SVM", svm, X_test, y_test)
    results.append(metrics)

    # ------------------------
    # Print Model Comparison
    # ------------------------
    results_df = pd.DataFrame(results)
    print("\nModel Comparison:\n")
    print(results_df)

    # ------------------------
    # Hyperparameter Tuning (Random Forest for Recall)
    # ------------------------
    print("\nTuning Random Forest for Recall...\n")

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5]
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=42),
        param_grid,
        cv=5,
        scoring="recall"
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    print("Best Parameters:", grid_search.best_params_)

    # ------------------------
    # Threshold Tuning
    # ------------------------
    print("\nThreshold Tuning Results:\n")

    threshold_results = []

    for t in [0.5, 0.4, 0.3]:
        metrics, y_pred = evaluate_model("Tuned RF", best_model, X_test, y_test, threshold=t)
        threshold_results.append(metrics)
        print(metrics)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix (Threshold={t}):\n{cm}\n")

    threshold_df = pd.DataFrame(threshold_results)

    # ------------------------
    # Select Best Threshold (Manual Logic)
    # ------------------------
    # You can manually inspect, but typically choose ~0.4
    best_threshold = 0.4

    print(f"\nSelected Threshold: {best_threshold}")

    # ------------------------
    # Save Model + Scaler
    # ------------------------
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = os.path.join(BASE_DIR, "model")

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(best_model, os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(best_threshold, os.path.join(MODEL_DIR, "threshold.pkl"))
    print("\nModel and scaler saved successfully.")


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    main()