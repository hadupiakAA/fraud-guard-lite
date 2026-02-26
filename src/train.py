import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def load_data() -> pd.DataFrame:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "creditcard.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at: {data_path}\n"
            "Make sure you have data/creditcard.csv in the project root."
        )

    return pd.read_csv(data_path)


def prepare_data(df: pd.DataFrame):
    if "Class" not in df.columns:
        raise ValueError("Column 'Class' not found. Dataset format is unexpected.")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def train_model(X_train, y_train):
    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        ))
    ])
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, threshold: float = 0.2) -> dict:
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)

    print("\n=== Metrics ===")
    print("ROC-AUC:", roc_auc)
    print("PR-AUC:", pr_auc)
    print("F1:", f1)
    print(f"\n=== Classification report (threshold={threshold:.2f}) ===")
    print(classification_report(y_test, y_pred))

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "f1": float(f1),
        "threshold": float(threshold)
    }


def save_artifacts(model, metrics: dict):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifacts_dir = os.path.join(base_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    model_path = os.path.join(artifacts_dir, "model.joblib")
    metrics_path = os.path.join(artifacts_dir, "metrics.json")

    joblib.dump(model, model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved artifacts:")
    print("-", model_path)
    print("-", metrics_path)


if __name__ == "__main__":
    print("Loading data...")
    df = load_data()

    print("Preparing data...")
    X_train, X_test, y_train, y_test = prepare_data(df)
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("Fraud rate in train:", float(np.mean(y_train)))
    print("Fraud rate in test:", float(np.mean(y_test)))

    print("\nTraining model (LogisticRegression)...")
    model = train_model(X_train, y_train)

    print("Evaluating...")
    metrics = evaluate_model(model, X_test, y_test, threshold=0.5)

    save_artifacts(model, metrics)