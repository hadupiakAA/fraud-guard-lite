import os
import joblib
import pandas as pd


def load_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "artifacts", "model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Run train.py first.")
    return joblib.load(model_path)


def load_sample_row():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "creditcard.csv")
    df = pd.read_csv(data_path)

    X = df.drop("Class", axis=1)
    sample = X.iloc[[0]]  # одна транзакція
    return sample


if __name__ == "__main__":
    model = load_model()
    sample = load_sample_row()

    proba = model.predict_proba(sample)[0, 1]
    pred = int(proba >= 0.5)

    print("Sample prediction:")
    print("risk_score =", float(proba))
    print("label =", "fraud" if pred == 1 else "legit")