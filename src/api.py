from __future__ import annotations

import os
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ---------- Config ----------
BASE_DIR = Path(__file__).resolve().parent.parent  # project root
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")
MODEL_ABS_PATH = (BASE_DIR / MODEL_PATH).resolve()

THRESHOLD = float(os.getenv("THRESHOLD", "0.2"))

app = FastAPI(title="FraudGuard Lite API", version="0.1.0")


# ---------- Load model ----------
def load_model():
    if not MODEL_ABS_PATH.exists():
        raise RuntimeError(
            f"Model not found: {MODEL_ABS_PATH}\n"
            f"Run training first: python scr/train.py"
        )
    return joblib.load(MODEL_ABS_PATH)


model = load_model()


# ---------- Schemas ----------
class Transaction(BaseModel):
    features: List[float] = Field(..., description="Exactly 30 numeric features")


# ---------- Routes ----------
@app.get("/")
def root():
    return {"message": "FraudGuard Lite API is running"}


@app.get("/health")
def health():
    # простий healthcheck: модель завантажилась + threshold читається
    return {
        "status": "ok",
        "model_loaded": True,
        "model_path": str(MODEL_ABS_PATH),
        "threshold": THRESHOLD,
    }


@app.post("/predict")
def predict(transaction: Transaction):

    df = pd.DataFrame([transaction.features])

    try:
        risk_score = float(model.predict_proba(df)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    decision = "fraud" if risk_score >= THRESHOLD else "legit"

    return {
        "risk_score": round(risk_score, 6),
        "threshold": THRESHOLD,
        "decision": decision,
        "model": "logistic_regression_v1",
    }