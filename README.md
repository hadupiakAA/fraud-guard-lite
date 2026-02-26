# FraudGuard Lite

FraudGuard Lite is an end-to-end machine learning microservice for fraud detection.

The project demonstrates a complete ML workflow:
- data preprocessing
- model training and evaluation
- model serialization
- inference via REST API
- Docker-based deployment

---

## Tech Stack

- Python 3.11
- Scikit-learn
- Pandas & NumPy
- FastAPI
- Docker & Docker Compose

---

## Project Structure

```
fraudguard-lite/
│
├── scr/
│   ├── train.py        # model training script
│   ├── api.py          # FastAPI application
│   ├── predict.py      # prediction logic
│   └── _logistic.py    # model definition
│
├── artifacts/
│   ├── model.joblib    # trained model
│   └── metrics.json    # evaluation metrics
│
├── data/
│   └── creditcard.csv  # dataset (local)
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Model

- Algorithm: Logistic Regression  
- Dataset: Credit Card Fraud Detection  
- Features: 30 numerical features  
- Output: Fraud probability score  

The model predicts the probability of fraud and returns a decision based on a configurable threshold.

---

## API Endpoints

### GET /

Returns basic service information.

---

### GET /health

Returns service health status and confirms model availability.

Example response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "threshold": 0.2
}
```

---

### POST /predict

Predicts fraud probability for a transaction.

Request body (must contain exactly 30 float values):

```json
{
  "features": [0.1, -1.2, 0.3, ..., 0.0]
}
```

Example response:

```json
{
  "risk_score": 0.018241,
  "threshold": 0.2,
  "decision": "legit",
  "model": "logistic_regression_v1"
}
```

---

## Quick Start

### 1. Train the model locally

```bash
python scr/train.py
```

This generates:
- `artifacts/model.joblib`
- `artifacts/metrics.json`

---

### 2. Run the API with Docker

```bash
docker compose up --build
```

Swagger documentation will be available at:

```
http://localhost:8000/docs
```

---

## Architecture

```
Client
   ↓
FastAPI REST API
   ↓
Logistic Regression Model
   ↓
Fraud Probability Score
```

---

## What This Project Demonstrates

- End-to-end ML pipeline
- Separation of training and inference logic
- REST API design for ML models
- Docker-based deployment
- Reproducible environment setup
- Clean project structure

---

## Author

Hadupiak Anastasia
