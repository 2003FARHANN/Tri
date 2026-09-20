# 🩺 TriNexus-VitaGuard-AI — Backend API

This directory contains the backend service for **TriNexus | VitaGuard AI**, an Explainable Artificial Intelligence (XAI) clinical decision support system. The backend is built with **FastAPI** to serve real-time predictions for diabetes risk and cardiovascular disease risk, provide SHAP model explainability data, and persist clinical prediction records.

---

## 📌 Project Overview

The backend acts as the bridge between trained machine learning models, database storage, and the frontend web dashboard.

### Core Responsibilities:
* **Machine Learning Inference**: Preprocesses incoming clinical metrics and runs inference via trained XGBoost and Random Forest models.
* **Explainable AI (XAI)**: Serves global SHAP (SHapley Additive exPlanations) feature importance scores so clinicians can understand feature contributions.
* **Data Validation**: Strict request/response payload validation using Pydantic schemas.
* **Persistence**: Automatically logs patient queries, risk classifications, and probability percentages into a SQLite database using SQLAlchemy.

---

## ⚙️ Prerequisites

Before setting up the backend locally, ensure you have the following installed:

* **Python**: Version `3.10` or higher (Python `3.11`–`3.14` supported)
* **pip**: Python package manager
* **virtualenv / venv**: Standard Python virtual environment module
* **macOS Users (Apple Silicon)**: If running on macOS, install the OpenMP runtime for XGBoost support:
  ```bash
  brew install libomp
  ```

---

## 🚀 Installation Steps

Follow these steps to set up and run the backend environment locally:

### 1. Navigate to the Backend Directory
From the root of the project:
```bash
cd backend
```

### 2. Create & Activate Virtual Environment

* **On Windows:**
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```

* **On macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies
Install all required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## ⚡ Running the Server

Start the local development server using **Uvicorn**:

```bash
uvicorn main:app --reload
```

Once running, the API server will be available at:
* **Base URL**: `http://127.0.0.1:8000`
* **Health Check**: `http://127.0.0.1:8000/`

### 📖 Interactive API Documentation
FastAPI provides automatic interactive documentation out of the box:
* **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 📡 API Endpoints Summary

| Method | Endpoint | Description | Request Body | Response |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/` | Root verification endpoint | None | `{"message": "..."}` |
| `POST` | `/api/predict` | Diabetes risk prediction | 8 clinical features (Glucose, BMI, Age, etc.) | Prediction (`0`/`1`), Risk Level, Probability % |
| `GET` | `/api/shap-importance` | SHAP feature importance rankings | None | Array of `{feature, importance}` |
| `POST` | `/api/predict/cardio` | Cardiovascular risk prediction | 13 cardiac features (`chol`, `thalach`, etc.) | Prediction (`0`/`1`), Risk Level, Probability % |

---

## 🗄️ Database

* **Database Engine**: SQLite
* **Database File**: `vitaguard.db` (auto-created on application startup)
* **Tables**:
  * `prediction_records`: Historical records of diabetes risk assessments.
  * `cardio_predictions`: Historical records of cardiovascular risk assessments.

---

## 🧪 Running Automated Tests

To run the API test suite using pytest:

```bash
# From the project root or backend directory:
pytest ../tests/test_api.py -v
```
*(Or `./venv/bin/pytest ../tests/test_api.py -v` if in the root with an activated virtual environment)*

---

## 📂 Backend Directory Structure

```
backend/
│
├── app/
│   ├── database/
│   │   └── database.py        # SQLAlchemy engine and session configuration
│   ├── models/
│   │   ├── models.py          # SQLAlchemy ORM database models
│   │   └── schemas.py         # Pydantic schemas for request/response validation
│   ├── routes/
│   │   └── prediction.py      # FastAPI route handlers (/predict, /shap-importance)
│   └── services/
│       ├── model_service.py   # Model artifact loading and prediction logic
│       └── shap_service.py    # Service serving SHAP explainability data
│
├── notebooks/
│   └── cardio_training.ipynb  # Prototyping notebook for cardiovascular model
│
├── main.py                    # FastAPI application initialization & CORS config
├── requirements.txt           # Python dependency specifications
├── vitaguard.db               # SQLite database file
└── README.md                  # Backend setup and developer documentation
```
