import os
# Prevent numpy/sklearn multithreading deadlock on Windows
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import joblib

# Set base paths for model directories
# Set base paths for model directories
# This points to the "backend" directory
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# This points to the root directory (TriNexus-VitaGuard-AI)
ROOT_DIR = os.path.dirname(BACKEND_DIR)

DIABETES_MODEL_DIR = os.path.join(ROOT_DIR, "ML_Model", "models")
CARDIO_MODEL_DIR = os.path.join(BACKEND_DIR, "app", "ml_models")
# ----------------- 1. DIABETES MODEL ARTIFACTS -----------------
DIABETES_MODEL_PATH = os.path.join(DIABETES_MODEL_DIR, "best_model.pkl")
DIABETES_SCALER_PATH = os.path.join(DIABETES_MODEL_DIR, "scaler.pkl")
DIABETES_IMPUTER_PATH = os.path.join(DIABETES_MODEL_DIR, "imputer.pkl")

try:
    diabetes_model = joblib.load(DIABETES_MODEL_PATH)
    diabetes_scaler = joblib.load(DIABETES_SCALER_PATH)
    diabetes_imputer = joblib.load(DIABETES_IMPUTER_PATH)
except Exception as e:
    print(f"⚠️ ERROR LOADING DIABETES MODELS: {e}")
    diabetes_model = None
    diabetes_scaler = None
    diabetes_imputer = None

DIABETES_FEATURE_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", 
    "SkinThickness", "Insulin", "BMI", 
    "DiabetesPedigreeFunction", "Age"
]

def process_and_predict(input_data: dict) -> dict:
    """
    Preprocess raw user input data and predict diabetes risk.
    """
    if diabetes_model is None or diabetes_scaler is None or diabetes_imputer is None:
        raise RuntimeError("Diabetes ML model artifacts are not loaded properly.")

    df = pd.DataFrame([input_data], columns=DIABETES_FEATURE_NAMES)

    # 1. Biological zero validation
    zero_invalid_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_invalid_cols:
        df[col] = df[col].replace(0, pd.NA)

    # 2. Imputation
    df_imputed = pd.DataFrame(diabetes_imputer.transform(df), columns=DIABETES_FEATURE_NAMES)

    # 3. Feature Scaling
    df_scaled = pd.DataFrame(diabetes_scaler.transform(df_imputed), columns=DIABETES_FEATURE_NAMES)

    # 4. Prediction
    prediction = int(diabetes_model.predict(df_scaled)[0])
    probability = float(diabetes_model.predict_proba(df_scaled)[0][1])

    return {
        "prediction": prediction,
        "risk_probability_percentage": round(probability * 100, 2),
        "risk_level": "High Risk" if prediction == 1 else "Low Risk"
    }


# ----------------- 2. CARDIOVASCULAR MODEL ARTIFACTS -----------------
CARDIO_MODEL_PATH = os.path.join(CARDIO_MODEL_DIR, "cardio_model.joblib")
CARDIO_SCALER_PATH = os.path.join(CARDIO_MODEL_DIR, "cardio_scaler.joblib")

try:
    cardio_model = joblib.load(CARDIO_MODEL_PATH)
    cardio_scaler = joblib.load(CARDIO_SCALER_PATH)
except Exception as e:
    print(f"⚠️ ERROR LOADING CARDIO MODELS: {e}")
    cardio_model = None
    cardio_scaler = None

CARDIO_FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", 
    "restecg", "thalach", "exang", "oldpeak", 
    "slope", "ca", "thal"
]

def process_cardio_prediction(input_dict: dict) -> dict:
    """
    Process input features, apply scaling, predict heart disease risk, and return results.
    """
    if cardio_model is None or cardio_scaler is None:
        raise RuntimeError("Cardiovascular ML model or scaler is not loaded properly.")

    # 1. Convert directly to DataFrame (Fixes Scikit-learn feature name crash)
    df_cardio = pd.DataFrame([input_dict], columns=CARDIO_FEATURE_NAMES)
    
    # 2. Scale input data
    scaled_input = cardio_scaler.transform(df_cardio)
    
    # 3. Predict and calculate probability
    prediction = int(cardio_model.predict(scaled_input)[0])
    probabilities = cardio_model.predict_proba(scaled_input)[0]
    risk_probability = float(probabilities[1] * 100)
    
    risk_level = "High Risk" if prediction == 1 else "Low Risk"

    return {
        "prediction": prediction,
        "risk_level": risk_level,
        "risk_probability_percentage": round(risk_probability, 2)
    }