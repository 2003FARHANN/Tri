import os
import joblib
import pandas as pd

# Set paths based on the project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODEL_DIR = os.path.join(BASE_DIR, "ML_Model", "models")

MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
IMPUTER_PATH = os.path.join(MODEL_DIR, "imputer.pkl")

# Load the champion model and preprocessing artifacts
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    imputer = joblib.load(IMPUTER_PATH)
except FileNotFoundError as e:
    print(f"Error loading artifacts: {e}")

# Exact feature order expected by the trained model
FEATURE_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", 
    "SkinThickness", "Insulin", "BMI", 
    "DiabetesPedigreeFunction", "Age"
]

def process_and_predict(input_data: dict) -> dict:
    """
    Preprocess raw user input data and predict diabetes risk.
    """
    # Convert input dictionary to DataFrame with the correct column order
    df = pd.DataFrame([input_data], columns=FEATURE_NAMES)

    # 1. Biological zero validation (Replace 0 with NaN for specific columns)
    zero_invalid_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_invalid_cols:
        df[col] = df[col].replace(0, pd.NA)

    # 2. Imputation (Fill missing values using the trained imputer)
    df_imputed = pd.DataFrame(imputer.transform(df), columns=FEATURE_NAMES)

    # 3. Feature Scaling (Scale features using the trained scaler)
    df_scaled = pd.DataFrame(scaler.transform(df_imputed), columns=FEATURE_NAMES)

    # 4. Generate prediction and probability
    prediction = int(model.predict(df_scaled)[0])
    probability = float(model.predict_proba(df_scaled)[0][1])

    return {
        "prediction": prediction,
        "risk_probability_percentage": round(probability * 100, 2),
        "risk_level": "High Risk" if prediction == 1 else "Low Risk"
    }