from sqlalchemy import Column, Integer, Float, String, DateTime
from datetime import datetime
from app.database.database import Base

class PredictionRecord(Base):
    """
    SQLAlchemy model to store user inputs and prediction results in the database.
    """
    __tablename__ = "prediction_records"

    id = Column(Integer, primary_key=True, index=True)
    
    # Biological Data (User Inputs)
    pregnancies = Column(Integer)
    glucose = Column(Float)
    blood_pressure = Column(Float)
    skin_thickness = Column(Float)
    insulin = Column(Float)
    bmi = Column(Float)
    diabetes_pedigree = Column(Float)
    age = Column(Integer)
    
    # Prediction Results
    prediction = Column(Integer)
    risk_level = Column(String)
    risk_probability_percentage = Column(Float)
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow)
class CardioPredictionRecord(Base):
    """
    SQLAlchemy model to store cardiovascular prediction history in the database.
    """
    __tablename__ = "cardio_predictions"

    id = Column(Integer, primary_key=True, index=True)
    age = Column(Integer)
    sex = Column(Integer)
    cp = Column(Integer)
    trestbps = Column(Integer)
    chol = Column(Integer)
    fbs = Column(Integer)
    restecg = Column(Integer)
    thalach = Column(Integer)
    exang = Column(Integer)
    oldpeak = Column(Float)
    slope = Column(Integer)
    ca = Column(Integer)
    thal = Column(Integer)
    prediction = Column(Integer)
    risk_level = Column(String)
    risk_probability_percentage = Column(Float)


import joblib
import numpy as np
import os

# Define absolute paths for the cardio model and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARDIO_MODEL_PATH = os.path.join(BASE_DIR, "../ml_models/cardio_model.joblib")
CARDIO_SCALER_PATH = os.path.join(BASE_DIR, "../ml_models/cardio_scaler.joblib")

# Load cardio model and scaler if available
try:
    cardio_model = joblib.load(CARDIO_MODEL_PATH)
    cardio_scaler = joblib.load(CARDIO_SCALER_PATH)
except Exception as e:
    cardio_model = None
    cardio_scaler = None

def process_cardio_prediction(input_dict: dict) -> dict:
    """
    Process input features, apply scaling, predict heart disease risk, and return results.
    """
    if cardio_model is None or cardio_scaler is None:
        raise RuntimeError("Cardiovascular ML model or scaler is not loaded properly.")

    # Extract features in the exact order expected by the model
    features = [
        input_dict["age"],
        input_dict["sex"],
        input_dict["cp"],
        input_dict["trestbps"],
        input_dict["chol"],
        input_dict["fbs"],
        input_dict["restecg"],
        input_dict["thalach"],
        input_dict["exang"],
        input_dict["oldpeak"],
        input_dict["slope"],
        input_dict["ca"],
        input_dict["thal"]
    ]

    # Convert to numpy array and reshape for single prediction
    input_array = np.array(features).reshape(1, -1)
    
    # Scale input data
    scaled_input = cardio_scaler.transform(input_array)
    
    # Predict and calculate probability
    prediction = int(cardio_model.predict(scaled_input)[0])
    probabilities = cardio_model.predict_proba(scaled_input)[0]
    risk_probability = float(probabilities[1] * 100)
    
    risk_level = "High Risk" if prediction == 1 else "Low Risk"

    return {
        "prediction": prediction,
        "risk_level": risk_level,
        "risk_probability_percentage": round(risk_probability, 2)
    }