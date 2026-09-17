from pydantic import BaseModel

class PatientInputSchema(BaseModel):
    """
    Pydantic schema to validate the incoming request body from the frontend.
    The field names match the Exact FEATURE_NAMES required by the ML model.
    """
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

class PredictionResponseSchema(BaseModel):
    """
    Pydantic schema for the API response format.
    """
    prediction: int
    risk_probability_percentage: float
    risk_level: str