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



from pydantic import BaseModel

class CardioInputSchema(BaseModel):
    """
    Pydantic schema for validating incoming cardiovascular patient data.
    """
    age: int
    sex: int
    cp: int          # Chest pain type
    trestbps: int    # Resting blood pressure
    chol: int        # Serum cholesterol
    fbs: int         # Fasting blood sugar
    restecg: int     # Resting electrocardiographic results
    thalach: int     # Maximum heart rate achieved
    exang: int       # Exercise induced angina
    oldpeak: float   # ST depression induced by exercise relative to rest
    slope: int       # The slope of the peak exercise ST segment
    ca: int          # Number of major vessels colored by fluoroscopy
    thal: int        # Thalassemia type

class CardioPredictionResponseSchema(BaseModel):
    """
    Pydantic schema for returning cardiovascular prediction results.
    """
    prediction: int
    risk_level: str
    risk_probability_percentage: float