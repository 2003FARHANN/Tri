from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database.database import SessionLocal
from app.models.schemas import PatientInputSchema, PredictionResponseSchema
from app.models.models import PredictionRecord
from app.services.model_service import process_and_predict
from app.services.shap_service import get_global_feature_importance

# Initialize API Router
router = APIRouter()

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/predict", response_model=PredictionResponseSchema)
def predict_diabetes_risk(patient_data: PatientInputSchema, db: Session = Depends(get_db)):
    """
    Receive patient data from frontend, get ML prediction, save to database, and return the result.
    """
    # Convert Pydantic input schema to a dictionary
    input_dict = patient_data.model_dump()
    
    # Process data and get prediction from the ML service
    result = process_and_predict(input_dict)
    
    # Create a new database record
    db_record = PredictionRecord(
        pregnancies=input_dict["Pregnancies"],
        glucose=input_dict["Glucose"],
        blood_pressure=input_dict["BloodPressure"],
        skin_thickness=input_dict["SkinThickness"],
        insulin=input_dict["Insulin"],
        bmi=input_dict["BMI"],
        diabetes_pedigree=input_dict["DiabetesPedigreeFunction"],
        age=input_dict["Age"],
        prediction=result["prediction"],
        risk_level=result["risk_level"],
        risk_probability_percentage=result["risk_probability_percentage"]
    )
    
    # Save the record to the database
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    
    return result

@router.get("/shap-importance")
def get_shap_data():
    """
    Fetch and return the global feature importance JSON data for frontend visualization.
    """
    return get_global_feature_importance()


from app.models.schemas import CardioInputSchema, CardioPredictionResponseSchema
from app.models.models import CardioPredictionRecord
from app.services.model_service import process_cardio_prediction

@router.post("/predict/cardio", response_model=CardioPredictionResponseSchema)
def predict_cardio_risk(patient_data: CardioInputSchema, db: Session = Depends(get_db)):
    """
    Receive cardiovascular patient data from frontend, get ML prediction, save to database, and return result.
    """
    # Convert Pydantic input schema to a dictionary
    input_dict = patient_data.model_dump()
    
    # Process data and get prediction from the ML service
    result = process_cardio_prediction(input_dict)
    
    # Create a new database record for cardio
    db_record = CardioPredictionRecord(
        age=input_dict["age"],
        sex=input_dict["sex"],
        cp=input_dict["cp"],
        trestbps=input_dict["trestbps"],
        chol=input_dict["chol"],
        fbs=input_dict["fbs"],
        restecg=input_dict["restecg"],
        thalach=input_dict["thalach"],
        exang=input_dict["exang"],
        oldpeak=input_dict["oldpeak"],
        slope=input_dict["slope"],
        ca=input_dict["ca"],
        thal=input_dict["thal"],
        prediction=result["prediction"],
        risk_level=result["risk_level"],
        risk_probability_percentage=result["risk_probability_percentage"]
    )
    
    # Save the record to the database
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    
    return result