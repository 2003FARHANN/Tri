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