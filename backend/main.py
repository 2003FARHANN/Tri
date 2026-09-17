import os
# Prevent numpy/sklearn multithreading deadlock on Windows
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import prediction
from app.database.database import engine, Base

# Create all database tables based on SQLAlchemy models
Base.metadata.create_all(bind=engine)

# Initialize FastAPI Application with the exact variable name 'app'
app = FastAPI(
    title="TriNexus-VitaGuard-AI API",
    description="Backend API for Diabetes and Cardiovascular Risk Prediction and Explainable AI",
    version="1.0.0"
)

# Configure CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the prediction router with /api prefix
app.include_router(prediction.router, prefix="/api", tags=["Prediction"])

@app.get("/")
def root():
    """
    Root endpoint to verify the API is running.
    """
    return {"message": "Welcome to TriNexus-VitaGuard-AI API Server is running!"}