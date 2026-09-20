import requests

# Base URL for the FastAPI server
BASE_URL = "http://127.0.0.1:8000"

print("--- Starting Backend API Tests ---\n")

# 1. Test the Root Endpoint
try:
    response = requests.get(f"{BASE_URL}/")
    print(f"✅ Root Endpoint: {response.status_code} | {response.json()}")
except Exception as e:
    print(f"❌ Root Endpoint Failed: {e}")

# 2. Test the Cardiovascular Prediction Model
cardio_payload = {
    "age": 45, "sex": 1, "cp": 3, "trestbps": 120, "chol": 230,
    "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
    "oldpeak": 1.5, "slope": 2, "ca": 0, "thal": 2
}
try:
    response = requests.post(f"{BASE_URL}/api/predict/cardio", json=cardio_payload)
    print(f"✅ Cardio Model: {response.status_code} | {response.json()}")
except Exception as e:
    print(f"❌ Cardio Model Failed: {e}")

# 3. Test the Diabetes Prediction Model (Using correct route: /api/predict)
diabetes_payload = {
    "Pregnancies": 2,
    "Glucose": 120,
    "BloodPressure": 80,
    "SkinThickness": 20,
    "Insulin": 85,
    "BMI": 25.5,
    "DiabetesPedigreeFunction": 0.5,
    "Age": 35
}
try:
    response = requests.post(f"{BASE_URL}/api/predict", json=diabetes_payload)
    print(f"✅ Diabetes Model: {response.status_code} | {response.json()}")
except Exception as e:
    print(f"❌ Diabetes Model Failed: {e}")

print("\n--- All tests executed! ---")