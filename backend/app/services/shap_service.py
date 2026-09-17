import os
import json

# Set paths based on the project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SHAP_JSON_PATH = os.path.join(BASE_DIR, "ML_Model", "explainable_ai", "plots", "feature_importance.json")

def get_global_feature_importance() -> list:
    """
    Read and return the pre-generated SHAP feature importance JSON artifact.
    """
    if not os.path.exists(SHAP_JSON_PATH):
        return [{"error": "SHAP feature importance data not found"}]
    
    with open(SHAP_JSON_PATH, "r") as f:
        data = json.load(f)
        
    return data