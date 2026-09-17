// Base URL for the FastAPI backend server
const API_BASE_URL = "http://127.0.0.1:8000";

/**
 * Send patient data to the backend for Diabetes prediction
 * @param {Object} patientData - Object containing the 8 diabetes features
 * @returns {Promise<Object>} Prediction result (risk_level, percentage, etc.)
 */
export async function getDiabetesPrediction(patientData) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/predict`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            body: JSON.stringify(patientData)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        return data;
    } catch (error) {
        console.error("Failed to fetch diabetes prediction:", error);
        throw error;
    }
}

/**
 * Send cardiovascular data to the backend for Cardio prediction
 * @param {Object} cardioData - Object containing the 13 cardio features
 * @returns {Promise<Object>} Prediction result (risk_level, percentage, etc.)
 */
export async function getCardioPrediction(cardioData) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/predict/cardio`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            body: JSON.stringify(cardioData)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        return data;
    } catch (error) {
        console.error("Failed to fetch cardio prediction:", error);
        throw error;
    }
}