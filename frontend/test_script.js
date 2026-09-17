import { getDiabetesPrediction, getCardioPrediction } from './src/services/api.js';

async function runTest() {
    console.log("Testing Diabetes API from JS...");
    try {
        const diabetesRes = await getDiabetesPrediction({
            "Pregnancies": 2, "Glucose": 120, "BloodPressure": 80,
            "SkinThickness": 20, "Insulin": 85, "BMI": 25.5,
            "DiabetesPedigreeFunction": 0.5, "Age": 35
        });
        console.log("Diabetes Result:", diabetesRes);
    } catch (err) {
        console.error("Diabetes Test Failed:", err);
    }
}

runTest();