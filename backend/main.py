from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

# Load model and tools
model = joblib.load('models/xgb_model.pkl')
scaler = joblib.load('models/scaler.pkl')
explainer = joblib.load('models/shap_explainer.pkl')

class PatientData(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.post("/predict")
def predict(data: PatientData):
    # BMI category dummies (match training logic)
    bmi = data.BMI
    bmi_normal = 1 if 18.5 <= bmi <= 24.9 else 0
    bmi_overweight = 1 if 25 <= bmi <= 29.9 else 0
    bmi_obese = 1 if bmi > 29.9 else 0

    # Feature list (matching training)
    features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
                "BMI_Category_Normal", "BMI_Category_Overweight", "BMI_Category_Obese"]

    # Create DataFrame for scaler (avoids warnings)
    X = pd.DataFrame([[
        data.Pregnancies, data.Glucose, data.BloodPressure,
        data.SkinThickness, data.Insulin, data.BMI,
        data.DiabetesPedigreeFunction, data.Age,
        bmi_normal, bmi_overweight, bmi_obese
    ]], columns=features)

    X_scaled = scaler.transform(X)
    prediction = int(model.predict(X_scaled)[0])

    # Load DejaVu font for multilingual PDF support
    FONT_PATH = "DejaVuSans.ttf"
    FONT_PATH = "bangla.ttf"

    # Get SHAP values
    shap_values = explainer(X_scaled)
    if isinstance(shap_values, list):
        values = shap_values[prediction][0]
    else:
        values = shap_values.values[0]

    top_features = sorted(
        zip(features, values),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:3]

    # Convert to native Python types
    return {
        "prediction": prediction,
        "risk_level": "High" if prediction else "Low",
        "top_risk_factors": [{"feature": f, "impact": float(round(v, 3))} for f, v in top_features]
    }
