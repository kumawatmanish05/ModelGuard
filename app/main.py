from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import json 
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(
    title = "ModelGuard API",
    description = "ML model Reliability and prediction platform",
    version = "1.0.0",
)


BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"


#Load the Saved model once when the API starts
MODEL_PATH = "models/modelguard.pkl"
model = joblib.load(MODEL_PATH)
METRICS_PATH = "reports/metrics.json"


#Pydantic Pattern
class IncomeInput(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int 
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str


@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "index.html")

@app.get("/health")
def health_check():
    return{
        "status": "healthy",
    }

@app.get("/model-info")
def model_info():
    with open(METRICS_PATH, "r" , encoding="utf-8") as file :
        metrics = json.load(file)
    return metrics


@app.get("/quality-check")
def quality_check():
    with open(METRICS_PATH, "r", encoding="utf-8") as file :
        metrics = json.load(file)

    return{
        "selected_model": metrics["selected_model"],
        "quality_gate_passed": metrics["quality_gate_passed"],
        "recall_over_50k": metrics["recall_over_50k"],
        "required_recall":0.80,
    }


@app.post("/predict")
def predict_income(data: IncomeInput):
    input_data = pd.DataFrame([data.model_dump()])
    prediction = model.predict(input_data)[0]

    return {"prediction": str(prediction)}

app.mount(
    "/",
    StaticFiles(directory=FRONTEND_DIR),
    name="frontent",
)