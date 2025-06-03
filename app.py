from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI()


model = joblib.load("model.joblib")

app.mount("/static", StaticFiles(directory="static"), name="static")

# Passenger data model
class Passenger(BaseModel):
    Sex: str
    Age: float
    Pclass: int
    Fare: float
    Embarked: str
    Title: str
    Family_size: str

# Serve the main HTML page
@app.get("/", response_class=HTMLResponse)
async def get_ui():
    index_path = Path("templates/index.html")
    return HTMLResponse(content=index_path.read_text(), status_code=200)

# Prediction endpoint
@app.post("/predict")
async def predict(data: Passenger):
    df = pd.DataFrame([data.dict()])

    # Optional: enforce types
    df["Pclass"] = df["Pclass"].astype(int)
    df["Age"] = df["Age"].astype(float)
    df["Fare"] = df["Fare"].astype(float)

    prediction = model.predict(df)[0]

    return {"prediction": int(prediction)}
