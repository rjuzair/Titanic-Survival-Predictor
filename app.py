from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).parent

app = FastAPI(title="Titanic Survival Predictor")
model = joblib.load(BASE_DIR / "model.joblib")

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


# Passenger data model (categories match the training data)
class Passenger(BaseModel):
    Sex: Literal["male", "female"]
    Age: float = Field(ge=0, le=100)
    Pclass: Literal[1, 2, 3]
    Fare: float = Field(ge=0)
    Embarked: Literal["C", "Q", "S"]
    Title: Literal["Mr", "Mrs", "Miss", "Master", "Rare"]
    Family_size: Literal["Alone", "Small", "Large"]


# Serve the main HTML page
@app.get("/", response_class=HTMLResponse)
async def get_ui():
    return HTMLResponse(content=(BASE_DIR / "templates" / "index.html").read_text())


@app.get("/health")
async def health():
    return {"status": "ok"}


# Prediction endpoint
@app.post("/predict")
async def predict(data: Passenger):
    df = pd.DataFrame([data.model_dump()])
    prediction = model.predict(df)[0]
    return {"prediction": int(prediction)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
