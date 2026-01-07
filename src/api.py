from fastapi import FastAPI
from pydantic import BaseModel
import os
import pickle
import numpy as np
from catboost import CatBoostRegressor
from src.features import build_features
import requests
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request, Form
import pandas as pd
from io import StringIO
from fastapi import UploadFile, File
import gdown

templates = Jinja2Templates(directory="templates")

app = FastAPI(title="Telegram Ads Reach Forecast")
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ----------- Google Drive helper -----------
def download_from_gdrive(file_id, dest_path):
    if os.path.exists(dest_path):
        return

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest_path, quiet=False)

# ----------- ID файлов на Google Drive -----------
MODEL_ID = "16D7SmfEy_UfParBRvpAwBcfFW7MFd1CH"   # замените на свой ID
STATS_ID = "12YaddCb-bzws1u6RZS6K9cfQ4SWlk3u_"   # замените на свой ID

MODEL_PATH = "models/catboost_views_model.cbm"
STATS_PATH = "models/channel_stats.pkl"

# Скачиваем модели при старте
download_from_gdrive(MODEL_ID, MODEL_PATH)
download_from_gdrive(STATS_ID, STATS_PATH)

# ----------- Загружаем модель и статистику -----------
model = CatBoostRegressor()
model.load_model(MODEL_PATH)

with open(STATS_PATH, "rb") as f:
    channel_stats = pickle.load(f)

# ----------- API -----------
class PredictRequest(BaseModel):
    cpm: float
    channel: str
    date: str

@app.post("/predict")
def predict(req: PredictRequest):
    X = build_features(req.cpm, req.channel, req.date, channel_stats)
    pred_log = model.predict(X)[0]
    pred = int(np.expm1(pred_log))
    return {"predicted_views": max(pred, 0)}

@app.post("/predict_form", response_class=HTMLResponse)
def predict_form(
    request: Request,
    cpm: float = Form(...),
    channel: str = Form(...),
    date: str = Form(...)
):
    X = build_features(cpm, channel, date, channel_stats)
    pred_log = model.predict(X)[0]
    pred = int(np.expm1(pred_log))

    return HTMLResponse(
        f"""
        <h2>Prediction result</h2>
        <p><b>Predicted views:</b> {max(pred, 0)}</p>
        <a href="/">Back</a>
        """
    )

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(StringIO(content.decode("utf-8")))

    predictions = []

    for _, row in df.iterrows():
        X = build_features(
            cpm=row["cpm"],
            channel=row["channel"],
            date=row["date"],
            channel_stats=channel_stats
        )
        pred_log = model.predict(X)[0]
        pred = int(np.expm1(pred_log))
        predictions.append(max(pred, 0))

    df["predicted_views"] = predictions

    return df.to_dict(orient="records")

