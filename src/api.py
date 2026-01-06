from fastapi import FastAPI
from pydantic import BaseModel
import os
import pickle
import numpy as np
from catboost import CatBoostRegressor
from src.features import build_features
import requests

app = FastAPI(title="Telegram Ads Reach Forecast")

# ----------- Google Drive helper -----------
def download_from_gdrive(file_id, dest_path):
    if os.path.exists(dest_path):
        return

    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None

    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(32768):
            f.write(chunk)

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
