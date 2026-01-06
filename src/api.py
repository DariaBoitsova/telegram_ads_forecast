from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
from catboost import CatBoostRegressor
from src.features import build_features

app = FastAPI(title="Telegram Ads Reach Forecast")

class PredictRequest(BaseModel):
    cpm: float
    channel: str
    date: str

model = CatBoostRegressor()
model.load_model("models/catboost_views_model.cbm")

with open("models/channel_stats.pkl", "rb") as f:
    channel_stats = pickle.load(f)

@app.post("/predict")
def predict(req: PredictRequest):
    X = build_features(req.cpm, req.channel, req.date, channel_stats)
    pred_log = model.predict(X)[0]
    pred = int(np.expm1(pred_log))
    return {"predicted_views": max(pred, 0)}
