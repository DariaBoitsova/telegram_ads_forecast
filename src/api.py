from fastapi import FastAPI
from pydantic import BaseModel
import os
import pickle
import numpy as np
from catboost import CatBoostRegressor
from src.features import build_features
from fastapi.responses import StreamingResponse
import io
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request, Form
import pandas as pd
from io import StringIO
from fastapi import UploadFile, File
import gdown
import uuid

TMP_DIR = "tmp"
os.makedirs(TMP_DIR, exist_ok=True)

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

CALIBRATION_ALPHA = 0.9
MAX_VIEWS_MULTIPLIER = 3.0
# ----------- API -----------
class PredictRequest(BaseModel):
    CPM: float
    CHANNEL_NAME: str
    DATE: str

@app.post("/predict")
def predict(req: PredictRequest):
    X = build_features(req.CPM, req.CHANNEL_NAME, req.DATE, channel_stats)
    pred_log = model.predict(X)[0]
    raw_pred = np.expm1(pred_log)
    pred = int(CALIBRATION_ALPHA * raw_pred)
    stats = channel_stats.get(req.CHANNEL_NAME, channel_stats["__global__"])
    cap = MAX_VIEWS_MULTIPLIER * stats["mean"]

    pred = min(pred, int(cap))
    pred = max(pred, 0)

    return {"VIEWS": max(pred, 0)}

@app.post("/predict_form", response_class=HTMLResponse)
def predict_form(
    request: Request,
    CPM: float = Form(...),
    CHANNEL_NAME: str = Form(...),
    DATE: str = Form(...)
):
    X = build_features(CPM, CHANNEL_NAME, DATE, channel_stats)
    pred_log = model.predict(X)[0]
    raw_pred = np.expm1(pred_log)
    pred = int(CALIBRATION_ALPHA * raw_pred)
    stats = channel_stats.get(CHANNEL_NAME, channel_stats["__global__"])
    cap = MAX_VIEWS_MULTIPLIER * stats["mean"]

    pred = min(pred, int(cap))
    pred = max(pred, 0)
    return HTMLResponse(
        f"""
        <h2>Prediction result</h2>
        <p><b>VIEWS:</b> {max(pred, 0)}</p>
        <a href="/">Back</a>
        """
    )



@app.post("/predict_csv", response_class=HTMLResponse)
async def predict_csv(request: Request, file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(StringIO(content.decode("utf-8")))
    df.columns = (
        df.columns
          .str.strip()
          .str.upper()
    )
    rename_map = {
        "CHANNEL": "CHANNEL_NAME",
        "CHANNELNAME": "CHANNEL_NAME"
    }

    df = df.rename(columns=rename_map)
    required_cols = {"CPM", "CHANNEL_NAME", "DATE"}
    missing = required_cols - set(df.columns)

    if missing:
        return {
            "error": f"Missing required columns: {', '.join(missing)}",
            "expected_columns": list(required_cols)
        }
    
    predictions = []

    for _, row in df.iterrows():
        CHANNEL_NAME = row["CHANNEL_NAME"]

        stats = channel_stats.get(
            CHANNEL_NAME,
            channel_stats["__global__"]
        )
        try:
            cpm = float(row["CPM"])
            channel = str(row["CHANNEL_NAME"])
            date = row["DATE"]

            X = build_features(
                CPM=cpm,
                CHANNEL_NAME=channel,
                DATE=date,
                channel_stats=channel_stats
            )

            pred_log = model.predict(X)[0]
            raw_pred = np.expm1(pred_log)
            pred = int(CALIBRATION_ALPHA * raw_pred)

            stats = channel_stats.get(channel, channel_stats["__global__"])
            cap = MAX_VIEWS_MULTIPLIER * stats["mean"]

            pred = min(pred, int(cap))
            pred = max(pred, 0)
            predictions.append(pred)

        except Exception:
            predictions.append(None)

    df["VIEWS"] = predictions

    # --- сохраняем CSV ---
    file_id = str(uuid.uuid4())
    output_path = os.path.join(TMP_DIR, f"{file_id}.csv")
    df.to_csv(output_path, index=False)

    # --- preview таблица ---
    preview_html = df.head(10).to_html(
        classes="table table-striped",
        index=False
    )

    return templates.TemplateResponse(
        "csv_result.html",
        {
            "request": request,
            "table": preview_html,
            "download_id": file_id
        }
    )
@app.get("/download_csv/{file_id}")
def download_csv(file_id: str):
    path = os.path.join(TMP_DIR, f"{file_id}.csv")
    return FileResponse(
        path,
        media_type="text/csv",
        filename="prediction_result.csv"
    )
