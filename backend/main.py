import io
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

BACKEND_DIR = Path(__file__).resolve().parent
ROOT = BACKEND_DIR.parent
for import_path in (ROOT, BACKEND_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

try:
    from .evaluation import evaluate
    from .features import ENGINEERED_COLUMNS
    from .model_service import ModelService
except ImportError:  # pragma: no cover
    from evaluation import evaluate
    from features import ENGINEERED_COLUMNS
    from model_service import ModelService

MODEL_PATH = Path(os.getenv("CHURN_MODEL_PATH", str(ROOT / "model.joblib")))
if not MODEL_PATH.exists():
    alt = ROOT / "data" / "model.joblib"
    if alt.exists():
        MODEL_PATH = alt
METRICS_PATH = ROOT / "metrics.json"
if not METRICS_PATH.exists():
    alt = ROOT / "data" / "metrics.json"
    if alt.exists():
        METRICS_PATH = alt
MODEL_CARD_PATH = ROOT / "model_card.json"
DEFAULT_THRESHOLD = 0.5
if MODEL_CARD_PATH.exists():
    try:
        card_threshold = float(json.loads(MODEL_CARD_PATH.read_text()).get("threshold", 0.5))
        if 0.01 <= card_threshold <= 0.99:
            DEFAULT_THRESHOLD = card_threshold
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass
MAX_BATCH_ROWS = 100_000

service: ModelService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    service = ModelService(MODEL_PATH)
    yield


app = FastAPI(title="Customer Churn API", version="1.0.0", lifespan=lifespan)


class CustomerInput(BaseModel):
    AccountAge: int = Field(ge=0, description="Account age in months")
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)
    SubscriptionType: str
    PaymentMethod: str
    PaperlessBilling: str
    ContentType: str
    MultiDeviceAccess: str
    DeviceRegistered: str
    ViewingHoursPerWeek: float = Field(ge=0)
    AverageViewingDuration: float = Field(ge=0)
    ContentDownloadsPerMonth: int = Field(ge=0)
    GenrePreference: str
    UserRating: float = Field(ge=0, le=5)
    SupportTicketsPerMonth: int = Field(ge=0)
    Gender: str
    WatchlistSize: int = Field(ge=0)
    ParentalControl: str
    SubtitlesEnabled: str


def risk_level(proba, threshold: float):
    """High = predicted churner; Medium = within 20% of the threshold; otherwise Low."""
    return np.where(proba >= threshold, "High", np.where(proba >= 0.8 * threshold, "Medium", "Low"))


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": service is not None}


@app.get("/schema")
def schema():
    """Everything the UI needs to build its input form, read straight from the fitted model."""
    return {
        "raw_numeric": service.raw_numeric,
        "raw_categorical": service.cat_cols,
        "raw_columns": service.raw_columns,
        "categories": service.categories,
        "engineered": ENGINEERED_COLUMNS,
        "default_threshold": DEFAULT_THRESHOLD,
    }


@app.get("/metrics")
def metrics():
    """Scores recorded in the training notebook (held-out test set)."""
    return json.loads(METRICS_PATH.read_text())


@app.get("/feature-importance")
def feature_importance(
    importance_type: str = Query("weight", pattern="^(weight|gain|cover|total_gain|total_cover)$"),
    top_n: int = Query(20, ge=1, le=100),
):
    return {"importance_type": importance_type, "features": service.importance(importance_type, top_n)}


@app.post("/predict")
def predict(customer: CustomerInput, threshold: float = Query(DEFAULT_THRESHOLD, ge=0.01, le=0.99)):
    raw = pd.DataFrame([customer.model_dump()])
    unseen = service.unseen_categories(raw)
    if unseen:
        raise HTTPException(422, detail=f"Unknown category value(s): {unseen}. Valid options: {service.categories}")
    try:
        X = service.prepare(raw)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))

    proba = float(service.predict_proba(X)[0])
    return {
        "churn_probability": proba,
        "predicted_churn": bool(proba >= threshold),
        "risk_level": str(risk_level(np.array([proba]), threshold)[0]),
        "threshold": threshold,
        "top_drivers": service.contributions(X, top_k=10),
    }


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...), threshold: float = Query(DEFAULT_THRESHOLD, ge=0.01, le=0.99)):
    """Score a CSV. If it also has a 0/1 `Churn` column, live evaluation graphs/scores are returned too."""
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
    except Exception as e:
        raise HTTPException(400, detail=f"Could not read CSV: {e}")
    if df.empty:
        raise HTTPException(400, detail="The CSV has no rows.")
    if len(df) > MAX_BATCH_ROWS:
        raise HTTPException(413, detail=f"Too many rows ({len(df):,}); the limit is {MAX_BATCH_ROWS:,}.")

    try:
        X = service.prepare(df)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except Exception as e:
        raise HTTPException(400, detail=f"Could not process the data: {e}")

    proba = service.predict_proba(X)
    ids = df["CustomerID"].astype(str).tolist() if "CustomerID" in df.columns else list(map(str, range(len(df))))

    evaluation, evaluation_note = None, None
    if "Churn" in df.columns:
        labelled = df["Churn"].notna()
        try:
            evaluation = evaluate(df.loc[labelled, "Churn"].to_numpy(), proba[labelled.to_numpy()], threshold)
        except (ValueError, TypeError) as e:
            evaluation_note = str(e)

    return {
        "n_rows": int(len(df)),
        "threshold": threshold,
        "customer_id": ids,
        "churn_probability": proba.tolist(),
        "predicted_churn": (proba >= threshold).astype(int).tolist(),
        "risk_level": risk_level(proba, threshold).tolist(),
        "unseen_categories": service.unseen_categories(df),
        "evaluation": evaluation,
        "evaluation_note": evaluation_note,
    }
