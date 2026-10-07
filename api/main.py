import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from api.schemas import ForecastPoint, ForecastRequest, ForecastResponse, SeriesKey
from forecasting.model import Forecaster


@dataclass
class Bundle:
    forecaster: Forecaster
    metadata: dict
    history: dict[tuple[int, int], np.ndarray]
    origin: pd.Timestamp


def load_bundle(directory: Path) -> Bundle | None:
    if not (directory / "metadata.json").exists():
        return None
    metadata = json.loads((directory / "metadata.json").read_text())
    hist = pd.read_csv(directory / "history.csv", parse_dates=["date"]).sort_values("date")
    history = {
        (int(s), int(i)): g["sales"].to_numpy(dtype=float) for (s, i), g in hist.groupby(["store", "item"])
    }
    return Bundle(Forecaster.load(directory), metadata, history, hist["date"].max())


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = load_bundle(Path(os.getenv("ARTIFACT_DIR", "artifacts")))
    yield


app = FastAPI(title="Sales Forecasting API", version="1.0.0", lifespan=lifespan)


def get_bundle(request: Request) -> Bundle:
    bundle = request.app.state.bundle
    if bundle is None:
        raise HTTPException(503, "No trained model found. Run `python -m forecasting.train` first.")
    return bundle


@app.get("/health", tags=["meta"])
def health(request: Request):
    return {"status": "ok", "model_loaded": request.app.state.bundle is not None}


@app.get("/model/info", tags=["model"])
def model_info(request: Request):
    """Training metadata: tuned parameters, backtest scores for every strategy and feature importance."""
    return get_bundle(request).metadata


@app.get("/series", response_model=list[SeriesKey], tags=["model"])
def list_series(request: Request):
    return [SeriesKey(store=s, item=i) for s, i in sorted(get_bundle(request).history)]


@app.post("/forecast", response_model=ForecastResponse, tags=["forecast"])
def forecast(payload: ForecastRequest, request: Request):
    bundle = get_bundle(request)
    key = (payload.store, payload.item)
    if key not in bundle.history:
        raise HTTPException(404, f"No history for store={payload.store} item={payload.item}")

    max_horizon = bundle.forecaster.horizon
    if payload.horizon > max_horizon:
        raise HTTPException(422, f"horizon must be at most {max_horizon}")

    origin = bundle.origin
    dates = [origin + pd.Timedelta(days=s) for s in range(1, payload.horizon + 1)]
    valid_days = {d.date() for d in dates}
    outside = [d.isoformat() for d in payload.promo_dates if d not in valid_days]
    if outside:
        raise HTTPException(422, f"promo_dates outside the forecast window ({dates[0].date()} to {dates[-1].date()}): {outside}")
    promo_days = set(payload.promo_dates)
    promo = [int(d.date() in promo_days) for d in dates]

    strategy = payload.strategy or bundle.metadata["default_strategy"]
    values = bundle.forecaster.predict(
        bundle.history[key], origin, payload.store, payload.item, payload.horizon, strategy, promo
    )
    return ForecastResponse(
        store=payload.store,
        item=payload.item,
        strategy=strategy,
        origin_date=origin.date(),
        horizon=payload.horizon,
        forecasts=[ForecastPoint(date=d.date(), forecast=round(float(v), 2)) for d, v in zip(dates, values)],
    )
