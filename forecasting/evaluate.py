"""Rolling-origin backtest comparing recursive, direct, combined and a seasonal-naive baseline."""
import math

import numpy as np
import pandas as pd

from forecasting.model import Forecaster


def seasonal_naive(history: np.ndarray, horizon: int) -> np.ndarray:
    """Forecast each day with the value from the same weekday in the latest observed week."""
    return np.array([history[-1 + h - 7 * math.ceil(h / 7)] for h in range(1, horizon + 1)], dtype=float)


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    err = np.asarray(predicted) - np.asarray(actual)
    mse = float(np.mean(err**2))
    return {"mse": round(mse, 3), "rmse": round(math.sqrt(mse), 3), "mae": round(float(np.mean(np.abs(err))), 3)}


def backtest(
    forecaster: Forecaster, df: pd.DataFrame, cutoff: pd.Timestamp, horizon: int, n_origins: int = 2
) -> dict[str, dict[str, float]]:
    """Forecast `horizon` days from `n_origins` origins starting at `cutoff`, spaced `horizon` days apart.

    The forecaster must have been trained only on target dates <= cutoff, so every evaluated
    target date lies after the training data.
    """
    actuals, preds = [], {name: [] for name in ("recursive", "direct", "combined", "seasonal_naive")}
    for (store, item), g in df.groupby(["store", "item"]):
        g = g.set_index("date")
        for o in range(n_origins):
            origin = cutoff + pd.Timedelta(days=o * horizon)
            window = g.loc[origin + pd.Timedelta(days=1): origin + pd.Timedelta(days=horizon)]
            if len(window) < horizon:
                continue
            history = g.loc[:origin, "sales"].to_numpy(dtype=float)
            promo = window["promo"].astype(int).tolist()
            actuals.append(window["sales"].to_numpy(dtype=float))
            for strategy in ("recursive", "direct", "combined"):
                preds[strategy].append(forecaster.predict(history, origin, store, item, horizon, strategy, promo))
            preds["seasonal_naive"].append(seasonal_naive(history, horizon))
    if not actuals:
        raise ValueError("Not enough data after the cutoff for a backtest")
    actual = np.concatenate(actuals)
    return {name: metrics(actual, np.concatenate(p)) for name, p in preds.items()}
