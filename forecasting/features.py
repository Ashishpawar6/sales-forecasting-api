"""Feature engineering shared by training and serving.

A model for horizon h predicts sales on a *target* date using only information available at the
*origin* date (target - h days): lags and rolling means of sales up to the origin, plus things
known in advance about the target date (calendar, planned promotion).

`training_frame` builds those features for a whole history at once with shifts.
`inference_row` builds the same features for one forecast from a history array.
tests/test_features.py checks that both produce identical values, which guards against leakage.
"""
import numpy as np
import pandas as pd

from forecasting.config import CALENDAR_FEATURES, CATEGORICAL, FEATURES, LAGS, WINDOWS


def calendar_frame(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dow": dates.dayofweek,
            "month": dates.month,
            "day": dates.day,
            "dayofyear": dates.dayofyear,
            "weekofyear": dates.isocalendar().week.astype(int).to_numpy(),
        }
    )[CALENDAR_FEATURES]


def set_categories(frame: pd.DataFrame, stores: list[int], items: list[int]) -> pd.DataFrame:
    frame = frame.copy()
    frame["store"] = pd.Categorical(frame["store"], categories=stores)
    frame["item"] = pd.Categorical(frame["item"], categories=items)
    return frame[FEATURES]


def training_frame(df: pd.DataFrame, h: int, stores: list[int], items: list[int]) -> pd.DataFrame:
    """Rows indexed by target date; columns FEATURES + 'date' + 'target'. Rows with no full history are dropped."""
    parts = []
    for (store, item), g in df.groupby(["store", "item"], sort=True):
        g = g.sort_values("date")
        y = g["sales"].astype(float).reset_index(drop=True)
        feats = pd.DataFrame({f"lag_{k}": y.shift(h + k) for k in LAGS})
        for w in WINDOWS:
            feats[f"roll_mean_{w}"] = y.shift(h).rolling(w).mean()
        feats = pd.concat([feats, calendar_frame(pd.DatetimeIndex(g["date"]))], axis=1)
        feats["promo"] = g["promo"].to_numpy()
        feats["store"], feats["item"] = store, item
        feats["date"] = g["date"].to_numpy()
        feats["target"] = y
        parts.append(feats.dropna())
    out = pd.concat(parts, ignore_index=True)
    out[FEATURES] = set_categories(out, stores, items)
    return out


def inference_row(
    history: np.ndarray, target_date: pd.Timestamp, promo: int, store: int, item: int
) -> dict:
    """Features for forecasting `target_date` from `history` (sales up to and including the origin day)."""
    row = {f"lag_{k}": float(history[-1 - k]) for k in LAGS}
    for w in WINDOWS:
        row[f"roll_mean_{w}"] = float(np.mean(history[-w:]))
    row.update(calendar_frame(pd.DatetimeIndex([target_date])).iloc[0].to_dict())
    row.update({"promo": int(promo), "store": store, "item": item})
    return row


def rows_to_frame(rows: list[dict], stores: list[int], items: list[int]) -> pd.DataFrame:
    return set_categories(pd.DataFrame(rows), stores, items)


__all__ = ["training_frame", "inference_row", "rows_to_frame", "calendar_frame", "CATEGORICAL"]
