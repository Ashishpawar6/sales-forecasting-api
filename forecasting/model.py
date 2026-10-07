"""Multi-step forecasting with LightGBM.

One model per horizon h = 1..H is trained (the *direct* strategy). The h = 1 model can also be
applied repeatedly, feeding each prediction back in as history (the *recursive* strategy).
*Combined* is the average of the two.
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from forecasting.config import CATEGORICAL, FEATURES, MAX_HISTORY, STRATEGIES
from forecasting.features import inference_row, rows_to_frame, training_frame

DEFAULT_PARAMS = {"num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 20}


def fit_horizon(frame: pd.DataFrame, params: dict, n_estimators: int) -> lgb.Booster:
    model = lgb.LGBMRegressor(
        n_estimators=n_estimators, subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
        random_state=42, verbose=-1, **params,
    )
    model.fit(frame[FEATURES], frame["target"], categorical_feature=CATEGORICAL)
    return model.booster_


class Forecaster:
    def __init__(self, models: dict[int, lgb.Booster], stores: list[int], items: list[int]):
        self.models = models
        self.stores = stores
        self.items = items
        self.horizon = max(models)

    # ---- training --------------------------------------------------------------------------
    @classmethod
    def train(
        cls, df: pd.DataFrame, horizon: int, params: dict | None = None,
        n_estimators: int = 300, max_target_date: pd.Timestamp | None = None,
    ) -> "Forecaster":
        """Fit one model per horizon. Only rows with target date <= max_target_date are used."""
        stores, items = sorted(df["store"].unique().tolist()), sorted(df["item"].unique().tolist())
        models = {}
        for h in range(1, horizon + 1):
            frame = training_frame(df, h, stores, items)
            if max_target_date is not None:
                frame = frame[frame["date"] <= max_target_date]
            models[h] = fit_horizon(frame, params or DEFAULT_PARAMS, n_estimators)
        return cls(models, stores, items)

    # ---- prediction ------------------------------------------------------------------------
    def predict(
        self, history: np.ndarray, origin: pd.Timestamp, store: int, item: int,
        horizon: int, strategy: str, promo: list[int] | None = None,
    ) -> np.ndarray:
        """Forecast `horizon` days after `origin`. `history` is sales up to and including `origin`."""
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        if horizon > self.horizon:
            raise ValueError(f"horizon {horizon} exceeds the trained horizon {self.horizon}")
        if len(history) < MAX_HISTORY:
            raise ValueError(f"need at least {MAX_HISTORY} days of history, got {len(history)}")
        promo = list(promo or [0] * horizon)
        dates = [origin + pd.Timedelta(days=s) for s in range(1, horizon + 1)]

        if strategy == "recursive":
            return self._recursive(history, dates, promo, store, item)
        if strategy == "direct":
            return self._direct(history, dates, promo, store, item)
        return (self._recursive(history, dates, promo, store, item) + self._direct(history, dates, promo, store, item)) / 2

    def _direct(self, history, dates, promo, store, item) -> np.ndarray:
        rows = [inference_row(history, d, promo[i], store, item) for i, d in enumerate(dates)]
        frame = rows_to_frame(rows, self.stores, self.items)
        preds = [self.models[i + 1].predict(frame.iloc[[i]])[0] for i in range(len(dates))]
        return np.clip(preds, 0, None)

    def _recursive(self, history, dates, promo, store, item) -> np.ndarray:
        rolling = list(history[-MAX_HISTORY:])
        preds = []
        for i, d in enumerate(dates):
            row = inference_row(np.asarray(rolling), d, promo[i], store, item)
            pred = max(0.0, float(self.models[1].predict(rows_to_frame([row], self.stores, self.items))[0]))
            preds.append(pred)
            rolling.append(pred)  # the prediction becomes the newest "observed" day
        return np.asarray(preds)

    def feature_importance(self) -> dict[str, float]:
        """Gain importance of the 1-step model, normalised to sum to 1."""
        booster = self.models[1]
        gain = booster.feature_importance(importance_type="gain")
        total = float(gain.sum()) or 1.0
        pairs = sorted(zip(booster.feature_name(), gain / total), key=lambda p: -p[1])
        return {name: round(float(v), 4) for name, v in pairs}

    # ---- persistence -----------------------------------------------------------------------
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for h, booster in self.models.items():
            booster.save_model(str(directory / f"model_h{h:02d}.txt"))
        (directory / "model_index.json").write_text(
            json.dumps({"horizon": self.horizon, "stores": self.stores, "items": self.items})
        )

    @classmethod
    def load(cls, directory: Path) -> "Forecaster":
        index = json.loads((directory / "model_index.json").read_text())
        models = {h: lgb.Booster(model_file=str(directory / f"model_h{h:02d}.txt")) for h in range(1, index["horizon"] + 1)}
        return cls(models, index["stores"], index["items"])
