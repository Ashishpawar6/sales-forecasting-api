# Sales Forecasting API

Multi-step **sales forecasting with LightGBM**, served through a **FastAPI** service.
It compares three forecasting strategies (recursive, direct and combined) against a seasonal-naive baseline, tunes hyperparameters with time-series cross-validation, and exposes the best model through a validated, tested REST API.

**Stack:** Python, LightGBM, pandas, NumPy, scikit-learn, FastAPI, pytest, Docker.

## Why this project

Retail-style demand forecasting needs more than a model fit: leakage-free features, honest time-based evaluation, a baseline to beat, and a way for other systems to use the result. This repo covers that path end to end.

## Results

Backtest on held-out days the models never saw (2 forecast origins × 14 days × 12 store/item series), using synthetic data:

| Strategy | MSE | RMSE | MAE |
|---|---|---|---|
| Seasonal naive (baseline) | 65.34 | 8.08 | 5.77 |
| Recursive | 22.93 | 4.79 | 3.50 |
| Direct | 22.19 | 4.71 | 3.39 |
| **Combined (average of both)** | **21.38** | **4.62** | **3.33** |

All three LightGBM strategies cut error by about 40% against the baseline, and averaging recursive and direct beats either alone. The API serves the best strategy by default.
In the 1-step model the most important features (gain) are `lag_6`, `lag_13` and `lag_27` (about 85% together), which are the same weekday 1, 2 and 4 weeks earlier, then `promo`; this matches the weekly seasonality built into the data.

> **About the data.** The default dataset is a *synthetic* generator (`forecasting/data.py`): 12 store/item series, 3 years of daily sales with trend, weekly and yearly seasonality, promotions and noise. It keeps the repo self-contained and reproducible. Real data works unchanged: pass any CSV with `date, store, item, sales[, promo]` to `--data` (each series must be continuous daily data). Metrics on real data will differ.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

macOS only: LightGBM needs the OpenMP runtime, so run `brew install libomp` once.

## Run

```bash
python -m forecasting.train                              # synthetic data, ~1 minute
python -m forecasting.train --data my_sales.csv          # your own data
python -m forecasting.train --no-tune --n-estimators 100 # quicker experiment
uvicorn api.main:app --reload                            # serve at http://127.0.0.1:8000 (docs at /docs)
pytest                                                   # run the tests
```

Training writes `artifacts/` (one LightGBM model per horizon, `metadata.json`, and recent `history.csv`). The API reads it from `ARTIFACT_DIR` (default `artifacts`).

Docker (not built in my development environment, so treat it as untested):

```bash
docker build -t sales-forecasting-api .
docker run -p 8000:8000 sales-forecasting-api
```

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness and whether a model is loaded |
| `GET` | `/model/info` | Tuned params, backtest scores per strategy, feature importance |
| `GET` | `/series` | Available store/item pairs |
| `POST` | `/forecast` | Forecast the next `horizon` days for one series |

```bash
curl -X POST http://127.0.0.1:8000/forecast -H 'Content-Type: application/json' \
  -d '{"store": 1, "item": 2, "horizon": 5, "promo_dates": ["2025-01-03"]}'
```

```json
{
  "store": 1, "item": 2, "strategy": "combined", "origin_date": "2024-12-31", "horizon": 5,
  "forecasts": [
    {"date": "2025-01-01", "forecast": 36.49},
    {"date": "2025-01-02", "forecast": 38.78},
    {"date": "2025-01-03", "forecast": 60.54},
    {"date": "2025-01-04", "forecast": 55.59},
    {"date": "2025-01-05", "forecast": 49.93}
  ]
}
```

The 3 January jump reflects the planned promotion. Request fields: `store`, `item`, `horizon` (1 up to the trained horizon, default 7), optional `strategy` (`recursive`, `direct` or `combined`) and optional `promo_dates` (planned promotion days inside the window). Errors: `404` unknown series, `422` invalid input (bad horizon/strategy/dates), `503` no trained model.

## How it works

```
forecasting/
  data.py       synthetic generator, CSV loading and validation
  features.py   lag / rolling / calendar features (training and serving share one definition)
  model.py      per-horizon LightGBM models; recursive, direct and combined prediction
  evaluate.py   rolling-origin backtest and metrics, seasonal-naive baseline
  train.py      tuning, backtest, final fit, artifact writing (CLI)
api/            FastAPI app and schemas
tests/          37 tests
```

**Strategies.** For a horizon `h`, a model predicts the day `h` days after the *origin* (last observed day) from information known at the origin only.
- *Direct:* a separate model for each `h = 1..14`.
- *Recursive:* one 1-step model applied repeatedly, feeding each prediction back in as history.
- *Combined:* the mean of the two. Recursive errors accumulate while direct models ignore the dependency between days, so averaging usually helps.

**Features.** Lags of sales at the origin (0, 1, 6, 13, 27 days), rolling means (7 and 28 days), calendar features of the target day, a planned-promotion flag (known in advance) and categorical store/item. All series share one global model per horizon.

**Evaluation.** The last 28 days are held out. Models are trained only on target dates before that cutoff, then forecast 14 days from two origins. Hyperparameters (`num_leaves`, `learning_rate`) are tuned on the 1-step model with 3-fold expanding-window time-series cross-validation that never touches the held-out days. The final models are refit on all data for serving.

**Leakage guard.** `tests/test_features.py` checks that the vectorised training features equal the features the API computes from raw history, and that altering any value after the origin day does not change a row's features.

## Design decisions and limitations

- **Global models per horizon** instead of one model per series, so series with little history borrow strength; the cost is more models to store (14 small files).
- **Gradient boosting over deep learning:** strong on tabular lag features, fast to train, and easy to explain through feature importance.
- **Native LightGBM text files + JSON metadata** instead of pickle, so artifacts are portable and safe to load.
- **The API needs recent history**, so it stores the last 60 days per series with the model and always forecasts from the end of the training data. A production version would read fresh sales from a database and add a retraining schedule.
- **Synthetic data** means the absolute numbers are illustrative; the pipeline and evaluation method are what carry over.
- No authentication, monitoring or drift detection.

## Future scope

- Prediction intervals with quantile regression (LightGBM `objective="quantile"`).
- Hierarchical forecasts (reconcile item, store and total) and holiday features.
- Model registry and scheduled retraining; drift monitoring.
- Real public data (for example the Kaggle Store Item Demand or M5 datasets).
