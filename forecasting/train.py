"""Train, tune, backtest and save the forecasting models.

    python -m forecasting.train --data data/sales.csv --out artifacts
"""
import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting.config import DEFAULT_HORIZON, KEEP_HISTORY_DAYS
from forecasting.data import generate_sales, load_sales, validate_sales
from forecasting.evaluate import backtest
from forecasting.features import training_frame
from forecasting.model import DEFAULT_PARAMS, Forecaster, fit_horizon

PARAM_GRID = {"num_leaves": [15, 31], "learning_rate": [0.03, 0.1]}


def tune(df: pd.DataFrame, cutoff: pd.Timestamp, n_estimators: int, n_folds: int = 3) -> tuple[dict, list[dict]]:
    """Grid search on the 1-step model with expanding-window time-series cross-validation.

    Only data up to `cutoff` is used, so the backtest period stays untouched.
    """
    stores, items = sorted(df["store"].unique().tolist()), sorted(df["item"].unique().tolist())
    frame = training_frame(df, 1, stores, items)
    frame = frame[frame["date"] <= cutoff]
    dates = np.sort(frame["date"].unique())
    chunks = np.array_split(dates, n_folds + 1)  # fold i: train on chunks[:i+1], validate on chunks[i+1]

    results = []
    for values in itertools.product(*PARAM_GRID.values()):
        params = {**DEFAULT_PARAMS, **dict(zip(PARAM_GRID, values))}
        rmses = []
        for i in range(n_folds):
            train = frame[frame["date"] <= chunks[i][-1]]
            valid = frame[frame["date"].isin(chunks[i + 1])]
            booster = fit_horizon(train, params, n_estimators)
            pred = booster.predict(valid[booster.feature_name()])
            rmses.append(float(np.sqrt(np.mean((pred - valid["target"].to_numpy()) ** 2))))
        results.append({"params": params, "cv_rmse": round(float(np.mean(rmses)), 3)})
    results.sort(key=lambda r: r["cv_rmse"])
    return results[0]["params"], results


def train_pipeline(
    df: pd.DataFrame, out_dir: Path, horizon: int = DEFAULT_HORIZON, n_estimators: int = 300,
    do_tune: bool = True, n_origins: int = 2,
) -> dict:
    df = validate_sales(df)
    last_date = df["date"].max()
    cutoff = last_date - pd.Timedelta(days=horizon * n_origins)  # last `horizon * n_origins` days are held out

    best_params, tuning = (tune(df, cutoff, n_estimators) if do_tune else (DEFAULT_PARAMS, []))

    backtest_model = Forecaster.train(df, horizon, best_params, n_estimators, max_target_date=cutoff)
    scores = backtest(backtest_model, df, cutoff, horizon, n_origins)
    default_strategy = min(("recursive", "direct", "combined"), key=lambda s: scores[s]["rmse"])

    final = Forecaster.train(df, horizon, best_params, n_estimators)  # refit on all data for serving
    final.save(out_dir)

    history = df.groupby(["store", "item"], group_keys=False).tail(KEEP_HISTORY_DAYS)
    history.to_csv(out_dir / "history.csv", index=False)

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "horizon": horizon,
        "last_date": last_date.strftime("%Y-%m-%d"),
        "backtest_cutoff": cutoff.strftime("%Y-%m-%d"),
        "backtest_origins": n_origins,
        "stores": final.stores,
        "items": final.items,
        "best_params": best_params,
        "tuning_results": tuning,
        "backtest": scores,
        "default_strategy": default_strategy,
        "feature_importance": final.feature_importance(),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", help="CSV with date,store,item,sales[,promo]; synthetic data is used if omitted")
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--no-tune", action="store_true")
    args = parser.parse_args()

    df = load_sales(args.data) if args.data else generate_sales()
    meta = train_pipeline(df, Path(args.out), args.horizon, args.n_estimators, not args.no_tune)

    print(f"Best params: {meta['best_params']}")
    print(f"Backtest ({meta['backtest_origins']} origins x {meta['horizon']} days, after {meta['backtest_cutoff']}):")
    for name, m in meta["backtest"].items():
        print(f"  {name:15s} MSE={m['mse']:>10.2f}  RMSE={m['rmse']:>7.2f}  MAE={m['mae']:>7.2f}")
    print(f"Default strategy: {meta['default_strategy']}")
    print(f"Top features: {list(meta['feature_importance'])[:5]}")
    print(f"Artifacts written to {args.out}/")


if __name__ == "__main__":
    main()
