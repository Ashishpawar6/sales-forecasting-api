"""Synthetic daily sales generator and CSV loading/validation.

Real data can be used instead: any CSV with the columns date, store, item, sales
(and optionally promo) works, as long as every store/item series is a continuous daily series.
"""
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["date", "store", "item", "sales"]


def generate_sales(
    start: str = "2022-01-01",
    end: str = "2024-12-31",
    n_stores: int = 4,
    n_items: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Daily sales with trend, weekly + yearly seasonality, promotions and noise."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)
    t = np.arange(n)
    weekly = np.array([0.85, 0.85, 0.9, 0.95, 1.1, 1.35, 1.2])[dates.dayofweek]  # Mon..Sun
    yearly = 1 + 0.2 * np.sin(2 * np.pi * (dates.dayofyear - 80) / 365.25)

    frames = []
    for store in range(1, n_stores + 1):
        for item in range(1, n_items + 1):
            base = 40 * (0.7 + 0.2 * store) * (0.6 + 0.3 * item)
            trend = 1 + 0.15 * t / n * (0.5 + rng.random())
            promo = (rng.random(n) < 0.1).astype(int)
            level = base * trend * weekly * yearly * (1 + 0.3 * promo)
            sales = np.maximum(0, np.round(level * rng.normal(1, 0.08, n)))
            frames.append(pd.DataFrame({"date": dates, "store": store, "item": item, "promo": promo, "sales": sales}))
    return pd.concat(frames, ignore_index=True)


def load_sales(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return validate_sales(df)


def validate_sales(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "promo" not in df.columns:
        df["promo"] = 0
    if df[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Data contains missing values in date/store/item/sales")
    if (df["sales"] < 0).any():
        raise ValueError("Sales must be non-negative")
    if df.duplicated(["date", "store", "item"]).any():
        raise ValueError("Duplicate (date, store, item) rows found")
    df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)
    for (store, item), g in df.groupby(["store", "item"]):
        expected = (g["date"].iloc[-1] - g["date"].iloc[0]).days + 1
        if len(g) != expected:
            raise ValueError(f"Series store={store} item={item} has gaps; daily data must be continuous")
    return df


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Generate a synthetic sales CSV")
    parser.add_argument("--out", default="data/sales.csv")
    args = parser.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frame = generate_sales()
    frame.to_csv(args.out, index=False)
    print(f"Wrote {len(frame):,} rows to {args.out}")
