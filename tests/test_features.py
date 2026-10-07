import numpy as np
import pandas as pd
import pytest

from forecasting.config import FEATURES, LAGS, WINDOWS
from forecasting.data import generate_sales
from forecasting.features import inference_row, training_frame


@pytest.fixture(scope="module")
def df():
    return generate_sales(end="2022-06-30", n_stores=2, n_items=2, seed=3)


@pytest.mark.parametrize("h", [1, 5, 14])
def test_training_features_match_serving_features(df, h):
    """The vectorised training features must equal what the API computes from raw history."""
    frame = training_frame(df, h, [1, 2], [1, 2])
    sample = frame.sample(40, random_state=0)
    for _, row in sample.iterrows():
        g = df[(df.store == row["store"]) & (df.item == row["item"])].sort_values("date").reset_index(drop=True)
        target_idx = g.index[g["date"] == row["date"]][0]
        history = g.loc[: target_idx - h, "sales"].to_numpy(dtype=float)  # sales up to the origin day only
        served = inference_row(history, row["date"], int(row["promo"]), int(row["store"]), int(row["item"]))
        for name in FEATURES:
            assert float(served[name]) == pytest.approx(float(row[name])), name


def test_no_feature_uses_data_after_the_origin(df):
    """Changing sales on/after the target date must not change that row's features."""
    h = 3
    g = df[(df.store == 1) & (df.item == 1)].reset_index(drop=True)
    target_idx = 100
    before = training_frame(g, h, [1], [1])
    altered = g.copy()
    altered.loc[target_idx - h + 1:, "sales"] = 10_000  # everything after the origin day
    after = training_frame(altered, h, [1], [1])
    date = g.loc[target_idx, "date"]
    a, b = before[before.date == date].iloc[0], after[after.date == date].iloc[0]
    for name in FEATURES:
        assert float(a[name]) == float(b[name]), name
    assert a["target"] != b["target"]  # only the label changed


def test_rows_without_full_history_are_dropped(df):
    frame = training_frame(df, 7, [1, 2], [1, 2])
    first_usable = df.date.min() + pd.Timedelta(days=7 + max(max(LAGS), max(WINDOWS) - 1))
    assert frame["date"].min() == first_usable
    assert not frame[FEATURES].isna().any().any()


def test_rolling_mean_is_over_the_last_window_days():
    history = np.arange(1.0, 61.0)
    row = inference_row(history, pd.Timestamp("2022-03-01"), 0, 1, 1)
    assert row["roll_mean_7"] == pytest.approx(history[-7:].mean())
    assert row["roll_mean_28"] == pytest.approx(history[-28:].mean())
    assert row["lag_0"] == 60.0 and row["lag_27"] == 33.0
