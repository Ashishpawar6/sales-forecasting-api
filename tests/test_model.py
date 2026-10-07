import numpy as np
import pandas as pd
import pytest

from forecasting.evaluate import metrics, seasonal_naive
from forecasting.model import Forecaster
from tests.conftest import ARTIFACTS, HORIZON


def test_pipeline_writes_all_artifacts(metadata):
    assert (ARTIFACTS / "metadata.json").exists() and (ARTIFACTS / "history.csv").exists()
    assert len(list(ARTIFACTS.glob("model_h*.txt"))) == HORIZON
    assert metadata["horizon"] == HORIZON


def test_models_beat_the_seasonal_naive_baseline(metadata):
    scores = metadata["backtest"]
    for strategy in ("recursive", "direct", "combined"):
        assert scores[strategy]["rmse"] < scores["seasonal_naive"]["rmse"], strategy


def test_default_strategy_is_the_best_in_the_backtest(metadata):
    scores = metadata["backtest"]
    best = min(("recursive", "direct", "combined"), key=lambda s: scores[s]["rmse"])
    assert metadata["default_strategy"] == best


def test_feature_importance_is_normalised(metadata):
    importance = metadata["feature_importance"]
    assert sum(importance.values()) == pytest.approx(1.0, abs=0.01)
    assert importance == dict(sorted(importance.items(), key=lambda kv: -kv[1]))


def test_combined_is_the_mean_of_recursive_and_direct(small_df, metadata):
    model = Forecaster.load(ARTIFACTS)
    g = small_df[(small_df.store == 1) & (small_df.item == 1)].sort_values("date")
    history, origin = g["sales"].to_numpy(dtype=float), g["date"].max()
    rec = model.predict(history, origin, 1, 1, HORIZON, "recursive")
    dir_ = model.predict(history, origin, 1, 1, HORIZON, "direct")
    comb = model.predict(history, origin, 1, 1, HORIZON, "combined")
    np.testing.assert_allclose(comb, (rec + dir_) / 2)
    assert (comb >= 0).all() and len(comb) == HORIZON


def test_predict_validates_inputs(small_df, metadata):
    model = Forecaster.load(ARTIFACTS)
    history = small_df[(small_df.store == 1) & (small_df.item == 1)]["sales"].to_numpy(dtype=float)
    origin = pd.Timestamp("2024-12-31")
    with pytest.raises(ValueError, match="strategy"):
        model.predict(history, origin, 1, 1, 3, "magic")
    with pytest.raises(ValueError, match="exceeds"):
        model.predict(history, origin, 1, 1, HORIZON + 1, "direct")
    with pytest.raises(ValueError, match="history"):
        model.predict(history[:10], origin, 1, 1, 3, "direct")


def test_seasonal_naive_repeats_the_last_week():
    history = np.arange(1.0, 15.0)  # last week: 8..14
    np.testing.assert_array_equal(seasonal_naive(history, 9), [8, 9, 10, 11, 12, 13, 14, 8, 9])


def test_metrics():
    m = metrics(np.array([10.0, 20.0]), np.array([12.0, 17.0]))
    assert m["mse"] == pytest.approx(6.5) and m["mae"] == pytest.approx(2.5)
    assert m["rmse"] == pytest.approx(6.5**0.5, abs=1e-3)
