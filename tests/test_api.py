import pandas as pd
import pytest

from tests.conftest import HORIZON


def forecast(client, **overrides):
    body = {"store": 1, "item": 1, "horizon": HORIZON}
    body.update(overrides)
    return client.post("/forecast", json=body)


def test_health_reports_model_loaded(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_model_info_exposes_backtest_and_importance(client):
    info = client.get("/model/info").json()
    assert {"recursive", "direct", "combined", "seasonal_naive"} <= set(info["backtest"])
    assert info["feature_importance"] and info["default_strategy"] in {"recursive", "direct", "combined"}


def test_series_lists_all_store_item_pairs(client):
    series = client.get("/series").json()
    assert {(s["store"], s["item"]) for s in series} == {(1, 1), (1, 2), (2, 1), (2, 2)}


def test_forecast_returns_consecutive_days_after_history(client, small_df):
    resp = forecast(client)
    assert resp.status_code == 200
    body = resp.json()
    last = small_df["date"].max()
    assert body["origin_date"] == last.strftime("%Y-%m-%d")
    dates = [p["date"] for p in body["forecasts"]]
    assert dates == [(last + pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, HORIZON + 1)]
    assert all(p["forecast"] >= 0 for p in body["forecasts"])


@pytest.mark.parametrize("strategy", ["recursive", "direct", "combined"])
def test_every_strategy_can_be_requested(client, strategy):
    resp = forecast(client, strategy=strategy)
    assert resp.status_code == 200 and resp.json()["strategy"] == strategy


def test_default_strategy_comes_from_the_backtest(client, metadata):
    assert forecast(client).json()["strategy"] == metadata["default_strategy"]


def test_planned_promotion_raises_the_forecast(client):
    base = forecast(client, strategy="direct").json()["forecasts"]
    promo_day = base[2]["date"]
    promoted = forecast(client, strategy="direct", promo_dates=[promo_day]).json()["forecasts"]
    assert promoted[2]["forecast"] > base[2]["forecast"]
    assert promoted[0]["forecast"] == base[0]["forecast"]  # days without a promo are unchanged


def test_unknown_series_is_404(client):
    assert forecast(client, store=99).status_code == 404


def test_horizon_beyond_trained_horizon_is_422(client):
    assert forecast(client, horizon=HORIZON + 1).status_code == 422


@pytest.mark.parametrize("bad", [{"horizon": 0}, {"strategy": "magic"}, {"store": "abc"}, {"promo_dates": ["not-a-date"]}])
def test_invalid_payloads_are_422(client, bad):
    assert forecast(client, **bad).status_code == 422


def test_promo_date_outside_window_is_422(client):
    assert forecast(client, promo_dates=["2020-01-01"]).status_code == 422
