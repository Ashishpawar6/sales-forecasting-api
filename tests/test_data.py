import pandas as pd
import pytest

from forecasting.data import generate_sales, validate_sales


def test_generator_is_deterministic():
    a = generate_sales(end="2022-03-01", seed=1)
    b = generate_sales(end="2022-03-01", seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(generate_sales(end="2022-03-01", seed=2))


def test_generated_data_is_valid_and_one_row_per_day_per_series():
    df = validate_sales(generate_sales(end="2022-06-30", n_stores=2, n_items=2))
    assert df[["date", "store", "item"]].duplicated().sum() == 0
    assert df.groupby(["store", "item"]).size().nunique() == 1
    assert (df["sales"] >= 0).all()


def test_promotions_lift_sales():
    df = generate_sales()
    assert df[df.promo == 1]["sales"].mean() > df[df.promo == 0]["sales"].mean()


def test_missing_column_is_rejected():
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_sales(pd.DataFrame({"date": ["2022-01-01"], "sales": [1]}))


def test_negative_sales_are_rejected():
    df = generate_sales(end="2022-01-10", n_stores=1, n_items=1)
    df.loc[0, "sales"] = -1
    with pytest.raises(ValueError, match="non-negative"):
        validate_sales(df)


def test_gaps_in_a_series_are_rejected():
    df = generate_sales(end="2022-01-20", n_stores=1, n_items=1).drop(index=5)
    with pytest.raises(ValueError, match="gaps"):
        validate_sales(df)


def test_duplicates_are_rejected():
    df = generate_sales(end="2022-01-10", n_stores=1, n_items=1)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_sales(pd.concat([df, df.iloc[[0]]]))
