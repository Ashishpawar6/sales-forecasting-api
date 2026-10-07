from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

Strategy = Literal["recursive", "direct", "combined"]


class ForecastRequest(BaseModel):
    store: int
    item: int
    horizon: int = Field(default=7, ge=1, description="Days to forecast; at most the trained horizon")
    strategy: Strategy | None = Field(default=None, description="Defaults to the best strategy from the backtest")
    promo_dates: list[date] = Field(default_factory=list, description="Planned promotion days inside the forecast window")


class ForecastPoint(BaseModel):
    date: date
    forecast: float


class ForecastResponse(BaseModel):
    store: int
    item: int
    strategy: Strategy
    origin_date: date = Field(description="Last day of observed history; forecasts start the day after")
    horizon: int
    forecasts: list[ForecastPoint]


class SeriesKey(BaseModel):
    store: int
    item: int
