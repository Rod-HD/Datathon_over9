"""Prophet wrapper for Revenue / COGS forecasting.

Uses train-discovered anomaly dates as custom holidays.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from prophet import Prophet


def _anomaly_to_holidays(anomaly_md: pd.DataFrame, years=range(2012, 2025)) -> pd.DataFrame:
    """Expand (month, day) pairs into a Prophet-compatible holidays frame."""
    rows = []
    for _, r in anomaly_md.iterrows():
        for y in years:
            try:
                d = pd.Timestamp(year=y, month=int(r["month"]), day=int(r["day"]))
                rows.append({"holiday": "auto_anomaly", "ds": d, "lower_window": -1, "upper_window": 1})
            except ValueError:
                continue
    return pd.DataFrame(rows)


@dataclass
class ProphetForecaster:
    target_col: str
    anomaly_md: pd.DataFrame | None = None
    growth: str = "linear"
    changepoint_prior_scale: float = 0.05
    seasonality_prior_scale: float = 10.0
    seasonality_mode: str = "multiplicative"
    yearly_seasonality: int = 20
    weekly_seasonality: int = 7
    changepoint_range: float = 0.8
    holiday_lower: int = -1
    holiday_upper: int = 1
    log_target: bool = True
    extra_regressors: list[str] = field(default_factory=list)
    model_: Prophet | None = None
    # Optional regressor values per date (dict: date -> {regressor: value})
    _regressor_lookup: dict = field(default_factory=dict)

    def fit(self, sales: pd.DataFrame, regressor_frame: pd.DataFrame | None = None) -> None:
        df = sales.rename(columns={"Date": "ds", self.target_col: "y"})[["ds", "y"]].copy()
        df["ds"] = pd.to_datetime(df["ds"])
        if self.log_target:
            df["y"] = np.log1p(df["y"].clip(lower=0))

        holidays = None
        if self.anomaly_md is not None and len(self.anomaly_md) > 0:
            holidays = _anomaly_to_holidays(self.anomaly_md)
            if holidays is not None and len(holidays) > 0:
                holidays["lower_window"] = self.holiday_lower
                holidays["upper_window"] = self.holiday_upper

        self.model_ = Prophet(
            growth=self.growth,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=False,
            changepoint_range=self.changepoint_range,
            holidays=holidays,
        )

        if self.extra_regressors and regressor_frame is not None:
            rf = regressor_frame.rename(columns={"Date": "ds"}).copy()
            rf["ds"] = pd.to_datetime(rf["ds"])
            keep = ["ds"] + [r for r in self.extra_regressors if r in rf.columns]
            rf = rf[keep]
            df = df.merge(rf, on="ds", how="left")
            for r in self.extra_regressors:
                if r in df.columns:
                    df[r] = df[r].fillna(0.0)
                    self.model_.add_regressor(r)
            # store regressor values for predict
            self._regressor_lookup = rf.set_index("ds").to_dict(orient="index")

        self.model_.fit(df)

    def predict(self, dates: pd.Series) -> pd.Series:
        assert self.model_ is not None
        future = pd.DataFrame({"ds": pd.to_datetime(dates).values})
        if self.extra_regressors:
            for r in self.extra_regressors:
                future[r] = future["ds"].map(
                    lambda d: self._regressor_lookup.get(d, {}).get(r, 0.0)
                )
        forecast = self.model_.predict(future)
        yhat = forecast["yhat"].values
        if self.log_target:
            yhat = np.expm1(yhat)
        return pd.Series(np.clip(yhat, 0, None), index=future.index)
