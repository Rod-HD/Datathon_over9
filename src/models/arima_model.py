"""Hybrid ARIMA forecaster.

The model forecasts a linear_last3 seasonal baseline, then fits ARIMA(p,0,q)
on recent residuals. This keeps the long-horizon growth trend from the
baseline while adding a short-term residual correction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA as SM_ARIMA

from src.models.baseline import fit_and_predict as baseline_pred


BEST_ORDER: dict[str, tuple[int, int, int]] = {
    "Revenue": (2, 0, 2),
    "COGS": (1, 0, 3),
}
RESIDUAL_WINDOW_YEARS = 3


@dataclass
class ARIMAForecaster:
    target_col: str
    order: tuple[int, int, int] | None = None
    residual_window_years: int = 0

    _arima_result: object = field(default=None, init=False, repr=False)
    _last_date: pd.Timestamp | None = field(default=None, init=False, repr=False)
    _train_df: pd.DataFrame | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.order is None:
            self.order = BEST_ORDER.get(self.target_col, (2, 0, 2))

    def fit(self, sales: pd.DataFrame) -> None:
        train = sales.sort_values("Date").copy()
        train["Date"] = pd.to_datetime(train["Date"])
        self._train_df = train.copy()

        baseline = baseline_pred(train, train["Date"], self.target_col, "linear_last3", 6).values
        residuals = train[self.target_col].values - baseline

        window = self.residual_window_years or RESIDUAL_WINDOW_YEARS
        max_year = train["Date"].dt.year.max()
        recent_mask = train["Date"].dt.year >= (max_year - window + 1)

        series = pd.Series(
            residuals[recent_mask],
            index=pd.DatetimeIndex(train.loc[recent_mask, "Date"].values),
        )
        series = series.asfreq("D").interpolate("linear").ffill().bfill()

        self._arima_result = SM_ARIMA(series, order=self.order, trend="n").fit()
        self._last_date = series.index[-1]

    def predict(self, dates: pd.Series) -> pd.Series:
        if self._arima_result is None or self._train_df is None or self._last_date is None:
            raise RuntimeError("Call fit() first.")

        dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
        trend_forecast = baseline_pred(
            self._train_df,
            dates,
            self.target_col,
            "linear_last3",
            6,
        ).values

        horizon = (dates.max() - self._last_date).days
        if horizon <= 0:
            raise ValueError("Target dates must be after the training end date.")

        residual_forecast = self._arima_result.forecast(steps=horizon)
        forecast_index = pd.date_range(
            self._last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )
        residuals = (
            pd.Series(np.asarray(residual_forecast), index=forecast_index)
            .reindex(dates.values)
            .fillna(0.0)
            .values
        )

        output = np.clip(trend_forecast + residuals, 0.0, None)
        return pd.Series(output, index=dates.index)


def fit_and_predict(
    train_sales: pd.DataFrame,
    dates: pd.Series,
    target: str = "Revenue",
    order: tuple[int, int, int] | None = None,
) -> pd.Series:
    model = ARIMAForecaster(target_col=target, order=order)
    model.fit(train_sales[["Date", target]])
    return model.predict(dates)
