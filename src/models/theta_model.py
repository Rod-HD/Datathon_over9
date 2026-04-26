"""Theta method — M3/M4 winning simple forecaster.

Uses statsmodels ThetaModel, applied on log1p(revenue). Also adds a 7-day
seasonal multiplicative adjustment on top because the native ThetaModel
only supports single seasonality.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.forecasting.theta import ThetaModel


@dataclass
class ThetaForecaster:
    target_col: str
    log_target: bool = True
    period: int = 365
    deseasonalize: bool = True
    weekly_dow_adjust: bool = True
    model_ = None
    _last_date: pd.Timestamp | None = None
    _dow_profile = None

    def fit(self, sales: pd.DataFrame) -> None:
        s = sales.sort_values("Date").copy()
        s["Date"] = pd.to_datetime(s["Date"])
        y_raw = s[self.target_col].astype(float).clip(lower=0).values

        if self.weekly_dow_adjust:
            # Daily detrended DoW multiplier from full train
            dow = s["Date"].dt.dayofweek.values
            global_mean = y_raw.mean() if y_raw.mean() > 0 else 1.0
            dow_factor = np.ones(7)
            for k in range(7):
                mask = dow == k
                if mask.sum() > 0:
                    dow_factor[k] = y_raw[mask].mean() / global_mean
            self._dow_profile = dow_factor
            y_adj = y_raw / dow_factor[dow]
        else:
            self._dow_profile = None
            y_adj = y_raw

        y_in = np.log1p(y_adj) if self.log_target else y_adj
        y_series = pd.Series(y_in, index=pd.to_datetime(s["Date"]))
        # ThetaModel needs a frequency
        y_series = y_series.asfreq("D")
        y_series = y_series.interpolate(method="linear").ffill().bfill()

        self.model_ = ThetaModel(
            y_series,
            period=self.period,
            deseasonalize=self.deseasonalize,
            use_test=False,
            method="auto",
        ).fit()
        self._last_date = y_series.index[-1]

    def predict(self, dates: pd.Series) -> pd.Series:
        if self.model_ is None:
            raise RuntimeError("Call fit() first")
        dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
        horizon = (dates.max() - self._last_date).days
        if horizon <= 0:
            raise ValueError("target dates must be after train_end")
        fc = self.model_.forecast(steps=horizon)
        full_index = pd.date_range(self._last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
        series = pd.Series(np.asarray(fc), index=full_index)
        out = series.reindex(dates.values)
        if self.log_target:
            out = np.expm1(out)
        # Re-apply DoW multiplier for the predicted dates
        if self._dow_profile is not None:
            dow = dates.dt.dayofweek.values
            out = out.values * self._dow_profile[dow]
            out = pd.Series(out, index=dates.index)
        else:
            out.index = dates.index
        return out.clip(lower=0.0)


def fit_and_predict(train_sales: pd.DataFrame,
                    dates: pd.Series,
                    target: str = "Revenue",
                    **kwargs) -> pd.Series:
    m = ThetaForecaster(target_col=target, **kwargs)
    m.fit(train_sales[["Date", target]])
    return m.predict(dates)
