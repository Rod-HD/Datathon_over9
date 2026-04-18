"""Seasonal-naive baseline with pluggable growth estimators.

Prediction = base_daily * growth_factor(year) * seasonal_profile(month, day)

Growth modes:
  - geo_full:     geometric mean of ALL YoY ratios (default, original).
  - geo_last3:    geometric mean of last 3 complete YoY ratios.
  - linear_last3: linear regression of log(annual_total) on year using the
                  last 3 complete years, extrapolated exponentially.
  - linear_last5: same but 5 years.
  - flat:         growth = 1.0 (base stays constant).

`season_window` optionally restricts the seasonal profile to the last N
complete years, reducing COVID / pre-2015 noise.

`log_transform` uses log1p(daily / annual_mean) for the seasonal profile,
which dampens single-day spikes (Jan 1 effect).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _annual_totals(sales: pd.DataFrame, target_col: str, full_year_only: bool = True) -> pd.Series:
    """Sum target per year. If `full_year_only`, drops years with <360 observed
    days — a partial last year (e.g. train ending mid-year) would otherwise
    collapse the base value."""
    df = sales.copy()
    df["year"] = df["Date"].dt.year
    counts = df.groupby("year")["Date"].nunique()
    totals = df.groupby("year")[target_col].sum()
    if full_year_only:
        keep = counts[counts >= 360].index
        totals = totals.loc[totals.index.isin(keep)]
    return totals


def _growth_factor(annual: pd.Series, mode: str) -> float:
    full_years = annual.loc[2013:]
    if len(full_years) < 2:
        return 1.0
    yoy = full_years.pct_change().dropna()

    if mode == "geo_full":
        return float((1 + yoy).prod() ** (1 / len(yoy)))
    if mode == "flat":
        return 1.0
    if mode == "geo_last3":
        tail = yoy.tail(3)
        return float((1 + tail).prod() ** (1 / len(tail)))
    if mode == "linear_last3":
        return _linear_growth(full_years, k=3)
    if mode == "linear_last5":
        return _linear_growth(full_years, k=5)
    raise ValueError(f"Unknown growth_mode: {mode}")


def _linear_growth(annual: pd.Series, k: int) -> float:
    tail = annual.tail(k)
    if len(tail) < 2:
        return 1.0
    x = tail.index.values.astype(float)
    y = np.log(tail.values.astype(float))
    slope = np.polyfit(x, y, 1)[0]
    return float(np.exp(slope))


def _seasonal_profile(
    sales: pd.DataFrame,
    target_col: str,
    season_window: int | None,
    log_transform: bool,
    base_year: int,
) -> pd.DataFrame:
    df = sales.copy()
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day

    if season_window is not None:
        df = df[df["year"] > base_year - season_window]

    annual_means = df.groupby("year")[target_col].transform("mean")
    norm = df[target_col] / annual_means
    if log_transform:
        norm = np.log1p(norm.clip(lower=0))

    seasonal = (
        pd.DataFrame({"month": df["month"].values, "day": df["day"].values, "norm": norm.values})
        .groupby(["month", "day"])["norm"].mean()
        .reset_index()
    )

    if log_transform:
        seasonal["norm"] = np.expm1(seasonal["norm"])

    seasonal = seasonal.rename(columns={"norm": f"{target_col}_norm"})
    return seasonal


def fit_and_predict(
    sales: pd.DataFrame,
    test_dates: pd.Series,
    target_col: str = "Revenue",
    growth_mode: str = "geo_full",
    season_window: int | None = None,
    log_transform: bool = False,
) -> pd.Series:
    df = sales.copy()
    df["year"] = df["Date"].dt.year

    annual = _annual_totals(df, target_col)
    base_year = int(annual.index.max())
    growth = _growth_factor(annual, growth_mode)

    seasonal = _seasonal_profile(df, target_col, season_window, log_transform, base_year)

    base = float(annual.loc[base_year] / 365.0)

    test = pd.DataFrame({"Date": pd.to_datetime(test_dates)})
    test["month"] = test["Date"].dt.month
    test["day"] = test["Date"].dt.day
    test["years_ahead"] = test["Date"].dt.year - base_year

    test = test.merge(seasonal, on=["month", "day"], how="left")
    test[f"{target_col}_norm"] = test[f"{target_col}_norm"].fillna(1.0)

    test["pred"] = base * (growth ** test["years_ahead"]) * test[f"{target_col}_norm"]
    return test["pred"].round(2)


def prophet_trend_anchor(
    sales: pd.DataFrame,
    target_col: str,
    test_dates: pd.Series,
) -> pd.Series:
    """Anchor-based alternative: use Prophet's learned trend component as the
    per-day base (seasonality + holidays stripped), then multiply by the
    seasonal profile. Returns a Series of anchor values (one per test date).

    Prophet is fit in a trend-only mode (seasonality priors near zero), so
    its `yhat` is essentially the trend baseline.
    """
    from prophet import Prophet  # local import — keeps baseline dep-free otherwise

    df = sales.rename(columns={"Date": "ds", target_col: "y"})[["ds", "y"]].copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = np.log1p(df["y"].clip(lower=0))

    m = Prophet(
        growth="linear",
        changepoint_prior_scale=0.25,
        changepoint_range=0.95,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_prior_scale=0.01,
    )
    m.fit(df)
    future = pd.DataFrame({"ds": pd.to_datetime(test_dates).values})
    fc = m.predict(future)
    anchor = np.expm1(fc["trend"].values)
    return pd.Series(np.clip(anchor, 0, None), index=pd.RangeIndex(len(anchor)))


def predict_with_prophet_trend(
    sales: pd.DataFrame,
    test_dates: pd.Series,
    target_col: str = "Revenue",
    season_window: int | None = None,
    log_transform: bool = False,
) -> pd.Series:
    """Predict = prophet_trend(date) * seasonal_profile(month, day)."""
    anchor = prophet_trend_anchor(sales, target_col, test_dates)
    # seasonal profile needs base_year for season_window cut
    annual = _annual_totals(sales, target_col)
    base_year = int(annual.index.max())
    seasonal = _seasonal_profile(sales, target_col, season_window, log_transform, base_year)

    test = pd.DataFrame({"Date": pd.to_datetime(test_dates)})
    test["month"] = test["Date"].dt.month
    test["day"] = test["Date"].dt.day
    test = test.merge(seasonal, on=["month", "day"], how="left")
    test[f"{target_col}_norm"] = test[f"{target_col}_norm"].fillna(1.0)
    return (anchor.values * test[f"{target_col}_norm"].values).round(2)
