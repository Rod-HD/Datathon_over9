"""Seasonal baseline with simple growth estimators."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _annual_totals(sales: pd.DataFrame, target_col: str) -> pd.Series:
    df = sales.copy()
    df["year"] = df["Date"].dt.year
    counts = df.groupby("year")["Date"].nunique()
    totals = df.groupby("year")[target_col].sum()
    full_years = counts[counts >= 360].index
    return totals.loc[totals.index.isin(full_years)]


def _linear_growth(annual: pd.Series, years: int) -> float:
    tail = annual.tail(years)
    if len(tail) < 2:
        return 1.0
    x = tail.index.values.astype(float)
    y = np.log(tail.values.astype(float))
    slope = np.polyfit(x, y, 1)[0]
    return float(np.exp(slope))


def _growth_factor(annual: pd.Series, mode: str) -> float:
    full_years = annual.loc[2013:]
    if len(full_years) < 2:
        return 1.0

    yoy = full_years.pct_change().dropna()
    if mode == "geo_full":
        return float((1 + yoy).prod() ** (1 / len(yoy)))
    if mode == "geo_last3":
        tail = yoy.tail(3)
        return float((1 + tail).prod() ** (1 / len(tail)))
    if mode == "linear_last2":
        return _linear_growth(full_years, 2)
    if mode == "linear_last3":
        return _linear_growth(full_years, 3)
    if mode == "linear_last5":
        return _linear_growth(full_years, 5)
    if mode == "flat":
        return 1.0
    raise ValueError(f"Unknown growth_mode: {mode}")


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
    normalized = df[target_col] / annual_means
    if log_transform:
        normalized = np.log1p(normalized.clip(lower=0))

    seasonal = (
        pd.DataFrame({"month": df["month"].values, "day": df["day"].values, "norm": normalized.values})
        .groupby(["month", "day"])["norm"]
        .mean()
        .reset_index()
    )
    if log_transform:
        seasonal["norm"] = np.expm1(seasonal["norm"])

    return seasonal.rename(columns={"norm": f"{target_col}_norm"})


def fit_and_predict(
    sales: pd.DataFrame,
    test_dates: pd.Series,
    target_col: str = "Revenue",
    growth_mode: str = "geo_full",
    season_window: int | None = None,
    log_transform: bool = False,
) -> pd.Series:
    df = sales.copy()
    df["Date"] = pd.to_datetime(df["Date"])
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
    norm_col = f"{target_col}_norm"
    test[norm_col] = test[norm_col].fillna(1.0)
    prediction = base * (growth ** test["years_ahead"]) * test[norm_col]
    return prediction.round(2)
