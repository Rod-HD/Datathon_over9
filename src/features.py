"""Feature engineering for the daily Revenue / COGS forecasting task.

Since every external table ends 2022-12-31, test-period features can only come
from the calendar. We still derive rich features from the *training span* so
models can learn historical seasonality — but the feature matrix for inference
is reconstructed purely from Date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------- calendar ---------- #

def calendar_features(dates: pd.Series) -> pd.DataFrame:
    dates = pd.to_datetime(dates)
    out = pd.DataFrame(index=dates.index)
    out["year"] = dates.dt.year
    out["month"] = dates.dt.month
    out["day"] = dates.dt.day
    out["day_of_week"] = dates.dt.dayofweek
    out["day_of_year"] = dates.dt.dayofyear
    out["week_of_year"] = dates.dt.isocalendar().week.astype(int)
    out["quarter"] = dates.dt.quarter
    out["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)
    out["is_month_start"] = dates.dt.is_month_start.astype(int)
    out["is_month_end"] = dates.dt.is_month_end.astype(int)
    out["is_quarter_end"] = dates.dt.is_quarter_end.astype(int)
    out["is_year_end"] = dates.dt.is_year_end.astype(int)
    out["days_since_2012"] = (dates - pd.Timestamp("2012-07-04")).dt.days
    return out


def fourier_features(dates: pd.Series, period: float, order: int, prefix: str) -> pd.DataFrame:
    """Fourier terms for a given seasonal period."""
    dates = pd.to_datetime(dates)
    t = (dates - pd.Timestamp("2012-07-04")).dt.days.to_numpy()
    out = {}
    for k in range(1, order + 1):
        angle = 2 * np.pi * k * t / period
        out[f"{prefix}_sin{k}"] = np.sin(angle)
        out[f"{prefix}_cos{k}"] = np.cos(angle)
    return pd.DataFrame(out, index=dates.index)


def calendar_and_fourier(dates: pd.Series) -> pd.DataFrame:
    cal = calendar_features(dates)
    f_year = fourier_features(dates, period=365.25, order=6, prefix="y")
    f_week = fourier_features(dates, period=7, order=3, prefix="w")
    f_month = fourier_features(dates, period=30.44, order=3, prefix="m")
    return pd.concat([cal, f_year, f_week, f_month], axis=1)


# ---------- target-derived features (train-only) ---------- #

def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lags: list[int],
) -> pd.DataFrame:
    """Add lag-N features. NaN-filled for the first N rows."""
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    windows: list[int],
    min_lag: int = 548,
) -> pd.DataFrame:
    """Rolling mean/std — shifted by `min_lag` so they only use pre-test data.

    `min_lag=548` means the feature value at day t uses values from t - 548 -
    window back, guaranteeing the entire window sits before 2023-01-01 when
    predicting the test horizon.
    """
    df = df.copy()
    shifted = df[target_col].shift(min_lag)
    for w in windows:
        df[f"{target_col}_rmean{w}_lag{min_lag}"] = shifted.rolling(w, min_periods=1).mean()
        df[f"{target_col}_rstd{w}_lag{min_lag}"] = shifted.rolling(w, min_periods=1).std()
    return df


# ---------- holidays discovered from train ---------- #

def discover_anomaly_dates(
    sales: pd.DataFrame,
    target: str = "Revenue",
    z_threshold: float = 2.5,
) -> pd.DataFrame:
    """Flag days whose revenue deviates strongly from its 30-day rolling baseline.

    Output: DataFrame of (month, day) pairs that recur as anomalies across years.
    Used as a proxy for Vietnamese holidays (Tet, 30/4, 2/9, Black Friday...)
    without needing an external calendar — keeps us compliant with the no-
    external-data rule.
    """
    s = sales.set_index("Date")[target]
    baseline = s.rolling(30, center=True, min_periods=10).mean()
    z = (s - baseline) / s.rolling(30, center=True, min_periods=10).std()

    flagged = pd.DataFrame({"z": z}).dropna()
    flagged["month"] = flagged.index.month
    flagged["day"] = flagged.index.day
    flagged["is_anomaly"] = (flagged["z"].abs() > z_threshold).astype(int)

    hits = (
        flagged.groupby(["month", "day"])["is_anomaly"].sum().reset_index()
        .query("is_anomaly >= 3")
        .rename(columns={"is_anomaly": "years_flagged"})
    )
    return hits


def add_anomaly_flag(dates: pd.Series, anomaly_md: pd.DataFrame) -> pd.Series:
    d = pd.to_datetime(dates)
    key = d.dt.month * 100 + d.dt.day
    anomaly_keys = set((anomaly_md["month"] * 100 + anomaly_md["day"]).tolist())
    return key.isin(anomaly_keys).astype(int)


# ---------- external-table aggregates (train-side only, projected to test) ---------- #

def promo_active_count_daily(promotions: pd.DataFrame, dates: pd.Series) -> pd.Series:
    """For each date, count how many promotions were active."""
    d = pd.to_datetime(dates).reset_index(drop=True)
    p = promotions.copy()
    p["start_date"] = pd.to_datetime(p["start_date"])
    p["end_date"] = pd.to_datetime(p["end_date"])

    counts = np.zeros(len(d), dtype=int)
    d_values = d.to_numpy().astype("datetime64[ns]")
    for _, row in p.iterrows():
        mask = (d_values >= np.datetime64(row["start_date"])) & (
            d_values <= np.datetime64(row["end_date"])
        )
        counts += mask.astype(int)
    return pd.Series(counts, index=d.index, name="promo_active_count")


# ---------- high-level builder ---------- #

def build_feature_frame(
    dates: pd.Series,
    sales: pd.DataFrame | None = None,
    promotions: pd.DataFrame | None = None,
    target_col: str = "Revenue",
    lag_list: tuple[int, ...] = (548, 729, 1094),
    roll_windows: tuple[int, ...] = (7, 28, 90),
    anomaly_md: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Construct the full feature matrix.

    Parameters
    ----------
    dates
        Index dates (train + test concatenated).
    sales
        DataFrame with Date + target column. Used for lags on the train side;
        test-side lag values will be NaN unless the lag >= 548 days (so their
        reference date is in the training period).
    promotions
        Optional table of promotions. Active-promo count is projected into test
        only if promo end_date >= test start.
    """
    d = pd.to_datetime(dates).reset_index(drop=True)
    frame = calendar_and_fourier(d)
    frame["Date"] = d.values

    if sales is not None:
        sales_idx = sales.set_index("Date")[target_col]
        merged = pd.DataFrame({"Date": d.values})
        merged[target_col] = merged["Date"].map(sales_idx)
        merged = add_lag_features(merged, target_col, list(lag_list))
        merged = add_rolling_features(merged, target_col, list(roll_windows), min_lag=min(lag_list))
        frame = frame.join(
            merged.drop(columns=["Date", target_col]).reset_index(drop=True),
            how="left",
        )

    if promotions is not None:
        frame["promo_active_count"] = promo_active_count_daily(promotions, d).values

    if anomaly_md is not None and len(anomaly_md) > 0:
        frame["is_anomaly_day"] = add_anomaly_flag(d, anomaly_md).values

    return frame
