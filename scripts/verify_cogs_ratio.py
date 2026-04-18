"""Verify if COGS/Revenue ratio is stable enough to use COGS = Revenue * ratio.

Tests:
1. grand mean ratio + grand std/mean (target: std/mean < 0.05)
2. Ratio drift by year — is mean ratio shifting across years?
3. Ratio by day-of-week and month — any strong pattern?
4. Simulate COGS = Revenue * ratio_profile on yearly_2022 and compare vs independent
   COGS model MAE.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.data_loader import load_sales
from src.evaluation import mae as _mae


def main() -> None:
    sales = load_sales().copy()
    sales["Date"] = pd.to_datetime(sales["Date"])
    sales["ratio"] = sales["COGS"] / sales["Revenue"]
    sales = sales[np.isfinite(sales["ratio"])].copy()

    grand_mean = sales["ratio"].mean()
    grand_std = sales["ratio"].std()
    print(f"Grand ratio mean={grand_mean:.4f}  std={grand_std:.4f}  std/mean={grand_std/grand_mean:.4f}")

    print("\nYear-level ratio:")
    by_year = sales.assign(year=sales["Date"].dt.year).groupby("year")["ratio"].agg(["mean", "std", "count"])
    print(by_year.round(4))

    print("\nDay-of-week ratio (2020-2022):")
    recent = sales[sales["Date"].dt.year >= 2020].copy()
    recent["dow"] = recent["Date"].dt.dayofweek
    print(recent.groupby("dow")["ratio"].agg(["mean", "std"]).round(4))

    print("\nMonth ratio (2020-2022):")
    recent["month"] = recent["Date"].dt.month
    print(recent.groupby("month")["ratio"].agg(["mean", "std"]).round(4))

    # === Simulate on 2022: predict COGS from Revenue using two strategies ===
    train = sales[sales["Date"] < pd.Timestamp("2022-01-01")].copy()
    val = sales[(sales["Date"] >= pd.Timestamp("2022-01-01")) & (sales["Date"] <= pd.Timestamp("2022-12-31"))].copy()

    # Strategy A: single grand-mean ratio from train
    r_grand = train["ratio"].mean()
    pred_a = val["Revenue"].values * r_grand

    # Strategy B: last-3-year mean
    r_last3 = train[train["Date"] >= pd.Timestamp("2019-01-01")]["ratio"].mean()
    pred_b = val["Revenue"].values * r_last3

    # Strategy C: (dow × month) profile from last-3-year train
    last3 = train[train["Date"] >= pd.Timestamp("2019-01-01")].copy()
    last3["dow"] = last3["Date"].dt.dayofweek
    last3["month"] = last3["Date"].dt.month
    profile = last3.groupby(["dow", "month"])["ratio"].mean()
    val["dow"] = val["Date"].dt.dayofweek
    val["month"] = val["Date"].dt.month
    ratio_c = val.apply(lambda r: profile.get((r["dow"], r["month"]), r_last3), axis=1).values
    pred_c = val["Revenue"].values * ratio_c

    print("\nSimulated 2022 COGS = Revenue_actual * ratio_strategy (uses true Revenue):")
    print(f"  A grand mean ({r_grand:.4f}):           MAE = {_mae(val['COGS'].values, pred_a):>12,.0f}")
    print(f"  B last-3-year mean ({r_last3:.4f}):     MAE = {_mae(val['COGS'].values, pred_b):>12,.0f}")
    print(f"  C dow x month profile:                  MAE = {_mae(val['COGS'].values, pred_c):>12,.0f}")

    # Baseline: trivial identity check — what MAE if we just replicated last-year COGS?
    train_2021 = sales[(sales["Date"] >= pd.Timestamp("2021-01-01")) & (sales["Date"] <= pd.Timestamp("2021-12-31"))].copy()
    train_2021["dayofyear"] = train_2021["Date"].dt.dayofyear
    val["dayofyear"] = val["Date"].dt.dayofyear
    cogs_2021_profile = train_2021.set_index("dayofyear")["COGS"]
    pred_d = val["dayofyear"].map(cogs_2021_profile).fillna(cogs_2021_profile.mean()).values
    print(f"  D naive 2021 COGS replicated:           MAE = {_mae(val['COGS'].values, pred_d):>12,.0f}")


if __name__ == "__main__":
    main()
