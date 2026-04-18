"""Generate submission_v0_baseline.csv — seasonal-naive + YoY growth.

Run:  .venv/Scripts/python.exe scripts/make_baseline_submission.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data_loader import load_sales, load_sample_submission
from src.models.baseline import fit_and_predict
from src.evaluation import score_all
from src.cv import yearly_walkforward


def main() -> None:
    sales = load_sales()
    sub = load_sample_submission()

    # CV check — predict each held-out year from the prior years only
    print("=== Walk-forward CV (baseline model) ===")
    for fold in yearly_walkforward(val_years=(2020, 2021, 2022)):
        train = sales[sales["Date"] <= fold.train_end]
        val = sales[fold.val_mask(sales["Date"])]

        pred_rev = _predict_fold(train, val["Date"], "Revenue")
        pred_cogs = _predict_fold(train, val["Date"], "COGS")

        print(f"Val year {fold.val_start.year}:")
        score_all(val["Revenue"], pred_rev, label="  Revenue")
        score_all(val["COGS"], pred_cogs, label="  COGS   ")

    # Final prediction on true test skeleton
    rev_pred = fit_and_predict(sales, sub["Date"], "Revenue")
    cogs_pred = fit_and_predict(sales, sub["Date"], "COGS")

    out = pd.DataFrame({
        "Date": sub["Date"].dt.strftime("%Y-%m-%d"),
        "Revenue": rev_pred.values,
        "COGS": cogs_pred.values,
    })

    out_path = ROOT / "submissions" / "submission_v0_baseline.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved {len(out)} rows to {out_path}")
    print(out.head())


def _predict_fold(train: pd.DataFrame, val_dates: pd.Series, col: str):
    """Baseline needs the most recent full year inside `train`. For 2020 fold,
    growth is based on 2013-2019, base = 2019 annual. Adapt on the fly."""
    # Temporarily edit the "2022" reference inside fit_and_predict by shifting
    base_year = int(train["Date"].max().year)
    df = train.copy()
    df["year"] = df["Date"].dt.year

    annual = df.groupby("year")[col].sum()
    full_years = annual.loc[2013:base_year]
    yoy = full_years.pct_change().dropna()
    growth = float((1 + yoy).prod() ** (1 / len(yoy)))

    annual_means = df.groupby("year")[col].transform("mean")
    df[f"{col}_norm"] = df[col] / annual_means
    seasonal = df.groupby([df["Date"].dt.month, df["Date"].dt.day])[f"{col}_norm"].mean()
    seasonal.index.names = ["month", "day"]
    seasonal = seasonal.reset_index()

    base = float(annual.loc[base_year] / 365.0)
    v = pd.DataFrame({"Date": pd.to_datetime(val_dates)})
    v["month"] = v["Date"].dt.month
    v["day"] = v["Date"].dt.day
    v["years_ahead"] = v["Date"].dt.year - base_year
    v = v.merge(seasonal, on=["month", "day"], how="left")
    v[f"{col}_norm"] = v[f"{col}_norm"].fillna(1.0)
    return base * (growth ** v["years_ahead"]) * v[f"{col}_norm"]


if __name__ == "__main__":
    main()
