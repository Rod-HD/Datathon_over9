"""Build ensemble submission: CV-weighted average of baseline + Prophet + detrended LGBM.

Workflow:
1. Run each base model under the yearly walk-forward CV, record MAE per target.
2. Derive weights = (1/MAE) / sum.
3. Fit each on full train, predict test, combine.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import logging
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

import numpy as np
import pandas as pd

from src.data_loader import load_sales, load_sample_submission, load_promotions
from src.features import build_feature_frame, discover_anomaly_dates
from src.cv import yearly_walkforward
from src.evaluation import mae as _mae, score_all
from src.ensemble import inverse_mae_weights, weighted_average

from src.models.baseline import fit_and_predict as baseline_predict
from src.models.detrended_lgbm import DetrendedLgbm
from src.models.prophet_model import ProphetForecaster


def _baseline_cv_mae(sales: pd.DataFrame, col: str) -> float:
    errs = []
    for fold in yearly_walkforward(val_years=(2020, 2021, 2022)):
        train = sales[sales["Date"] <= fold.train_end]
        val = sales[fold.val_mask(sales["Date"])]
        pred = baseline_predict(train, val["Date"], col)
        errs.append(_mae(val[col].values, pred.values))
    return float(np.mean(errs))


def _prophet_cv_mae(sales: pd.DataFrame, col: str, anomaly_md: pd.DataFrame) -> float:
    errs = []
    for fold in yearly_walkforward(val_years=(2020, 2021, 2022)):
        train = sales[sales["Date"] <= fold.train_end]
        val = sales[fold.val_mask(sales["Date"])]
        m = ProphetForecaster(target_col=col, anomaly_md=anomaly_md)
        m.fit(train)
        pred = m.predict(val["Date"])
        errs.append(_mae(val[col].values, pred.values))
    return float(np.mean(errs))


def _detrended_cv_mae(frame: pd.DataFrame, col: str) -> float:
    errs = []
    for fold in yearly_walkforward(val_years=(2020, 2021, 2022)):
        train = frame[(frame["Date"] <= fold.train_end) & frame[col].notna()].copy()
        val = frame[fold.val_mask(frame["Date"])].copy()
        m = DetrendedLgbm(target_col=col)
        m.fit(train, val_frame=val)
        pred = m.predict(val)
        errs.append(_mae(val[col].values, pred))
    return float(np.mean(errs))


def main() -> None:
    sales = load_sales()
    sub = load_sample_submission()
    promos = load_promotions()

    anomaly_md = discover_anomaly_dates(sales, "Revenue", z_threshold=2.0)
    all_dates = pd.concat([sales["Date"], sub["Date"]]).reset_index(drop=True)
    frame = build_feature_frame(all_dates, sales=sales, promotions=promos, anomaly_md=anomaly_md)

    def _attach(col: str) -> pd.DataFrame:
        lookup = sales.set_index("Date")[col]
        out = frame.copy()
        out[col] = pd.to_datetime(out["Date"]).map(lookup)
        return out

    frame_rev = _attach("Revenue")
    frame_cogs = _attach("COGS")
    test_mask = pd.to_datetime(frame["Date"]) >= pd.Timestamp("2023-01-01")
    test_frame = frame[test_mask].copy()

    # ---------- CV MAE per model per target ----------
    print("=== Collecting CV MAEs ===")
    cv = {"Revenue": {}, "COGS": {}}
    for col in ("Revenue", "COGS"):
        cv[col]["baseline"] = _baseline_cv_mae(sales, col)
        cv[col]["detrended_lgbm"] = _detrended_cv_mae(
            frame_rev if col == "Revenue" else frame_cogs, col
        )
        cv[col]["prophet"] = _prophet_cv_mae(sales, col, anomaly_md)
        print(f"{col}: {cv[col]}")

    weights = {col: inverse_mae_weights(cv[col]) for col in cv}
    print("\nEnsemble weights:")
    for col, w in weights.items():
        print(f"  {col}: {w}")

    # ---------- Fit each on full train, predict test ----------
    print("\n=== Final fits ===")
    preds = {"Revenue": {}, "COGS": {}}
    for col in ("Revenue", "COGS"):
        preds[col]["baseline"] = pd.Series(
            baseline_predict(sales, test_frame["Date"], col).values,
            index=test_frame.index,
        )

        train_col = (frame_rev if col == "Revenue" else frame_cogs)
        train_col = train_col[train_col[col].notna()].copy()
        h_mask = train_col["Date"] >= pd.Timestamp("2022-01-01")
        dl = DetrendedLgbm(target_col=col)
        dl.fit(train_col[~h_mask], val_frame=train_col[h_mask])
        preds[col]["detrended_lgbm"] = pd.Series(dl.predict(test_frame), index=test_frame.index)

        pm = ProphetForecaster(target_col=col, anomaly_md=anomaly_md)
        pm.fit(sales)
        preds[col]["prophet"] = pd.Series(pm.predict(test_frame["Date"]).values, index=test_frame.index)

    ensembled = {
        col: weighted_average(preds[col], weights[col]).round(2) for col in preds
    }

    # ---------- Assemble submission ----------
    out = pd.DataFrame({
        "Date": pd.to_datetime(test_frame["Date"]).dt.strftime("%Y-%m-%d").values,
        "Revenue": ensembled["Revenue"].values,
        "COGS": ensembled["COGS"].values,
    })
    order = sub["Date"].dt.strftime("%Y-%m-%d").tolist()
    out = out.set_index("Date").loc[order].reset_index()

    out_path = ROOT / "submissions" / "submission_v4_ensemble.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved {len(out)} rows to {out_path}")
    print(out.head())
    print(out.tail())

    # Sanity: compare to baseline submission
    print("\nAverage diff vs. baseline:")
    baseline_sub = pd.read_csv(ROOT / "submissions" / "submission_v0_baseline.csv")
    diff = out["Revenue"].values - baseline_sub["Revenue"].values
    print(f"  Revenue mean-abs-diff: {np.mean(np.abs(diff)):,.0f}")


if __name__ == "__main__":
    main()
