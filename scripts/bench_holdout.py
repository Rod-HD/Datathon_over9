"""Benchmark all models / submissions on the primary 18-month holdout fold.

This fold (2021-07-01 .. 2022-12-31) mimics the real test horizon length and
dodges the 2020 COVID shock that contaminated the previous yearly CV. It is
the new ground-truth for deciding which variants to keep.

Two modes:
  1. Train-and-score: fit each live model on train <= 2021-06-30, score on
     the holdout window.
  2. Submission score (if a submissions/*.csv contains 2022 dates, skip — our
     Kaggle submissions only cover 2023+, so this mode is not applicable).

Appends a row to reports/bench.csv.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import logging
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

import numpy as np
import pandas as pd

from src.data_loader import load_sales, load_promotions, load_sample_submission
from src.features import build_feature_frame, discover_anomaly_dates
from src.cv import primary_holdout
from src.evaluation import mae as _mae
from src.models.baseline import fit_and_predict as baseline_predict
from src.models.detrended_lgbm import DetrendedLgbm
from src.models.prophet_model import ProphetForecaster
from src.models.lgbm import LgbmForecaster


BENCH_PATH = ROOT / "reports" / "bench.csv"
BENCH_PATH.parent.mkdir(parents=True, exist_ok=True)


def _log_row(row: dict) -> None:
    header = not BENCH_PATH.exists()
    pd.DataFrame([row]).to_csv(BENCH_PATH, mode="a", header=header, index=False)


def _attach(frame: pd.DataFrame, sales: pd.DataFrame, col: str) -> pd.DataFrame:
    lookup = sales.set_index("Date")[col]
    out = frame.copy()
    out[col] = pd.to_datetime(out["Date"]).map(lookup)
    return out


def _score(name: str, params: str, y_rev, p_rev, y_cogs, p_cogs) -> None:
    r = _mae(y_rev, p_rev)
    c = _mae(y_cogs, p_cogs)
    total = r + c
    print(f"  {name:30s} rev={r:>12,.0f}  cogs={c:>12,.0f}  total={total:>12,.0f}")
    _log_row({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": name,
        "params": params,
        "fold": "primary_holdout",
        "revenue_mae": round(r, 2),
        "cogs_mae": round(c, 2),
        "total_mae": round(total, 2),
        "lb_score": "",
        "notes": "",
    })


def main() -> None:
    fold = primary_holdout()
    print(f"Primary holdout: train <= {fold.train_end.date()}, val {fold.val_start.date()} .. {fold.val_end.date()}")

    sales = load_sales()
    promos = load_promotions()
    sub = load_sample_submission()

    train_sales = sales[sales["Date"] <= fold.train_end].copy()
    val_sales = sales[fold.val_mask(sales["Date"])].copy()

    y_rev = val_sales["Revenue"].values
    y_cogs = val_sales["COGS"].values

    # ---------- baseline (current geo_full) ----------
    print("\n[Baseline geo_full]")
    p_rev = baseline_predict(train_sales, val_sales["Date"], "Revenue").values
    p_cogs = baseline_predict(train_sales, val_sales["Date"], "COGS").values
    _score("baseline_geo_full", "growth=geo_full", y_rev, p_rev, y_cogs, p_cogs)

    # ---------- Prophet default ----------
    print("\n[Prophet default]")
    anomaly_md = discover_anomaly_dates(train_sales, "Revenue", z_threshold=2.0)
    m_rev = ProphetForecaster(target_col="Revenue", anomaly_md=anomaly_md)
    m_rev.fit(train_sales)
    p_rev_ph = m_rev.predict(val_sales["Date"]).values
    m_cogs = ProphetForecaster(target_col="COGS", anomaly_md=anomaly_md)
    m_cogs.fit(train_sales)
    p_cogs_ph = m_cogs.predict(val_sales["Date"]).values
    _score("prophet_default", "log1p+mult+cps=0.05", y_rev, p_rev_ph, y_cogs, p_cogs_ph)

    # ---------- Detrended LGBM + LGBM plain need feature frame ----------
    print("\n[Detrended LightGBM current]")
    all_dates = pd.concat([train_sales["Date"], val_sales["Date"]]).reset_index(drop=True)
    frame = build_feature_frame(all_dates, sales=train_sales, promotions=promos, anomaly_md=anomaly_md)
    frame_rev = _attach(frame, sales, "Revenue")
    frame_cogs = _attach(frame, sales, "COGS")

    train_mask = pd.to_datetime(frame["Date"]) <= fold.train_end
    val_mask = fold.val_mask(pd.to_datetime(frame["Date"]))

    def _dl(col: str, f: pd.DataFrame):
        tr = f[train_mask & f[col].notna()].copy()
        vl = f[val_mask].copy()
        h_mask = tr["Date"] >= pd.Timestamp("2020-01-01")
        m = DetrendedLgbm(target_col=col)
        m.fit(tr[~h_mask], val_frame=tr[h_mask])
        return m.predict(vl)

    p_rev_dl = _dl("Revenue", frame_rev)
    p_cogs_dl = _dl("COGS", frame_cogs)
    _score("detrended_lgbm_current", "growth=geo_full,drop_year", y_rev, p_rev_dl, y_cogs, p_cogs_dl)

    print("\n[LightGBM plain current]")
    def _lgbm(col: str, f: pd.DataFrame):
        tr = f[train_mask & f[col].notna()].copy()
        vl = f[val_mask].copy()
        h_mask = tr["Date"] >= pd.Timestamp("2020-01-01")
        m = LgbmForecaster(target_col=col)
        m.fit(tr[~h_mask], val_frame=tr[h_mask])
        return m.predict(vl)

    p_rev_lg = _lgbm("Revenue", frame_rev)
    p_cogs_lg = _lgbm("COGS", frame_cogs)
    _score("lgbm_plain_current", "log1p+year+days_since", y_rev, p_rev_lg, y_cogs, p_cogs_lg)

    # ---------- Ensemble current weights ----------
    print("\n[Ensemble inverse-MAE CV current]")
    # Use the v4 weights we recorded: baseline 0.40, detrended 0.31, prophet 0.29
    w = {"baseline": 0.4047, "detrended": 0.3083, "prophet": 0.2870}
    p_rev_ens = w["baseline"] * p_rev + w["detrended"] * p_rev_dl + w["prophet"] * p_rev_ph
    p_cogs_ens = w["baseline"] * p_cogs + w["detrended"] * p_cogs_dl + w["prophet"] * p_cogs_ph
    _score("ensemble_current", "w=0.40/0.31/0.29", y_rev, p_rev_ens, y_cogs, p_cogs_ens)

    print(f"\nResults appended to {BENCH_PATH}")


if __name__ == "__main__":
    main()
