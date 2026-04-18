"""Compute CV MAE of the ensemble itself (not just individual models).

This tells us whether mixing in detrended_lgbm + prophet actually helps vs.
using the baseline alone. We reuse the same inverse-MAE weights the ensemble
script derives.
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
from src.evaluation import mae as _mae
from src.ensemble import inverse_mae_weights

from src.models.baseline import fit_and_predict as baseline_predict
from src.models.detrended_lgbm import DetrendedLgbm
from src.models.prophet_model import ProphetForecaster


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

    folds = yearly_walkforward(val_years=(2020, 2021, 2022))

    results = {
        "Revenue": {"baseline": [], "detrended_lgbm": [], "prophet": [], "ensemble": []},
        "COGS": {"baseline": [], "detrended_lgbm": [], "prophet": [], "ensemble": []},
    }

    for fold in folds:
        print(f"\n--- Fold val={fold.val_start.year} ---")
        for col in ("Revenue", "COGS"):
            frame_col = frame_rev if col == "Revenue" else frame_cogs
            train_sales = sales[sales["Date"] <= fold.train_end]
            val_sales = sales[fold.val_mask(sales["Date"])]
            train_frame = frame_col[(frame_col["Date"] <= fold.train_end) & frame_col[col].notna()].copy()
            val_frame = frame_col[fold.val_mask(frame_col["Date"])].copy()

            y_val = val_sales[col].values

            # baseline
            pred_base = baseline_predict(train_sales, val_sales["Date"], col).values

            # detrended lgbm
            dl = DetrendedLgbm(target_col=col)
            dl.fit(train_frame, val_frame=val_frame)
            pred_dl = dl.predict(val_frame)

            # prophet
            pm = ProphetForecaster(target_col=col, anomaly_md=anomaly_md)
            pm.fit(train_sales)
            pred_pm = pm.predict(val_sales["Date"]).values

            mae_base = _mae(y_val, pred_base)
            mae_dl = _mae(y_val, pred_dl)
            mae_pm = _mae(y_val, pred_pm)

            weights = inverse_mae_weights({
                "baseline": mae_base,
                "detrended_lgbm": mae_dl,
                "prophet": mae_pm,
            })
            pred_ens = (
                weights["baseline"] * pred_base
                + weights["detrended_lgbm"] * pred_dl
                + weights["prophet"] * pred_pm
            )
            mae_ens = _mae(y_val, pred_ens)

            results[col]["baseline"].append(mae_base)
            results[col]["detrended_lgbm"].append(mae_dl)
            results[col]["prophet"].append(mae_pm)
            results[col]["ensemble"].append(mae_ens)

            print(
                f"  {col}: base={mae_base:,.0f}  dl={mae_dl:,.0f}  "
                f"prophet={mae_pm:,.0f}  ENS={mae_ens:,.0f}"
            )

    print("\n=== Mean MAE across folds ===")
    for col, scores in results.items():
        print(f"{col}:")
        for name, vals in scores.items():
            print(f"  {name:18s} {np.mean(vals):>12,.0f}")


if __name__ == "__main__":
    main()
