"""Post-processing ablation on primary_holdout + yearly_2022.

For the tuned Prophet + baseline-flat combo, test each knob:
  - clip non-negative (always on; sanity)
  - soft smoothing alpha in {0.0, 0.2, 0.4}, window 7
  - pull to baseline (threshold 0.5, weight 0.5)
  - cap cogs / revenue at 1.05

Report delta MAE (Revenue + COGS) vs. raw tuned Prophet. Apply any step that
lowers geo-mean of the two folds by >= 0.3%.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import logging
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

import numpy as np
import pandas as pd

from src.data_loader import load_sales, load_promotions
from src.features import build_feature_frame, discover_anomaly_dates
from src.cv import primary_holdout, yearly_walkforward
from src.evaluation import mae as _mae
from src.models.prophet_model import ProphetForecaster
from src.models.baseline import fit_and_predict as baseline_predict
from src.postprocess import soft_smooth, pull_to_baseline, cap_cogs_by_revenue


PROPHET_CFG = {
    "changepoint_prior_scale": 0.1,
    "seasonality_prior_scale": 10.0,
    "yearly_seasonality": 20,
    "changepoint_range": 0.85,
}
REGRESSORS = ["promo_active_count", "is_anomaly_day"]


def _prep(sales, promos, fold):
    train_sales = sales[sales["Date"] <= fold.train_end].copy()
    val_sales = sales[fold.val_mask(sales["Date"])].copy()
    anomaly_md = discover_anomaly_dates(train_sales, "Revenue", z_threshold=2.0)
    all_dates = pd.concat([train_sales["Date"], val_sales["Date"]]).reset_index(drop=True)
    frame = build_feature_frame(all_dates, sales=train_sales, promotions=promos, anomaly_md=anomaly_md)
    rf = frame[["Date"] + [r for r in REGRESSORS if r in frame.columns]]
    return train_sales, val_sales, anomaly_md, rf


def _run_models(train_sales, val_sales, anomaly_md, rf):
    preds = {}
    for col in ("Revenue", "COGS"):
        m = ProphetForecaster(
            target_col=col,
            anomaly_md=anomaly_md,
            extra_regressors=list(REGRESSORS),
            **PROPHET_CFG,
        )
        m.fit(train_sales[["Date", col]], regressor_frame=rf)
        preds[f"{col}_prophet"] = m.predict(val_sales["Date"]).values
        preds[f"{col}_baseline"] = baseline_predict(
            train_sales, val_sales["Date"], col,
            growth_mode="flat", season_window=5,
        ).values
    return preds


def _score(y_rev, y_cogs, rev, cogs):
    return _mae(y_rev, rev) + _mae(y_cogs, cogs)


def main() -> None:
    sales = load_sales()
    promos = load_promotions()

    folds = [
        ("primary_holdout", primary_holdout()),
        ("yearly_2022", yearly_walkforward(val_years=(2022,))[0]),
    ]
    data = {}
    for name, fold in folds:
        print(f"Prepping {name}...")
        tr, vl, am, rf = _prep(sales, promos, fold)
        preds = _run_models(tr, vl, am, rf)
        data[name] = dict(
            y_rev=vl["Revenue"].values,
            y_cogs=vl["COGS"].values,
            preds=preds,
        )

    def eval_combo(label, get_rev, get_cogs):
        totals = {}
        for name, d in data.items():
            rev = get_rev(d)
            cogs = get_cogs(d)
            totals[name] = _score(d["y_rev"], d["y_cogs"], rev, cogs)
        geo = float(np.sqrt(totals["primary_holdout"] * totals["yearly_2022"]))
        print(f"  {label:<50s} ph={totals['primary_holdout']:>12,.0f}  y22={totals['yearly_2022']:>12,.0f}  geo={geo:>12,.0f}")
        return geo

    # identity baseline — raw tuned prophet
    base_geo = eval_combo("RAW prophet",
                          lambda d: d["preds"]["Revenue_prophet"],
                          lambda d: d["preds"]["COGS_prophet"])

    # clip non-neg (prophet already clips; sanity only)
    eval_combo("clip_nonneg",
               lambda d: np.clip(d["preds"]["Revenue_prophet"], 0, None),
               lambda d: np.clip(d["preds"]["COGS_prophet"], 0, None))

    # smoothing
    for alpha in (0.2, 0.4):
        eval_combo(f"smooth alpha={alpha} w7",
                   lambda d, a=alpha: soft_smooth(d["preds"]["Revenue_prophet"], alpha=a, window=7),
                   lambda d, a=alpha: soft_smooth(d["preds"]["COGS_prophet"], alpha=a, window=7))

    # pull to baseline — grid
    for thresh in (0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
        for w in (0.3, 0.5, 0.7):
            eval_combo(f"pull thresh={thresh} w={w}",
                       lambda d, t=thresh, wv=w: pull_to_baseline(d["preds"]["Revenue_prophet"], d["preds"]["Revenue_baseline"], threshold=t, weight=wv),
                       lambda d, t=thresh, wv=w: pull_to_baseline(d["preds"]["COGS_prophet"], d["preds"]["COGS_baseline"], threshold=t, weight=wv))

    # cap cogs / revenue
    eval_combo("cap cogs <= rev * 1.05",
               lambda d: d["preds"]["Revenue_prophet"],
               lambda d: cap_cogs_by_revenue(d["preds"]["COGS_prophet"], d["preds"]["Revenue_prophet"], max_ratio=1.05))

    # combo: smooth 0.2 + cap cogs 1.05
    eval_combo("smooth a=0.2 + cap cogs 1.05",
               lambda d: soft_smooth(d["preds"]["Revenue_prophet"], alpha=0.2, window=7),
               lambda d: cap_cogs_by_revenue(
                   soft_smooth(d["preds"]["COGS_prophet"], alpha=0.2, window=7),
                   soft_smooth(d["preds"]["Revenue_prophet"], alpha=0.2, window=7),
                   max_ratio=1.05))

    print(f"\nbaseline geo (raw) = {base_geo:,.0f}")


if __name__ == "__main__":
    main()
