"""Prophet tuned submission (v8). Reads best config from reports/prophet_best.txt."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import logging
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

import pandas as pd

from src.data_loader import load_sales, load_sample_submission, load_promotions
from src.features import build_feature_frame, discover_anomaly_dates
from src.models.prophet_model import ProphetForecaster


def _load_best():
    path = ROOT / "reports" / "prophet_best.txt"
    if not path.exists():
        print("[WARN] prophet_best.txt not found; using hand-picked fallback cfg")
        return {
            "cfg": {
                "changepoint_prior_scale": 0.1,
                "seasonality_prior_scale": 10.0,
                "yearly_seasonality": 20,
                "changepoint_range": 0.9,
            },
            "regressors": [],
        }
    best = ast.literal_eval(path.read_text())
    return best


def main() -> None:
    sales = load_sales()
    sub = load_sample_submission()
    promos = load_promotions()

    best = _load_best()
    cfg = best["cfg"]
    regressors = best["regressors"]
    print(f"Using cfg: {cfg}")
    print(f"Regressors: {regressors}")

    anomaly_md = discover_anomaly_dates(sales, "Revenue", z_threshold=2.0)

    all_dates = pd.concat([sales["Date"], sub["Date"]]).reset_index(drop=True)
    frame = build_feature_frame(all_dates, sales=sales, promotions=promos, anomaly_md=anomaly_md)
    rf = frame[["Date"] + [r for r in regressors if r in frame.columns]] if regressors else None

    def _fit(col):
        m = ProphetForecaster(
            target_col=col,
            anomaly_md=anomaly_md,
            changepoint_prior_scale=cfg["changepoint_prior_scale"],
            seasonality_prior_scale=cfg["seasonality_prior_scale"],
            yearly_seasonality=cfg["yearly_seasonality"],
            changepoint_range=cfg["changepoint_range"],
            extra_regressors=list(regressors),
        )
        m.fit(sales[["Date", col]], regressor_frame=rf)
        return m.predict(sub["Date"]).values

    rev_pred = _fit("Revenue")
    cogs_pred = _fit("COGS")

    out = pd.DataFrame({
        "Date": sub["Date"].dt.strftime("%Y-%m-%d"),
        "Revenue": pd.Series(rev_pred).round(2).values,
        "COGS": pd.Series(cogs_pred).round(2).values,
    })
    out_path = ROOT / "submissions" / "submission_v8_prophet_tuned.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved {len(out)} rows to {out_path}")
    print(out.head())


if __name__ == "__main__":
    main()
