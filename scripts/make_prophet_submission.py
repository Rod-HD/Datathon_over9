"""Train Prophet on train period, predict test, export submission."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import logging
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

import pandas as pd

from src.data_loader import load_sales, load_sample_submission
from src.features import discover_anomaly_dates
from src.models.prophet_model import ProphetForecaster
from src.evaluation import score_all
from src.cv import yearly_walkforward


def _cv_for(col: str, sales: pd.DataFrame, anomaly_md: pd.DataFrame) -> None:
    print(f"\n--- Prophet CV for {col} ---")
    for fold in yearly_walkforward(val_years=(2020, 2021, 2022)):
        train = sales[sales["Date"] <= fold.train_end]
        val = sales[fold.val_mask(sales["Date"])]

        model = ProphetForecaster(target_col=col, anomaly_md=anomaly_md)
        model.fit(train)
        pred = model.predict(val["Date"])
        score_all(val[col].values, pred.values, label=f"  {col} {fold.val_start.year}")


def main() -> None:
    sales = load_sales()
    sub = load_sample_submission()

    anomaly_md = discover_anomaly_dates(sales, "Revenue", z_threshold=2.0)
    print(f"Anomaly dates kept: {len(anomaly_md)}")

    _cv_for("Revenue", sales, anomaly_md)
    _cv_for("COGS", sales, anomaly_md)

    print("\n=== Refitting on full train (2012-2022) ===")
    rev_model = ProphetForecaster(target_col="Revenue", anomaly_md=anomaly_md)
    rev_model.fit(sales)
    rev_pred = rev_model.predict(sub["Date"])

    cogs_model = ProphetForecaster(target_col="COGS", anomaly_md=anomaly_md)
    cogs_model.fit(sales)
    cogs_pred = cogs_model.predict(sub["Date"])

    out = pd.DataFrame({
        "Date": sub["Date"].dt.strftime("%Y-%m-%d"),
        "Revenue": rev_pred.round(2).values,
        "COGS": cogs_pred.round(2).values,
    })

    out_path = ROOT / "submissions" / "submission_v2_prophet.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved {len(out)} rows to {out_path}")
    print(out.head())


if __name__ == "__main__":
    main()
