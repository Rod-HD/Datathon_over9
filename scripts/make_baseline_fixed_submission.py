"""Baseline submission with the holdout-winning config (flat + season_window=5)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data_loader import load_sales, load_sample_submission
from src.models.baseline import fit_and_predict


GROWTH_MODE = "flat"
SEASON_WINDOW = 5
LOG_TRANSFORM = False


def main() -> None:
    sales = load_sales()
    sub = load_sample_submission()

    rev_pred = fit_and_predict(
        sales, sub["Date"], "Revenue",
        growth_mode=GROWTH_MODE, season_window=SEASON_WINDOW, log_transform=LOG_TRANSFORM,
    )
    cogs_pred = fit_and_predict(
        sales, sub["Date"], "COGS",
        growth_mode=GROWTH_MODE, season_window=SEASON_WINDOW, log_transform=LOG_TRANSFORM,
    )

    out = pd.DataFrame({
        "Date": sub["Date"].dt.strftime("%Y-%m-%d"),
        "Revenue": rev_pred.values,
        "COGS": cogs_pred.values,
    })

    out_path = ROOT / "submissions" / "submission_v5_baseline_fixed.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} rows to {out_path}")
    print(out.head())
    print(out.tail())


if __name__ == "__main__":
    main()
