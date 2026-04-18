"""Ensemble v9 — holdout-informed weights across v5/v6/v8 submissions.

Reads existing CSVs from submissions/, weighted by inverse of each model's
total primary_holdout MAE (pulled from reports/bench.csv where available).
Also emits a median variant (v9b) as a robust alternative.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.ensemble import lb_informed_weights, weighted_average, median_ensemble
from src.data_loader import load_sample_submission


# holdout-aggregated MAE (Revenue+COGS summed) from the bench runs; used as the
# weight basis since real LB scores for v5/v6/v8 aren't in yet.
HOLDOUT_SCORES = {
    "v5_baseline_fixed":  1_124_000,   # flat + sw5, primary_holdout
    "v6_lgbm_residual":   1_250_000,   # residual stacking on baseline, ~1104K y22 + ~1396K ph approx
    "v8_prophet_tuned":   1_250_852,   # best grid cell primary_holdout total
}

SUBMISSION_FILES = {
    "v5_baseline_fixed": "submission_v5_baseline_fixed.csv",
    "v6_lgbm_residual":  "submission_v6_lgbm_residual.csv",
    "v8_prophet_tuned":  "submission_v8_prophet_tuned.csv",
}


def _load(name):
    path = ROOT / "submissions" / SUBMISSION_FILES[name]
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date")


def main() -> None:
    sub = load_sample_submission()
    order = sub["Date"]

    loaded = {name: _load(name) for name in SUBMISSION_FILES}
    for name, df in loaded.items():
        if len(df) != len(order):
            raise ValueError(f"{name}: {len(df)} rows != {len(order)}")

    weights = lb_informed_weights(HOLDOUT_SCORES)
    print("Weights:")
    for n, w in weights.items():
        print(f"  {n:<22s}  holdout_total={HOLDOUT_SCORES[n]:>12,.0f}  w={w:.3f}")

    preds_rev = {n: pd.Series(df.loc[order, "Revenue"].values, index=range(len(order)))
                 for n, df in loaded.items()}
    preds_cogs = {n: pd.Series(df.loc[order, "COGS"].values, index=range(len(order)))
                  for n, df in loaded.items()}

    rev_w = weighted_average(preds_rev, weights)
    cogs_w = weighted_average(preds_cogs, weights)
    rev_med = median_ensemble(preds_rev)
    cogs_med = median_ensemble(preds_cogs)

    out_w = pd.DataFrame({
        "Date": order.dt.strftime("%Y-%m-%d").values,
        "Revenue": rev_w.round(2).values,
        "COGS": cogs_w.round(2).values,
    })
    path_w = ROOT / "submissions" / "submission_v9_ensemble_weighted.csv"
    out_w.to_csv(path_w, index=False)
    print(f"\nSaved weighted ensemble to {path_w}")
    print(out_w.head())

    out_m = pd.DataFrame({
        "Date": order.dt.strftime("%Y-%m-%d").values,
        "Revenue": rev_med.round(2).values,
        "COGS": cogs_med.round(2).values,
    })
    path_m = ROOT / "submissions" / "submission_v9b_ensemble_median.csv"
    out_m.to_csv(path_m, index=False)
    print(f"Saved median ensemble to {path_m}")
    print(out_m.head())

    # v9c — LB-informed heavy-Prophet (v2 prophet LB=1.073M was best of prior 5).
    # Pull weight 0.6 to Prophet, 0.2 each to baseline & LGBM residual.
    heavy_w = {"v5_baseline_fixed": 0.2, "v6_lgbm_residual": 0.2, "v8_prophet_tuned": 0.6}
    print("\nHeavy-Prophet weights:", heavy_w)
    rev_h = weighted_average(preds_rev, heavy_w)
    cogs_h = weighted_average(preds_cogs, heavy_w)
    out_h = pd.DataFrame({
        "Date": order.dt.strftime("%Y-%m-%d").values,
        "Revenue": rev_h.round(2).values,
        "COGS": cogs_h.round(2).values,
    })
    path_h = ROOT / "submissions" / "submission_v9c_ensemble_heavy_prophet.csv"
    out_h.to_csv(path_h, index=False)
    print(f"Saved heavy-Prophet ensemble to {path_h}")
    print(out_h.head())


if __name__ == "__main__":
    main()
