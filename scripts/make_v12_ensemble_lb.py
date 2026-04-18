"""v12 — LB-informed ensemble. Uses ACTUAL LB scores for v2, v5, v9.

v2 Prophet default  : 1,073,100 (best)
v5 baseline flat+sw5: 1,095,017
v9 ensemble weighted: 1,111,369
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data_loader import load_sample_submission
from src.ensemble import lb_informed_weights, weighted_average, median_ensemble


LB_SCORES = {
    "v2_prophet_default":  1_073_100,
    "v5_baseline_fixed":   1_095_017,
    "v9_ensemble_weighted": 1_111_369,
}
FILES = {
    "v2_prophet_default":   "submission_v2_prophet.csv",
    "v5_baseline_fixed":    "submission_v5_baseline_fixed.csv",
    "v9_ensemble_weighted": "submission_v9_ensemble_weighted.csv",
}


def _load(name):
    df = pd.read_csv(ROOT / "submissions" / FILES[name])
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date")


def main() -> None:
    sub = load_sample_submission()
    order = sub["Date"]

    loaded = {name: _load(name) for name in FILES}
    preds_rev = {n: pd.Series(df.loc[order, "Revenue"].values, index=range(len(order)))
                 for n, df in loaded.items()}
    preds_cogs = {n: pd.Series(df.loc[order, "COGS"].values, index=range(len(order)))
                  for n, df in loaded.items()}

    weights = lb_informed_weights(LB_SCORES)
    print("LB-informed weights:")
    for n, w in weights.items():
        print(f"  {n:<24s} lb={LB_SCORES[n]:>10,.0f}  w={w:.3f}")

    # v12a — inverse LB weighted
    rev = weighted_average(preds_rev, weights)
    cogs = weighted_average(preds_cogs, weights)
    out = pd.DataFrame({
        "Date": order.dt.strftime("%Y-%m-%d").values,
        "Revenue": rev.round(2).values,
        "COGS": cogs.round(2).values,
    })
    out.to_csv(ROOT / "submissions" / "submission_v12_ensemble_lb.csv", index=False)
    print(f"\nSaved v12 LB-weighted")
    print(out.head())

    # v12b — 50/50 of v2+v5 only (the two best singles)
    rev2 = 0.5 * preds_rev["v2_prophet_default"] + 0.5 * preds_rev["v5_baseline_fixed"]
    cogs2 = 0.5 * preds_cogs["v2_prophet_default"] + 0.5 * preds_cogs["v5_baseline_fixed"]
    out2 = pd.DataFrame({
        "Date": order.dt.strftime("%Y-%m-%d").values,
        "Revenue": rev2.round(2).values,
        "COGS": cogs2.round(2).values,
    })
    out2.to_csv(ROOT / "submissions" / "submission_v12b_prophet_baseline.csv", index=False)
    print(f"\nSaved v12b 50/50 v2+v5")
    print(out2.head())

    # v12c — 60/40 v2+v5 (skew toward better model)
    rev3 = 0.6 * preds_rev["v2_prophet_default"] + 0.4 * preds_rev["v5_baseline_fixed"]
    cogs3 = 0.6 * preds_cogs["v2_prophet_default"] + 0.4 * preds_cogs["v5_baseline_fixed"]
    out3 = pd.DataFrame({
        "Date": order.dt.strftime("%Y-%m-%d").values,
        "Revenue": rev3.round(2).values,
        "COGS": cogs3.round(2).values,
    })
    out3.to_csv(ROOT / "submissions" / "submission_v12c_prophet60_baseline40.csv", index=False)
    print(f"\nSaved v12c 60/40 v2+v5")


if __name__ == "__main__":
    main()
