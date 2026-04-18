"""v10 — v8 Prophet tuned predictions + soft smoothing (alpha=0.4, window=7).

Ablation showed -1.5% holdout geo improvement with near-zero LB risk (smoothing
is intrinsic, doesn't pull toward another model). Pull_to_baseline variants
gave bigger gains but risk lowering LB since prior LB showed Prophet > baseline.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.postprocess import soft_smooth


def main() -> None:
    src = pd.read_csv(ROOT / "submissions" / "submission_v8_prophet_tuned.csv")
    rev = soft_smooth(src["Revenue"].to_numpy(), alpha=0.4, window=7)
    cogs = soft_smooth(src["COGS"].to_numpy(), alpha=0.4, window=7)
    out = pd.DataFrame({
        "Date": src["Date"].values,
        "Revenue": np.round(rev, 2),
        "COGS": np.round(cogs, 2),
    })
    path = ROOT / "submissions" / "submission_v10_prophet_tuned_smoothed.csv"
    out.to_csv(path, index=False)
    print(f"Saved {len(out)} rows to {path}")
    print(out.head())
    print("\nDiff vs v8 (mean abs):")
    print(f"  Revenue {np.mean(np.abs(rev - src['Revenue'].values)):,.0f}")
    print(f"  COGS    {np.mean(np.abs(cogs - src['COGS'].values)):,.0f}")


if __name__ == "__main__":
    main()
