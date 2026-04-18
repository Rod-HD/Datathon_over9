"""v11 — v8 Prophet tuned + pull_to_baseline(thresh=0.3, w=0.5) against v5.

Ablation showed -6.5% holdout geo. Risk: prior LB indicated Prophet > baseline,
so pulling toward baseline may hurt LB. Use as hedge submission.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.postprocess import pull_to_baseline, soft_smooth


def main() -> None:
    prophet = pd.read_csv(ROOT / "submissions" / "submission_v8_prophet_tuned.csv")
    baseline = pd.read_csv(ROOT / "submissions" / "submission_v5_baseline_fixed.csv")
    assert (prophet["Date"].values == baseline["Date"].values).all()

    # smooth then pull
    rev_smooth = soft_smooth(prophet["Revenue"].to_numpy(), alpha=0.2, window=7)
    cogs_smooth = soft_smooth(prophet["COGS"].to_numpy(), alpha=0.2, window=7)
    rev_pulled = pull_to_baseline(rev_smooth, baseline["Revenue"].to_numpy(),
                                  threshold=0.3, weight=0.5)
    cogs_pulled = pull_to_baseline(cogs_smooth, baseline["COGS"].to_numpy(),
                                   threshold=0.3, weight=0.5)

    out = pd.DataFrame({
        "Date": prophet["Date"].values,
        "Revenue": np.round(rev_pulled, 2),
        "COGS": np.round(cogs_pulled, 2),
    })
    path = ROOT / "submissions" / "submission_v11_prophet_pulled.csv"
    out.to_csv(path, index=False)
    print(f"Saved {len(out)} rows to {path}")
    print(out.head())
    print(f"\nDiff vs v8 prophet:")
    print(f"  Revenue mean-abs: {np.mean(np.abs(rev_pulled - prophet['Revenue'].values)):,.0f}")
    print(f"  Rows pulled (Rev): {int(np.sum(rev_pulled != rev_smooth))}/{len(out)}")
    print(f"  Rows pulled (COGS): {int(np.sum(cogs_pulled != cogs_smooth))}/{len(out)}")


if __name__ == "__main__":
    main()
