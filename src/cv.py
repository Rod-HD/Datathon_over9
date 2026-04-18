"""Time-series cross-validation — walk-forward expanding window.

Training data spans 2012-07-04 -> 2022-12-31.
We validate on the last N full years so every fold mimics the real forecast gap.
"""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class Fold:
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp

    def train_mask(self, dates: pd.Series) -> pd.Series:
        return dates <= self.train_end

    def val_mask(self, dates: pd.Series) -> pd.Series:
        return (dates >= self.val_start) & (dates <= self.val_end)


def yearly_walkforward(val_years=(2020, 2021, 2022)) -> list[Fold]:
    """Three folds, each validates on one full calendar year."""
    folds = []
    for year in val_years:
        folds.append(
            Fold(
                train_end=pd.Timestamp(f"{year - 1}-12-31"),
                val_start=pd.Timestamp(f"{year}-01-01"),
                val_end=pd.Timestamp(f"{year}-12-31"),
            )
        )
    return folds


def holdout_last_18_months() -> Fold:
    """Single fold mimicking the exact forecast horizon (Jul-2021 -> Dec-2022)."""
    return Fold(
        train_end=pd.Timestamp("2021-06-30"),
        val_start=pd.Timestamp("2021-07-01"),
        val_end=pd.Timestamp("2022-12-31"),
    )


def primary_holdout() -> Fold:
    """Primary validation fold used after LB feedback showed 2020-2022 CV
    was contaminated by COVID breaks. Mimics the real 18-month test horizon."""
    return holdout_last_18_months()
