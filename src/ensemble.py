"""Weighted-average ensemble: weight = (1/MAE) / sum(1/MAE)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def inverse_mae_weights(mae_scores: dict[str, float]) -> dict[str, float]:
    inv = {name: 1.0 / max(mae, 1.0) for name, mae in mae_scores.items()}
    total = sum(inv.values())
    return {name: w / total for name, w in inv.items()}


def lb_informed_weights(lb_scores: dict[str, float]) -> dict[str, float]:
    """Weight by inverse of Public LB score. Lower LB = higher weight."""
    inv = {name: 1.0 / max(score, 1.0) for name, score in lb_scores.items()}
    total = sum(inv.values())
    return {name: w / total for name, w in inv.items()}


def weighted_average(
    predictions: dict[str, pd.Series], weights: dict[str, float]
) -> pd.Series:
    missing = set(predictions) ^ set(weights)
    if missing:
        raise ValueError(f"Prediction/weight key mismatch: {missing}")
    arr = np.zeros_like(next(iter(predictions.values())).values, dtype=float)
    for name, pred in predictions.items():
        arr += weights[name] * pred.values
    return pd.Series(arr, index=next(iter(predictions.values())).index)


def median_ensemble(predictions: dict[str, pd.Series]) -> pd.Series:
    """Row-wise median across predictions — robust to a single bad model."""
    mat = np.stack([p.values for p in predictions.values()], axis=0)
    med = np.median(mat, axis=0)
    return pd.Series(med, index=next(iter(predictions.values())).index)
