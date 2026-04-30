"""Evaluation metrics for Revenue + COGS forecasts.

The formulas match the competition statement exactly:

MAE  = (1 / n) * sum(abs(F_i - A_i))
RMSE = sqrt((1 / n) * sum((F_i - A_i)^2))
R2   = 1 - sum((A_i - F_i)^2) / sum((A_i - mean(A))^2)

For this project, Revenue and COGS are evaluated as one combined vector:
[Revenue_1..Revenue_n, COGS_1..COGS_n].
"""
from __future__ import annotations

import numpy as np


def final_submission_values(
    revenue_pred: np.ndarray,
    cogs_pred: np.ndarray,
    max_cogs_ratio: float = 1.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact values written to a submission file.

    This applies the same constraints as the final CSV export:
    non-negative values, 2-decimal rounding, and COGS <= max ratio * Revenue.
    The COGS cap is floored to cents so a rounded COGS value cannot exceed
    the rounded Revenue cap by a cent.
    """
    revenue = np.round(np.clip(np.asarray(revenue_pred, dtype=float), 0.0, None), 2)  # Clip negative → 0, round to 2dp
    cogs = np.round(np.clip(np.asarray(cogs_pred, dtype=float), 0.0, None), 2)  # Same for COGS
    cogs_cap = np.floor((revenue * max_cogs_ratio) * 100.0) / 100.0  # COGS cap = 1.05 × Revenue, floored to cents
    cogs = np.round(np.minimum(cogs, cogs_cap), 2)  # Enforce COGS ≤ cap
    return revenue, cogs


def combine_targets(revenue: np.ndarray, cogs: np.ndarray) -> np.ndarray:
    """Combine Revenue and COGS into the 1D target vector used by metrics."""
    return np.concatenate([
        np.asarray(revenue, dtype=float),
        np.asarray(cogs, dtype=float),
    ])


def mae_rmse_r2(
    pred_revenue: np.ndarray,
    pred_cogs: np.ndarray,
    true_revenue: np.ndarray,
    true_cogs: np.ndarray,
) -> tuple[float, float, float]:
    """Compute MAE, RMSE, and R2 exactly as defined in the statement."""
    forecast = combine_targets(pred_revenue, pred_cogs)  # [Rev_1..n, COGS_1..n]
    actual = combine_targets(true_revenue, true_cogs)  # [Rev_1..n, COGS_1..n]
    errors = forecast - actual

    mae = float(np.mean(np.abs(errors)))  # Mean absolute error across all 2n values
    rmse = float(np.sqrt(np.mean(errors ** 2)))  # Root mean squared error

    ss_res = float(np.sum((actual - forecast) ** 2))  # Residual sum of squares
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))  # Total sum of squares
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0  # Coefficient of determination (R² = 1 - SS_res/SS_tot)
    return mae, rmse, float(r2)
