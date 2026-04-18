"""Post-processing hooks — gate each with holdout ablation before enabling."""
from __future__ import annotations

import numpy as np
import pandas as pd


def clip_non_negative(pred: np.ndarray) -> np.ndarray:
    return np.clip(pred, 0.0, None)


def round_2dp(pred: np.ndarray) -> np.ndarray:
    return np.round(pred, 2)


def soft_smooth(pred: np.ndarray, alpha: float = 0.3, window: int = 7) -> np.ndarray:
    """pred_smooth = (1-alpha)*pred + alpha*rolling_mean(pred, window, center).

    Edges keep the original value where the rolling mean is NaN.
    """
    s = pd.Series(pred)
    rolled = s.rolling(window=window, center=True, min_periods=1).mean()
    smoothed = (1 - alpha) * s + alpha * rolled
    return smoothed.to_numpy()


def pull_to_baseline(pred: np.ndarray, baseline: np.ndarray,
                    threshold: float = 0.5, weight: float = 0.5) -> np.ndarray:
    """If |pred - base| / base > threshold, shrink pred toward baseline."""
    base = np.where(np.abs(baseline) < 1.0, 1.0, baseline)
    dev = np.abs(pred - baseline) / np.abs(base)
    mask = dev > threshold
    out = pred.copy().astype(float)
    out[mask] = weight * pred[mask] + (1 - weight) * baseline[mask]
    return out


def cap_cogs_by_revenue(cogs_pred: np.ndarray, rev_pred: np.ndarray,
                        max_ratio: float = 1.05) -> np.ndarray:
    cap = rev_pred * max_ratio
    return np.minimum(cogs_pred, cap)
