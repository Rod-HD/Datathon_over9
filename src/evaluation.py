"""Evaluation helpers — MAE, RMSE, R² per the Kaggle rubric."""
from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def score_all(y_true, y_pred, label: str = "") -> dict:
    out = {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred), "r2": r2(y_true, y_pred)}
    if label:
        print(f"[{label}] MAE={out['mae']:.2f}  RMSE={out['rmse']:.2f}  R2={out['r2']:.4f}")
    return out


def score_frame(truth: pd.DataFrame, pred: pd.DataFrame, cols=("Revenue", "COGS")) -> pd.DataFrame:
    """Score each target column separately."""
    rows = []
    for c in cols:
        rows.append({"target": c, **{k: v for k, v in score_all(truth[c], pred[c]).items()}})
    return pd.DataFrame(rows).set_index("target")
