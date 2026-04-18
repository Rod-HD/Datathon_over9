"""LightGBM regressor for daily Revenue / COGS.

Uses the feature matrix from ``src.features.build_feature_frame`` with log1p
target transform to stabilise variance.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import lightgbm as lgb


DEFAULT_PARAMS: dict = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "lambda_l2": 0.1,
    "verbose": -1,
    "seed": 42,
}

# Columns that leak the linear trend — tree models can't extrapolate them.
_TREND_LEAK_COLS = {"year", "days_since_2012"}


@dataclass
class LgbmForecaster:
    target_col: str
    params: dict = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    num_boost_round: int = 4000
    early_stopping_rounds: int = 200
    residual_col: str | None = None
    drop_trend_leak: bool = False
    log_transform: bool = True
    feature_cols: list[str] = field(default_factory=list)
    model_: lgb.Booster | None = None

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        drop = {"Date", self.target_col}
        if self.residual_col:
            drop.add(self.residual_col)
        if self.drop_trend_leak:
            drop |= _TREND_LEAK_COLS
        return frame.drop(columns=[c for c in drop if c in frame.columns])

    def _target(self, frame: pd.DataFrame) -> np.ndarray:
        if self.residual_col is not None:
            # residual target = actual - base_pred; no log transform on residuals
            return (frame[self.target_col].astype(float) - frame[self.residual_col].astype(float)).values
        y = frame[self.target_col].astype(float)
        return np.log1p(y).values if self.log_transform else y.values

    def _invert(self, raw: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
        if self.residual_col is not None:
            return frame[self.residual_col].astype(float).values + raw
        return np.expm1(raw) if self.log_transform else raw

    def fit(self, train_frame: pd.DataFrame, val_frame: pd.DataFrame | None = None) -> None:
        X_train = self._prepare(train_frame)
        y_train = self._target(train_frame)
        self.feature_cols = list(X_train.columns)

        train_set = lgb.Dataset(X_train, y_train)
        valid_sets = [train_set]
        valid_names = ["train"]
        callbacks = [lgb.log_evaluation(period=0)]

        if val_frame is not None:
            X_val = self._prepare(val_frame)[self.feature_cols]
            y_val = self._target(val_frame)
            valid_set = lgb.Dataset(X_val, y_val, reference=train_set)
            valid_sets.append(valid_set)
            valid_names.append("val")
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))

        self.model_ = lgb.train(
            self.params,
            train_set,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        assert self.model_ is not None
        X = self._prepare(frame)[self.feature_cols]
        raw = self.model_.predict(X, num_iteration=self.model_.best_iteration)
        return np.clip(self._invert(raw, frame), 0, None)

    def feature_importance(self, importance_type: str = "gain") -> pd.Series:
        assert self.model_ is not None
        imp = self.model_.feature_importance(importance_type=importance_type)
        return pd.Series(imp, index=self.feature_cols).sort_values(ascending=False)
