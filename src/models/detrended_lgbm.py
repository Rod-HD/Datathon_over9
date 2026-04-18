"""Detrended LightGBM: model the seasonal residual, add trend back.

Raw LightGBM struggles to extrapolate the multi-year revenue growth trend. Here
we divide out an exponential YoY trend learned from train, let LightGBM predict
the seasonal ratio (mean ~1.0), then scale back.
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
    "num_leaves": 31,
    "min_data_in_leaf": 30,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "lambda_l2": 0.5,
    "verbose": -1,
    "seed": 42,
}


@dataclass
class DetrendedLgbm:
    target_col: str
    params: dict = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    num_boost_round: int = 3000
    early_stopping_rounds: int = 150
    growth_mode: str = "geo_full"  # geo_full, flat, geo_last3, linear_last3
    growth_: float = 1.0
    base_year_: int = 2022
    base_daily_: float = 0.0
    feature_cols: list[str] = field(default_factory=list)
    model_: lgb.Booster | None = None

    # columns that leak the linear trend — drop them so LGBM learns seasonality only
    _drop_features = {"Date", "year", "days_since_2012"}

    def _fit_growth(self, sales: pd.DataFrame) -> None:
        df = sales.copy()
        df["year"] = df["Date"].dt.year
        counts = df.groupby("year")["Date"].nunique()
        totals = df.groupby("year")[self.target_col].sum()
        full_year_mask = counts >= 360
        annual = totals.loc[full_year_mask[full_year_mask].index]
        last_year = int(annual.index.max())
        self.base_year_ = last_year
        self.base_daily_ = float(annual.loc[last_year] / 365.0)

        full = annual.loc[2013:last_year] if 2013 in annual.index else annual
        yoy = full.pct_change().dropna()

        if self.growth_mode == "flat" or len(yoy) == 0:
            self.growth_ = 1.0
        elif self.growth_mode == "geo_full":
            self.growth_ = float((1 + yoy).prod() ** (1 / len(yoy)))
        elif self.growth_mode == "geo_last3":
            tail = yoy.tail(3)
            self.growth_ = float((1 + tail).prod() ** (1 / len(tail)))
        elif self.growth_mode == "linear_last3":
            tail = full.tail(3)
            import numpy as _np
            slope = _np.polyfit(tail.index.values.astype(float), _np.log(tail.values.astype(float)), 1)[0]
            self.growth_ = float(_np.exp(slope))
        else:
            raise ValueError(self.growth_mode)

    def _trend(self, dates: pd.Series) -> np.ndarray:
        years_ahead = pd.to_datetime(dates).dt.year.values - self.base_year_
        return self.base_daily_ * (self.growth_ ** years_ahead)

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        drop = [c for c in self._drop_features | {self.target_col} if c in frame.columns]
        return frame.drop(columns=drop)

    def fit(
        self,
        train_frame: pd.DataFrame,
        val_frame: pd.DataFrame | None = None,
        trend_sales: pd.DataFrame | None = None,
    ) -> None:
        """trend_sales: optional separate dataframe (Date, target_col) used only
        to estimate growth/base. Useful when train_frame excludes the last
        year for early stopping — pass full sales here to keep base_year right.
        """
        sales_view = (trend_sales if trend_sales is not None else train_frame)[
            ["Date", self.target_col]
        ].dropna()
        self._fit_growth(sales_view)

        ratio_train = train_frame[self.target_col].values / self._trend(train_frame["Date"])
        X_train = self._prepare(train_frame)
        self.feature_cols = list(X_train.columns)

        dtrain = lgb.Dataset(X_train, ratio_train)
        valid_sets, valid_names = [dtrain], ["train"]
        callbacks = [lgb.log_evaluation(period=0)]

        if val_frame is not None:
            ratio_val = val_frame[self.target_col].values / self._trend(val_frame["Date"])
            X_val = self._prepare(val_frame)[self.feature_cols]
            dval = lgb.Dataset(X_val, ratio_val, reference=dtrain)
            valid_sets.append(dval)
            valid_names.append("val")
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))

        self.model_ = lgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        assert self.model_ is not None
        X = self._prepare(frame)[self.feature_cols]
        ratio = self.model_.predict(X, num_iteration=self.model_.best_iteration)
        return np.clip(ratio * self._trend(frame["Date"]), 0, None)

    def feature_importance(self, importance_type: str = "gain") -> pd.Series:
        assert self.model_ is not None
        imp = self.model_.feature_importance(importance_type=importance_type)
        return pd.Series(imp, index=self.feature_cols).sort_values(ascending=False)
