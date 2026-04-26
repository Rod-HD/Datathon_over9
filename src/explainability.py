"""Explainability hooks for the forecasting pipeline.

Reports drivers behind the final prediction in business language:
  - Per-model contribution (NNLS weights)
  - Day-of-Week effect (additive correction strength)
  - Year-over-year growth rate (multiplicative scale)
  - Seasonal profile shape (per-month profile)
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def report_drivers(
    out_path: Path,
    nnls_w_rev: dict,
    nnls_w_cog: dict,
    scale_rev: float,
    scale_cog: float,
    dow_strength: float,
    dow_factors_rev: np.ndarray,
    dow_factors_cog: np.ndarray,
    yoy_growth_rev: float,
    yoy_growth_cog: float,
    holdout_mae: float,
    holdout_rmse: float,
    holdout_r2: float,
) -> None:
    """Write a markdown report of the drivers learned by the model."""
    lines = []
    lines.append("# Forecasting Drivers — What the Model Learned\n")
    lines.append("## 1. Per-Model Contribution (NNLS Weights)\n")
    lines.append("Mỗi model đóng góp bao nhiêu phần trăm vào dự báo cuối cùng.\n")
    lines.append("| Model | Revenue Weight | COGS Weight | Vai trò kinh doanh |")
    lines.append("|---|---:|---:|---|")
    roles = {
        "prophet":        "Bắt trend dài hạn + seasonality năm",
        "linear_last3":   "Ngoại suy tăng trưởng tuyến tính từ 3 năm gần nhất",
        "theta":          "M3/M4 method — robust với dữ liệu nhiễu",
        "holtwinters":    "Triple Exponential Smoothing — trend + weekly seasonality",
    }
    for m in ["prophet", "linear_last3", "theta", "holtwinters"]:
        lines.append(
            f"| {m} | {nnls_w_rev.get(m, 0):.3f} | {nnls_w_cog.get(m, 0):.3f} "
            f"| {roles[m]} |"
        )

    lines.append("\n## 2. Day-of-Week Effect\n")
    lines.append(f"Strength applied: **{dow_strength:.2f}** (0=tắt, 1=full).\n")
    days = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ Nhật"]
    lines.append("| DoW | Revenue factor | COGS factor |")
    lines.append("|---|---:|---:|")
    for i, d in enumerate(days):
        lines.append(f"| {d} | {dow_factors_rev[i]:.3f} | {dow_factors_cog[i]:.3f} |")

    lines.append("\n## 3. Year-over-Year Growth\n")
    lines.append(f"Revenue scale (test vs holdout): **×{scale_rev:.3f}**")
    lines.append(f"COGS scale: **×{scale_cog:.3f}**")
    lines.append(f"\nYoY growth from training: Rev {yoy_growth_rev:.2%}, "
                 f"COGS {yoy_growth_cog:.2%}")

    lines.append("\n## 4. Validation Metrics (2022 Holdout)\n")
    lines.append(f"- **MAE**:  {holdout_mae:>14,.0f}")
    lines.append(f"- **RMSE**: {holdout_rmse:>14,.0f}")
    lines.append(f"- **R²**:   {holdout_r2:>14.4f}")

    lines.append("\n## 5. Business Interpretation\n")
    grow_w = nnls_w_rev.get("linear_last3", 0)
    if grow_w > 0.5:
        lines.append("- **Trend tăng trưởng tuyến tính dominant** "
                     f"({grow_w:.0%} weight): doanh thu tiếp tục tăng đều mỗi năm. "
                     "Mô hình tin tưởng vào việc ngoại suy xu hướng quá khứ.")
    prophet_w = nnls_w_rev.get("prophet", 0)
    if prophet_w > 0.1:
        lines.append(f"- **Prophet đóng góp {prophet_w:.0%}** cho seasonality năm và "
                     "các sự kiện đặc biệt (Tết, lễ hội).")
    theta_w = nnls_w_rev.get("theta", 0)
    if theta_w > 0.05:
        lines.append(f"- **Theta {theta_w:.0%}** giúp robustness với outlier.")

    peak_dow = np.argmax(dow_factors_rev)
    trough_dow = np.argmin(dow_factors_rev)
    lines.append(f"- **Ngày bán cao nhất**: {days[peak_dow]} "
                 f"(×{dow_factors_rev[peak_dow]:.2f}). "
                 f"**Ngày thấp nhất**: {days[trough_dow]} "
                 f"(×{dow_factors_rev[trough_dow]:.2f}).")
    lines.append(f"- **Holdout R² = {holdout_r2:.3f}** nghĩa là mô hình giải thích "
                 f"được {holdout_r2*100:.1f}% phương sai của doanh thu thực tế.")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def feature_importance_from_lgbm(
    sales_df: pd.DataFrame,
    target: str = "Revenue",
    seed: int = 42,
) -> pd.DataFrame:
    """Train a lightweight LightGBM on date features → output importance table.

    Used only for explainability (NOT for prediction — tree models cap trends).
    """
    from lightgbm import LGBMRegressor  # local import

    df = sales_df.copy()
    df["dow"]   = df["Date"].dt.dayofweek
    df["dom"]   = df["Date"].dt.day
    df["month"] = df["Date"].dt.month
    df["year"]  = df["Date"].dt.year
    df["doy"]   = df["Date"].dt.dayofyear
    df["week"]  = df["Date"].dt.isocalendar().week.astype(int)
    df["is_weekend"]    = (df["dow"] >= 5).astype(int)
    df["is_month_end"]  = (df["dom"] >= 28).astype(int)
    df["days_since"]    = (df["Date"] - df["Date"].min()).dt.days

    feats = ["dow", "dom", "month", "year", "doy", "week",
             "is_weekend", "is_month_end", "days_since"]

    X = df[feats].values
    y = df[target].values
    model = LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        random_state=seed, verbose=-1,
    ).fit(X, y)

    imp = pd.DataFrame({
        "feature": feats,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    imp["importance_pct"] = imp["importance"] / imp["importance"].sum() * 100
    return imp
