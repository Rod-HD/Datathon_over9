"""SHAP explainability for the detrended LightGBM model.

Produces:
  reports/figures/shap_summary_revenue.png
  reports/figures/shap_bar_revenue.png
  reports/figures/shap_summary_cogs.png
  reports/figures/shap_bar_cogs.png
  reports/figures/feature_importance.csv
  reports/figures/prophet_components_revenue.png
  reports/figures/prophet_components_cogs.png
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import logging
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from src.data_loader import load_sales, load_sample_submission, load_promotions
from src.features import build_feature_frame, discover_anomaly_dates
from src.models.detrended_lgbm import DetrendedLgbm
from src.models.prophet_model import ProphetForecaster


FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _attach_target(frame: pd.DataFrame, sales: pd.DataFrame, col: str) -> pd.DataFrame:
    lookup = sales.set_index("Date")[col]
    out = frame.copy()
    out[col] = pd.to_datetime(out["Date"]).map(lookup)
    return out


def _shap_for(col: str, frame_col: pd.DataFrame) -> pd.Series:
    train = frame_col[frame_col[col].notna()].copy()
    val_mask = train["Date"] >= pd.Timestamp("2022-01-01")
    train_fit = train[~val_mask].copy()
    val_fit = train[val_mask].copy()

    model = DetrendedLgbm(target_col=col)
    model.fit(train_fit, val_frame=val_fit)

    X = model._prepare(train)[model.feature_cols]
    explainer = shap.TreeExplainer(model.model_)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[0]

    shap.summary_plot(sv, X, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"shap_summary_{col.lower()}.png", dpi=130, bbox_inches="tight")
    plt.close()

    shap.summary_plot(sv, X, plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"shap_bar_{col.lower()}.png", dpi=130, bbox_inches="tight")
    plt.close()

    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=model.feature_cols, name=col)
    return mean_abs.sort_values(ascending=False)


def _prophet_components(col: str, sales: pd.DataFrame, anomaly_md: pd.DataFrame) -> None:
    pm = ProphetForecaster(target_col=col, anomaly_md=anomaly_md)
    pm.fit(sales)
    future = pm.model_.make_future_dataframe(periods=548, freq="D")
    forecast = pm.model_.predict(future)
    fig = pm.model_.plot_components(forecast)
    fig.savefig(FIG_DIR / f"prophet_components_{col.lower()}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    sales = load_sales()
    sub = load_sample_submission()
    promos = load_promotions()

    anomaly_md = discover_anomaly_dates(sales, "Revenue", z_threshold=2.0)
    all_dates = pd.concat([sales["Date"], sub["Date"]]).reset_index(drop=True)
    frame = build_feature_frame(all_dates, sales=sales, promotions=promos, anomaly_md=anomaly_md)

    frame_rev = _attach_target(frame, sales, "Revenue")
    frame_cogs = _attach_target(frame, sales, "COGS")

    print("=== SHAP for Revenue ===")
    rev_imp = _shap_for("Revenue", frame_rev)
    print(rev_imp.head(15))

    print("\n=== SHAP for COGS ===")
    cogs_imp = _shap_for("COGS", frame_cogs)
    print(cogs_imp.head(15))

    combined = pd.concat([rev_imp, cogs_imp], axis=1).fillna(0.0)
    combined.index.name = "feature"
    combined.to_csv(FIG_DIR / "feature_importance.csv")
    print(f"\nSaved SHAP figures + CSV to {FIG_DIR}")

    print("\n=== Prophet components ===")
    _prophet_components("Revenue", sales, anomaly_md)
    _prophet_components("COGS", sales, anomaly_md)
    print(f"Saved Prophet component plots to {FIG_DIR}")


if __name__ == "__main__":
    main()
