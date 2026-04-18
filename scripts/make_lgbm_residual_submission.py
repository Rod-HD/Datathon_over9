"""LightGBM residual-on-baseline submission (v6)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data_loader import load_sales, load_sample_submission, load_promotions
from src.features import build_feature_frame, discover_anomaly_dates
from src.models.lgbm import LgbmForecaster
from src.models.baseline import fit_and_predict as baseline_predict


def _attach(frame: pd.DataFrame, sales: pd.DataFrame, col: str) -> pd.DataFrame:
    lookup = sales.set_index("Date")[col]
    out = frame.copy()
    out[col] = pd.to_datetime(out["Date"]).map(lookup)
    return out


def _fit_and_predict(col: str, frame: pd.DataFrame, sales: pd.DataFrame, test_frame: pd.DataFrame):
    train = frame[frame[col].notna()].copy()
    train["base_pred"] = baseline_predict(sales, train["Date"], col, growth_mode="flat", season_window=5).values
    test = test_frame.copy()
    test["base_pred"] = baseline_predict(sales, test["Date"], col, growth_mode="flat", season_window=5).values

    # Early-stopping split: last 365 rows of train
    cut = 365
    fit_part = train.iloc[:-cut]
    hold_part = train.iloc[-cut:]

    m = LgbmForecaster(target_col=col, residual_col="base_pred", drop_trend_leak=True)
    m.fit(fit_part, val_frame=hold_part)
    return m.predict(test), m


def main() -> None:
    sales = load_sales()
    sub = load_sample_submission()
    promos = load_promotions()

    anomaly_md = discover_anomaly_dates(sales, "Revenue", z_threshold=2.0)
    all_dates = pd.concat([sales["Date"], sub["Date"]]).reset_index(drop=True)
    frame = build_feature_frame(all_dates, sales=sales, promotions=promos, anomaly_md=anomaly_md)

    frame_rev = _attach(frame, sales, "Revenue")
    frame_cogs = _attach(frame, sales, "COGS")

    test_mask = pd.to_datetime(frame["Date"]) >= pd.Timestamp("2023-01-01")
    test_frame = frame[test_mask].copy()

    rev_pred, rev_model = _fit_and_predict("Revenue", frame_rev, sales, test_frame)
    cogs_pred, _ = _fit_and_predict("COGS", frame_cogs, sales, test_frame)

    out = pd.DataFrame({
        "Date": pd.to_datetime(test_frame["Date"]).dt.strftime("%Y-%m-%d").values,
        "Revenue": rev_pred.round(2),
        "COGS": cogs_pred.round(2),
    })
    order = sub["Date"].dt.strftime("%Y-%m-%d").tolist()
    out = out.set_index("Date").loc[order].reset_index()

    out_path = ROOT / "submissions" / "submission_v6_lgbm_residual.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} rows to {out_path}")
    print(out.head())
    print("\nTop-10 features (Revenue residual):")
    print(rev_model.feature_importance().head(10))


if __name__ == "__main__":
    main()
