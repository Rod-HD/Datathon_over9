"""Train LightGBM on train period, predict test, export submission.

CV on yearly walk-forward first, then refit on 2012-2022 for final submission.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data_loader import load_sales, load_sample_submission, load_promotions
from src.features import build_feature_frame, discover_anomaly_dates
from src.models.lgbm import LgbmForecaster
from src.evaluation import score_all
from src.cv import yearly_walkforward


def _attach_target(frame: pd.DataFrame, sales: pd.DataFrame, col: str) -> pd.DataFrame:
    lookup = sales.set_index("Date")[col]
    out = frame.copy()
    out[col] = pd.to_datetime(out["Date"]).map(lookup)
    return out


def _cv_for(col: str, full_frame: pd.DataFrame):
    print(f"\n--- LightGBM CV for {col} ---")
    for fold in yearly_walkforward(val_years=(2020, 2021, 2022)):
        train = full_frame[
            (full_frame["Date"] <= fold.train_end) & full_frame[col].notna()
        ].copy()
        val = full_frame[fold.val_mask(full_frame["Date"])].copy()

        model = LgbmForecaster(target_col=col)
        model.fit(train, val_frame=val)
        pred = model.predict(val)
        score_all(val[col], pred, label=f"  {col} {fold.val_start.year}")


def main() -> None:
    sales = load_sales()
    sub = load_sample_submission()
    promos = load_promotions()

    anomaly_md = discover_anomaly_dates(sales, "Revenue", z_threshold=2.0)
    all_dates = pd.concat([sales["Date"], sub["Date"]]).reset_index(drop=True)
    frame = build_feature_frame(
        all_dates, sales=sales, promotions=promos, anomaly_md=anomaly_md
    )

    frame_rev = _attach_target(frame, sales, "Revenue")
    frame_cogs = _attach_target(frame, sales, "COGS")

    _cv_for("Revenue", frame_rev)
    _cv_for("COGS", frame_cogs)

    print("\n=== Refitting on full train (2012-2022) ===")
    train_rev = frame_rev[frame_rev["Revenue"].notna()].copy()
    train_cogs = frame_cogs[frame_cogs["COGS"].notna()].copy()
    test_mask = pd.to_datetime(frame["Date"]) >= pd.Timestamp("2023-01-01")
    test_frame = frame[test_mask].copy()

    # Use the last training year as validation for early stopping at refit
    holdout_mask_rev = train_rev["Date"] >= pd.Timestamp("2022-01-01")
    rev_model = LgbmForecaster(target_col="Revenue")
    rev_model.fit(train_rev[~holdout_mask_rev], val_frame=train_rev[holdout_mask_rev])
    rev_pred = rev_model.predict(test_frame)

    holdout_mask_cogs = train_cogs["Date"] >= pd.Timestamp("2022-01-01")
    cogs_model = LgbmForecaster(target_col="COGS")
    cogs_model.fit(train_cogs[~holdout_mask_cogs], val_frame=train_cogs[holdout_mask_cogs])
    cogs_pred = cogs_model.predict(test_frame)

    out = pd.DataFrame({
        "Date": pd.to_datetime(test_frame["Date"]).dt.strftime("%Y-%m-%d").values,
        "Revenue": rev_pred.round(2),
        "COGS": cogs_pred.round(2),
    })
    # Ensure ordering matches sample_submission exactly
    order = sub["Date"].dt.strftime("%Y-%m-%d").tolist()
    out = out.set_index("Date").loc[order].reset_index()

    out_path = ROOT / "submissions" / "submission_v1_lightgbm.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved {len(out)} rows to {out_path}")
    print(out.head())
    print("\nTop-15 feature importances (Revenue):")
    print(rev_model.feature_importance().head(15))


if __name__ == "__main__":
    main()
