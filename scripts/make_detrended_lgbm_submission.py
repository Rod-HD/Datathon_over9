"""Detrended LightGBM submission."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data_loader import load_sales, load_sample_submission, load_promotions
from src.features import build_feature_frame, discover_anomaly_dates
from src.models.detrended_lgbm import DetrendedLgbm
from src.evaluation import score_all
from src.cv import yearly_walkforward


def _attach_target(frame: pd.DataFrame, sales: pd.DataFrame, col: str) -> pd.DataFrame:
    lookup = sales.set_index("Date")[col]
    out = frame.copy()
    out[col] = pd.to_datetime(out["Date"]).map(lookup)
    return out


def _cv_for(col: str, full_frame: pd.DataFrame):
    print(f"\n--- Detrended LightGBM CV for {col} ---")
    for fold in yearly_walkforward(val_years=(2020, 2021, 2022)):
        train = full_frame[
            (full_frame["Date"] <= fold.train_end) & full_frame[col].notna()
        ].copy()
        val = full_frame[fold.val_mask(full_frame["Date"])].copy()

        model = DetrendedLgbm(target_col=col)
        model.fit(train, val_frame=val)
        pred = model.predict(val)
        score_all(val[col], pred, label=f"  {col} {fold.val_start.year}")


def main() -> None:
    sales = load_sales()
    sub = load_sample_submission()
    promos = load_promotions()

    anomaly_md = discover_anomaly_dates(sales, "Revenue", z_threshold=2.0)
    all_dates = pd.concat([sales["Date"], sub["Date"]]).reset_index(drop=True)
    frame = build_feature_frame(all_dates, sales=sales, promotions=promos, anomaly_md=anomaly_md)

    frame_rev = _attach_target(frame, sales, "Revenue")
    frame_cogs = _attach_target(frame, sales, "COGS")

    _cv_for("Revenue", frame_rev)
    _cv_for("COGS", frame_cogs)

    print("\n=== Refitting on full train (2012-2022) ===")
    train_rev = frame_rev[frame_rev["Revenue"].notna()].copy()
    train_cogs = frame_cogs[frame_cogs["COGS"].notna()].copy()
    test_mask = pd.to_datetime(frame["Date"]) >= pd.Timestamp("2023-01-01")
    test_frame = frame[test_mask].copy()

    h_rev = train_rev["Date"] >= pd.Timestamp("2022-01-01")
    rev_model = DetrendedLgbm(target_col="Revenue")
    rev_model.fit(train_rev[~h_rev], val_frame=train_rev[h_rev])
    rev_pred = rev_model.predict(test_frame)

    h_cogs = train_cogs["Date"] >= pd.Timestamp("2022-01-01")
    cogs_model = DetrendedLgbm(target_col="COGS")
    cogs_model.fit(train_cogs[~h_cogs], val_frame=train_cogs[h_cogs])
    cogs_pred = cogs_model.predict(test_frame)

    out = pd.DataFrame({
        "Date": pd.to_datetime(test_frame["Date"]).dt.strftime("%Y-%m-%d").values,
        "Revenue": rev_pred.round(2),
        "COGS": cogs_pred.round(2),
    })
    order = sub["Date"].dt.strftime("%Y-%m-%d").tolist()
    out = out.set_index("Date").loc[order].reset_index()

    out_path = ROOT / "submissions" / "submission_v3_detrended_lgbm.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved {len(out)} rows to {out_path}")
    print(out.head())
    print(f"\nLearned YoY growth (Revenue): {rev_model.growth_:.4f}")
    print(f"Top-15 feature importances (Revenue):\n{rev_model.feature_importance().head(15)}")


if __name__ == "__main__":
    main()
