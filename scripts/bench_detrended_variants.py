"""Bench DetrendedLgbm with different growth_mode."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data_loader import load_sales, load_promotions
from src.features import build_feature_frame, discover_anomaly_dates
from src.cv import primary_holdout, yearly_walkforward
from src.evaluation import mae as _mae
from src.models.detrended_lgbm import DetrendedLgbm


BENCH_PATH = ROOT / "reports" / "bench.csv"


def _log_row(row: dict) -> None:
    header = not BENCH_PATH.exists()
    pd.DataFrame([row]).to_csv(BENCH_PATH, mode="a", header=header, index=False)


def _attach(frame, sales, col):
    lookup = sales.set_index("Date")[col]
    out = frame.copy()
    out[col] = pd.to_datetime(out["Date"]).map(lookup)
    return out


def _run_fold(sales, promos, fold, fold_name):
    print(f"\n=== {fold_name} ===")
    train_sales = sales[sales["Date"] <= fold.train_end].copy()
    val_sales = sales[fold.val_mask(sales["Date"])].copy()

    anomaly_md = discover_anomaly_dates(train_sales, "Revenue", z_threshold=2.0)
    all_dates = pd.concat([train_sales["Date"], val_sales["Date"]]).reset_index(drop=True)
    frame = build_feature_frame(all_dates, sales=train_sales, promotions=promos, anomaly_md=anomaly_md)
    frame_rev = _attach(frame, sales, "Revenue")
    frame_cogs = _attach(frame, sales, "COGS")

    train_mask = pd.to_datetime(frame["Date"]) <= fold.train_end
    val_mask = fold.val_mask(pd.to_datetime(frame["Date"]))

    for col, f in [("Revenue", frame_rev), ("COGS", frame_cogs)]:
        tr_full = f[train_mask & f[col].notna()].copy()
        vl = f[val_mask].copy()
        cut = 365
        fit_part = tr_full.iloc[:-cut]
        hold = tr_full.iloc[-cut:]
        y = vl[col].values

        for mode in ("geo_full", "flat", "geo_last3", "linear_last3"):
            m = DetrendedLgbm(target_col=col, growth_mode=mode)
            m.fit(fit_part, val_frame=hold)
            pred = m.predict(vl)
            err = _mae(y, pred)
            print(f"  [{fold_name}] detrended growth={mode:<13s} {col:<8s} mae={err:>12,.0f}  (g={m.growth_:.4f})")
            _log_row({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": f"detrended_lgbm_{mode}",
                "params": f"growth_mode={mode},g={m.growth_:.4f}",
                "fold": fold_name,
                "revenue_mae": round(err, 2) if col == "Revenue" else "",
                "cogs_mae": round(err, 2) if col == "COGS" else "",
                "total_mae": "",
                "lb_score": "",
                "notes": col,
            })


def main() -> None:
    sales = load_sales()
    promos = load_promotions()
    _run_fold(sales, promos, primary_holdout(), "primary_holdout")
    _run_fold(sales, promos, yearly_walkforward(val_years=(2022,))[0], "yearly_2022")


if __name__ == "__main__":
    main()
