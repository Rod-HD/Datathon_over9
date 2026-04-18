"""Bench LightGBM variants on both folds."""
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
from src.models.lgbm import LgbmForecaster
from src.models.baseline import fit_and_predict as baseline_predict


BENCH_PATH = ROOT / "reports" / "bench.csv"


def _log_row(row: dict) -> None:
    header = not BENCH_PATH.exists()
    pd.DataFrame([row]).to_csv(BENCH_PATH, mode="a", header=header, index=False)


def _attach(frame: pd.DataFrame, sales: pd.DataFrame, col: str) -> pd.DataFrame:
    lookup = sales.set_index("Date")[col]
    out = frame.copy()
    out[col] = pd.to_datetime(out["Date"]).map(lookup)
    return out


def _variant_model(variant: str, col: str) -> LgbmForecaster:
    if variant == "plain_current":
        return LgbmForecaster(target_col=col, log_transform=True, drop_trend_leak=False)
    if variant == "plain_noyear":
        return LgbmForecaster(target_col=col, log_transform=True, drop_trend_leak=True)
    if variant == "residual":
        return LgbmForecaster(target_col=col, residual_col="base_pred", drop_trend_leak=True)
    if variant == "residual_keep_base":
        # keep base_pred as feature; still log-transform; NOT residual target
        m = LgbmForecaster(target_col=col, log_transform=True, drop_trend_leak=True)
        return m
    raise ValueError(variant)


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
        f_tr = f[train_mask].copy()
        f_vl = f[val_mask].copy()
        f_tr["base_pred"] = baseline_predict(train_sales, f_tr["Date"], col, growth_mode="flat", season_window=5).values
        f_vl["base_pred"] = baseline_predict(train_sales, f_vl["Date"], col, growth_mode="flat", season_window=5).values
        f_tr = f_tr[f_tr[col].notna()].copy()
        y_val = f_vl[col].values

        # Early-stopping cut: last 365 rows of train
        n = len(f_tr)
        cut_n = min(365, max(n // 10, 30))
        fit_part = f_tr.iloc[:-cut_n]
        hold_part = f_tr.iloc[-cut_n:]

        for variant in ("plain_current", "plain_noyear", "residual", "residual_keep_base"):
            if variant == "plain_current":
                fp = fit_part.drop(columns=["base_pred"])
                hp = hold_part.drop(columns=["base_pred"])
                vp = f_vl.drop(columns=["base_pred"])
            elif variant == "plain_noyear":
                fp = fit_part.drop(columns=["base_pred"])
                hp = hold_part.drop(columns=["base_pred"])
                vp = f_vl.drop(columns=["base_pred"])
            else:
                fp, hp, vp = fit_part, hold_part, f_vl

            m = _variant_model(variant, col)
            m.fit(fp, val_frame=hp)
            pred = m.predict(vp)
            err = _mae(y_val, pred)
            print(f"  [{fold_name}] {variant:<22s} {col:<8s} mae={err:>12,.0f}")
            _log_row({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": f"lgbm_{variant}",
                "params": f"variant={variant}",
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
