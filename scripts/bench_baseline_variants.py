"""Grid-score every baseline variant on both folds, pick best per fold.

Folds:
  primary_holdout: 2021-07-01 .. 2022-12-31 (18 months, matches test length)
  yearly_2022:     2022-01-01 .. 2022-12-31 (matches test base_year regime)

Second fold matters: on Kaggle test base_year=2022 (recovery), so a fold that
shares that base-year regime is a better LB proxy for the baseline family.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import itertools
import pandas as pd

from src.data_loader import load_sales
from src.cv import primary_holdout, yearly_walkforward
from src.evaluation import mae as _mae
from src.models.baseline import fit_and_predict, predict_with_prophet_trend


BENCH_PATH = ROOT / "reports" / "bench.csv"


def _log_row(row: dict) -> None:
    header = not BENCH_PATH.exists()
    pd.DataFrame([row]).to_csv(BENCH_PATH, mode="a", header=header, index=False)


def _score(fold_name, variant, params, y_rev, p_rev, y_cogs, p_cogs) -> float:
    r = _mae(y_rev, p_rev)
    c = _mae(y_cogs, p_cogs)
    total = r + c
    print(f"  [{fold_name}] {variant:<40s} rev={r:>11,.0f}  cogs={c:>11,.0f}  total={total:>11,.0f}")
    _log_row({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": variant,
        "params": params,
        "fold": fold_name,
        "revenue_mae": round(r, 2),
        "cogs_mae": round(c, 2),
        "total_mae": round(total, 2),
        "lb_score": "",
        "notes": "",
    })
    return total


def _run_variant(sales, fold, fold_name, growth_mode, season_window, log_transform, use_prophet_trend=False):
    train = sales[sales["Date"] <= fold.train_end]
    val = sales[fold.val_mask(sales["Date"])]

    if use_prophet_trend:
        p_rev = predict_with_prophet_trend(train, val["Date"], "Revenue", season_window, log_transform)
        p_cogs = predict_with_prophet_trend(train, val["Date"], "COGS", season_window, log_transform)
        variant = f"baseline_prophet_trend_sw{season_window}_log{int(log_transform)}"
    else:
        p_rev = fit_and_predict(train, val["Date"], "Revenue", growth_mode, season_window, log_transform).values
        p_cogs = fit_and_predict(train, val["Date"], "COGS", growth_mode, season_window, log_transform).values
        variant = f"baseline_{growth_mode}_sw{season_window}_log{int(log_transform)}"

    params = f"growth={growth_mode},season_window={season_window},log={log_transform},prophet_trend={use_prophet_trend}"
    return _score(fold_name, variant, params, val["Revenue"].values, p_rev, val["COGS"].values, p_cogs)


def main() -> None:
    sales = load_sales()

    folds = {
        "primary_holdout": primary_holdout(),
        "yearly_2022": yearly_walkforward(val_years=(2022,))[0],
    }

    modes = ["geo_full", "geo_last3", "linear_last3", "linear_last5", "flat"]
    season_windows = [None, 5, 3]
    log_flags = [False, True]

    results = []
    for fold_name, fold in folds.items():
        print(f"\n=== Fold: {fold_name} ({fold.val_start.date()}..{fold.val_end.date()}) ===")
        for gm, sw, lg in itertools.product(modes, season_windows, log_flags):
            total = _run_variant(sales, fold, fold_name, gm, sw, lg)
            results.append((fold_name, gm, sw, lg, False, total))

        # Prophet-trend anchor variants
        for sw, lg in itertools.product(season_windows, log_flags):
            total = _run_variant(sales, fold, fold_name, "prophet_trend", sw, lg, use_prophet_trend=True)
            results.append((fold_name, "prophet_trend", sw, lg, True, total))

    df = pd.DataFrame(results, columns=["fold", "growth", "season_window", "log", "prophet_trend", "total_mae"])
    print("\n=== Top 5 per fold ===")
    for fold_name in folds:
        print(f"\n{fold_name}:")
        top = df[df["fold"] == fold_name].sort_values("total_mae").head(5)
        print(top.to_string(index=False))

    print("\n=== Combined rank (geo mean of normalized MAE across both folds) ===")
    pivot = df.pivot_table(
        index=["growth", "season_window", "log", "prophet_trend"],
        columns="fold",
        values="total_mae",
    )
    pivot["geo_mean"] = (pivot["primary_holdout"] * pivot["yearly_2022"]) ** 0.5
    print(pivot.sort_values("geo_mean").head(10).to_string())


if __name__ == "__main__":
    main()
