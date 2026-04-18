"""Prophet grid search on primary_holdout + yearly_2022.

Rank combos by geo-mean of total MAE across the two folds (primary_holdout
mimics the 18-month horizon; yearly_2022 is the closest fully-labeled year).
Writes every trial to reports/bench.csv and prints the top-10.
"""
from __future__ import annotations

import itertools
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.data_loader import load_sales, load_promotions
from src.features import build_feature_frame, discover_anomaly_dates
from src.cv import primary_holdout, yearly_walkforward
from src.evaluation import mae as _mae
from src.models.prophet_model import ProphetForecaster


logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

BENCH_PATH = ROOT / "reports" / "bench.csv"


GRID = {
    "changepoint_prior_scale": [0.05, 0.1, 0.25],
    "seasonality_prior_scale": [10.0, 30.0],
    "yearly_seasonality": [20],
    "changepoint_range": [0.85, 0.95],
}
REGRESSOR_SETS = [
    [],  # baseline — no extra regressors
    ["promo_active_count", "is_anomaly_day"],
]


def _log_row(row: dict) -> None:
    header = not BENCH_PATH.exists()
    pd.DataFrame([row]).to_csv(BENCH_PATH, mode="a", header=header, index=False)


def _build_frame(sales, promos, train_end, val_end):
    train_sales = sales[sales["Date"] <= train_end].copy()
    val_slice = sales[(sales["Date"] > train_end) & (sales["Date"] <= val_end)].copy()
    anomaly_md = discover_anomaly_dates(train_sales, "Revenue", z_threshold=2.0)
    all_dates = pd.concat([train_sales["Date"], val_slice["Date"]]).reset_index(drop=True)
    frame = build_feature_frame(all_dates, sales=train_sales, promotions=promos, anomaly_md=anomaly_md)
    return train_sales, val_slice, frame, anomaly_md


def _run_prophet(train_sales, val_sales, frame, anomaly_md, col, cfg, regressors):
    m = ProphetForecaster(
        target_col=col,
        anomaly_md=anomaly_md,
        changepoint_prior_scale=cfg["changepoint_prior_scale"],
        seasonality_prior_scale=cfg["seasonality_prior_scale"],
        yearly_seasonality=cfg["yearly_seasonality"],
        changepoint_range=cfg["changepoint_range"],
        extra_regressors=list(regressors),
    )
    rf = frame[["Date"] + [r for r in regressors if r in frame.columns]] if regressors else None
    m.fit(train_sales[["Date", col]].rename(columns={}), regressor_frame=rf)
    pred = m.predict(val_sales["Date"]).values
    return _mae(val_sales[col].values, pred)


def _cfg_str(cfg, regressors):
    base = ",".join(f"{k[:4]}={v}" for k, v in cfg.items())
    reg = "+".join(r[:5] for r in regressors) or "none"
    return f"{base},reg={reg}"


def main() -> None:
    sales = load_sales()
    promos = load_promotions()

    holdouts = [
        ("primary_holdout", primary_holdout()),
        ("yearly_2022", yearly_walkforward(val_years=(2022,))[0]),
    ]
    prepared = {name: _build_frame(sales, promos, fold.train_end, fold.val_end)
                for name, fold in holdouts}

    combos = list(itertools.product(*GRID.values()))
    keys = list(GRID.keys())

    results = []
    total = len(combos) * len(REGRESSOR_SETS)
    done = 0
    t0 = time.time()

    for values in combos:
        cfg = dict(zip(keys, values))
        for regressors in REGRESSOR_SETS:
            done += 1
            row_scores = {}
            failed = False
            for name, (tr_sales, vl_sales, frame, anomaly_md) in prepared.items():
                for col in ("Revenue", "COGS"):
                    try:
                        err = _run_prophet(tr_sales, vl_sales, frame, anomaly_md, col, cfg, regressors)
                    except Exception as e:
                        failed = True
                        print(f"  [{name}/{col}] FAIL {_cfg_str(cfg, regressors)}: {e}")
                        break
                    row_scores[(name, col)] = err
                if failed:
                    break
            if failed:
                continue
            ph_total = row_scores[("primary_holdout", "Revenue")] + row_scores[("primary_holdout", "COGS")]
            y22_total = row_scores[("yearly_2022", "Revenue")] + row_scores[("yearly_2022", "COGS")]
            geo = float(np.sqrt(ph_total * y22_total))
            results.append({
                "cfg": cfg,
                "regressors": list(regressors),
                "ph_total": ph_total,
                "y22_total": y22_total,
                "geo": geo,
            })
            cfg_s = _cfg_str(cfg, regressors)
            print(f"[{done:>3}/{total}] {cfg_s} | ph={ph_total:>12,.0f}  y22={y22_total:>12,.0f}  geo={geo:>12,.0f}")
            _log_row({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": "prophet_grid",
                "params": cfg_s,
                "fold": "ph+y22",
                "revenue_mae": "",
                "cogs_mae": "",
                "total_mae": round(geo, 2),
                "lb_score": "",
                "notes": f"ph={ph_total:.0f};y22={y22_total:.0f}",
            })

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed / 60:.1f} min, valid {len(results)}/{total}")

    if not results:
        print("No successful runs.")
        return

    results.sort(key=lambda r: r["geo"])
    print("\n===== TOP 10 by geo(ph_total, y22_total) =====")
    for r in results[:10]:
        cfg_s = _cfg_str(r["cfg"], r["regressors"])
        print(f"  geo={r['geo']:>12,.0f}  ph={r['ph_total']:>12,.0f}  y22={r['y22_total']:>12,.0f}  | {cfg_s}")

    best = results[0]
    print("\nBest config:")
    print(f"  {best['cfg']}")
    print(f"  regressors = {best['regressors']}")

    # persist best cfg for the submission script
    best_path = ROOT / "reports" / "prophet_best.txt"
    with open(best_path, "w") as f:
        f.write(repr({"cfg": best["cfg"], "regressors": best["regressors"],
                      "ph_total": best["ph_total"], "y22_total": best["y22_total"],
                      "geo": best["geo"]}))
    print(f"Saved best to {best_path}")


if __name__ == "__main__":
    main()
