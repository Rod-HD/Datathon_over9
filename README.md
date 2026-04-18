# DATATHON 2026 — Part 3: Sales Forecasting

Daily Revenue & COGS forecasting for a Vietnamese fashion retailer, 2023-01-01 to 2024-07-01 (548 days).
Part of [DATATHON 2026 Round 1](https://www.kaggle.com/competitions/datathon-2026-round-1) by VinTelligence / VinUniversity.

## Results (CV mean MAE across 2020 / 2021 / 2022 folds)

| Model | Revenue MAE | COGS MAE |
|---|---:|---:|
| Seasonal-naive baseline | 610,020 | 523,701 |
| Detrended LightGBM | 800,923 | 705,244 |
| Prophet (log + multiplicative) | 860,218 | 707,396 |
| **Ensemble (inverse-MAE)** | **607,514** | **523,273** |

Submitted: `submissions/submission_v4_ensemble.csv`.

## Approach

Only `sales.csv` spans the test horizon, so the pipeline is a pure time-series stack:

1. **Calendar + Fourier** features (annual order=6, weekly order=3, monthly order=3).
2. **Vietnamese holidays** discovered via z-score anomaly detection on training Revenue — no external calendar (rule: no external data).
3. **Lag / rolling** features at lag ≥ 548 days so every feature at test time uses only pre-2023 values.
4. Three base models:
   - **Seasonal-naive baseline**: day-of-year profile × compound YoY growth.
   - **Detrended LightGBM**: divide out the exponential trend, let LGBM predict the seasonal ratio, scale back.
   - **Prophet**: log1p target + multiplicative seasonality + anomaly-derived holidays.
5. **Ensemble**: inverse-MAE weighted average derived from walk-forward CV.
6. **SHAP** on the detrended LGBM + Prophet components for explainability.

## Reproduce

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows (bash: source .venv/Scripts/activate)
pip install -r requirements.txt

# Restore ./data from the Kaggle dataset (not committed)

python run_pipeline.py
```

Fixed seed: `42`. Python 3.13, tested on Windows 11.

## Layout

```
src/
  data_loader.py            # single loader for all 14 CSVs
  features.py               # calendar, Fourier, lag/rolling, holiday discovery
  cv.py                     # walk-forward folds
  evaluation.py             # MAE / RMSE / R²
  ensemble.py               # inverse-MAE weights + weighted average
  models/
    baseline.py             # seasonal-naive × YoY growth
    detrended_lgbm.py       # trend-divided LightGBM
    prophet_model.py        # Prophet wrapper
    lgbm.py                 # plain LightGBM (for reference)
scripts/
  audit_data.py             # date-range + null audit
  make_*_submission.py      # per-model submissions
  evaluate_ensemble_cv.py   # fold-level ensemble MAE
  explain_shap.py           # SHAP + Prophet components
reports/
  figures/                  # SHAP plots, Prophet components, feature_importance.csv
  neurips/                  # 4-page report (LaTeX)
submissions/                # versioned CSVs (v0..v4)
run_pipeline.py             # end-to-end reproduction
```

## Key findings

- **Trend is non-monotonic**: revenue peaked in 2016, bottomed ~2019, rebounded 2022 → compound YoY geometric mean is ~0.96 per year. Models that assume monotonic growth (vanilla LGBM with `year` feature) extrapolate poorly.
- **Seasonality > trend** at day granularity: Fourier annual + weekly + the discovered-holiday flag carry most of the signal.
- **Ensemble gain is small** (~0.4% Revenue MAE over baseline alone). Baseline is remarkably hard to beat on an 18-month horizon with a single source table.

See `reports/figures/` for SHAP summary plots and Prophet component decomposition.

## Constraints compliance

- No external data (holidays discovered from train patterns).
- No test-period Revenue/COGS leaked into features (lag floor = 548 days).
- Deterministic (seed 42 across LGBM; Prophet uses its default MAP).
- Full reproduction via `run_pipeline.py`.
