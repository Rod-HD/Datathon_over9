# DATATHON 2026 — Part 3: Sales Forecasting

Daily Revenue & COGS forecasting cho công ty thương mại điện tử thời trang Việt Nam.
- **Train**: `data/sales.csv` (2012-07-04 → 2022-12-31)
- **Test**: 548 ngày (2023-01-01 → 2024-07-01)
- **Output**: `submissions/submission.csv`

## Chạy pipeline

```bash
# 1. Tiền xử lý dữ liệu raw -> preprocessed/
python scripts/preprocess_data.py

# 2. Train + dự báo -> submissions/submission.csv
python scripts/forecast.py
```

## Pipeline

```
sales.csv (2012-2022)
   │
   ├── Prophet (trend + yearly seasonality)
   ├── linear_last3 baseline (log-linear extrapolation, sw=6)
   ├── Theta (M3/M4 winning method)
   └── Holt-Winters (Triple ES, weekly seasonality)
   │
   ▼
NNLS blend on 2022 holdout (mean-based → optimal RMSE/R²)
   │
   ▼
Auto-tuned scale (Rev/COGS) + DoW correction strength
   │
   ▼
COGS cap (≤ 1.05 × Revenue) + clip non-negative + round 2dp
   │
   ▼
submission.csv
```

## Files

- `scripts/preprocess_data.py` — preprocessing logic từ notebook
- `scripts/forecast.py` — main pipeline, output `submission.csv`
- `src/data_loader.py` — data I/O
- `src/postprocess.py` — clip, round, COGS cap
- `src/explainability.py` — driver report + feature importance
- `src/models/baseline.py` — linear_last3 seasonal baseline
- `src/models/prophet_model.py` — Prophet wrapper
- `src/models/theta_model.py` — Theta forecaster

## Reproducibility

- Random seed = 42 (set ở đầu `forecast.py`)
- Auto-tune dùng grid search deterministic (không random)
- Mọi model dùng full training data 2012-2022 để fit final

## Explainability

Sau khi chạy `forecast.py`:
- `reports/drivers.md` — model contribution, DoW effect, YoY growth, holdout metrics
- `reports/feature_importance.csv` — LightGBM feature importance từ date features

## Metrics đánh giá

| Metric | Hướng tốt | Tối ưu bởi |
|---|---|---|
| MAE | thấp | median |
| RMSE | thấp | mean |
| R² | cao (≤1) | mean |

Pipeline dùng NNLS-blend (mean-based) để tối ưu cả 3 metrics; auto-tune scale + DoW
balance giữa MAE và RMSE.
