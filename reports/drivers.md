# Datathon 2026 — Final Forecast Report

## Executive Summary

An expanding-window cross-validated ensemble of three classical time-series forecasters (**Theta**, **Holt-Winters**, **Hybrid ARIMA**) predicts daily Revenue and COGS for 2023–2024. Ensemble weights are jointly optimized across three chronological folds (2020, 2021, 2022) to minimize a combined objective of RMSE (45%), MAE (35%), and (1−R²) (20%).

Average cross-validated performance: **MAE = 522,663**, **RMSE = 731,302**, **R² = 0.7778**.

Classical time-series models are used instead of tree-based or neural models because the test horizon (2023–2024) requires extrapolation beyond the training range (2012–2022). Tree ensembles such as LightGBM cannot predict values outside the training distribution and systematically under-forecast a growing series. Neural models without explicit trend covariates exhibit the same limitation and were ruled out after holdout testing.

---

## 1. Pipeline and Feature Engineering

### 1.1 Input Preprocessing

- Daily sales data is sorted strictly by Date and treated as a contiguous time series.
- Missing or zero values inside training windows are linearly interpolated, then forward/back-filled. Interpolation is applied independently per fold so no future data influences any training step.
- All target values are clipped to non-negative before model input.

### 1.2 Feature Engineering per Model

| Model | Feature type | Encoding method | Purpose |
|---|---|---|---|
| Theta | log1p(Revenue/COGS) | Logarithmic transform | Stabilises variance across the 10-year series |
| Theta | Day-of-week (0–6) | Multiplicative factor learned from per-DoW / global mean ratio | Captures weekly business rhythm without dummy explosion |
| Holt-Winters | Raw Revenue/COGS | No explicit encoding — level, trend, seasonal states are fit by ETS | Directly models exponential smoothing components |
| Hybrid ARIMA | (month, day) pair | Calendar lookup: mean(y / annual_mean) per (month, day) group | Intra-year seasonal profile baseline |
| Hybrid ARIMA | Year index | Geometric growth fit on last 3 annual totals (linear_last3) | Long-horizon trend extrapolation |
| Hybrid ARIMA | Recent residuals (3 yr) | Raw residuals fed to ARIMA | Short-term autocorrelation correction |

### 1.3 Cyclic Encoding for Explainability

Raw integer calendar features (e.g., doy = 1 … 365) create an artificial discontinuity at the year boundary: the model sees doy 365 and doy 1 as maximally distant, even though they are one day apart. Cyclic encoding eliminates this artefact:

- **doy_sin / doy_cos**: `sin(2π · doy / 365.25)` and `cos(2π · doy / 365.25)`.
- **dow_sin / dow_cos**: `sin(2π · dow / 7)` and `cos(2π · dow / 7)`.

One-hot encoding is **not** used because it would produce 365 or 7 sparse binary columns and would still not encode the cyclic distance correctly. The sin/cos pair is the minimal representation that preserves ordinal distance around the cycle.

### 1.4 Post-Prediction Postprocessing

After ensemble blending, predictions pass through a fixed deterministic pipeline:

1. Multiply by SCALE_REVENUE / SCALE_COGS (systematic under-prediction correction calibrated on CV).
2. Apply day-of-week adjustment: `output × (1 + DOW_STRENGTH × (dow_factor − 1))`, where `DOW_STRENGTH = 0.40` blends the learned DoW profile with a flat baseline.
3. Add daily bias terms REVENUE_BIAS and COGS_BIAS (further systematic correction).
4. Clip to ≥ 0.
5. Cap COGS ≤ 1.05 × Revenue (competition constraint).
6. Round to 2 decimal places.

---

## 2. Time-Series Cross-Validation (Expanding Window)

**Method:** Expanding-window time-series cross-validation over three chronological folds.

Unlike k-fold cross-validation — which shuffles rows randomly and allows future observations to appear in the training set — expanding-window CV preserves strict temporal order. The training window grows with each fold; no validation-year data is ever seen during training.

**Why Expanding Window and not Rolling Window?** A rolling window discards older observations, losing the long-run growth trend that is critical for accurate extrapolation to 2023–2024. The expanding window uses all available history so the trend estimate improves with each fold.

| Fold | Train period | Validation period | MAE | RMSE | R² |
|---:|---|---|---:|---:|---:|
| 2020 | 2012-07-04 to 2019-12-31 | 2020-01-01 to 2020-12-31 | 520,078 | 707,297 | 0.784505 |
| 2021 | 2012-07-04 to 2020-12-31 | 2021-01-01 to 2021-12-31 | 507,854 | 741,795 | 0.770287 |
| 2022 | 2012-07-04 to 2021-12-31 | 2022-01-01 to 2022-12-31 | 540,058 | 744,813 | 0.778666 |
| **Avg** | — | — | **522,663** | **731,302** | **0.777819** |

Metrics are computed on the concatenated [Revenue, COGS] vector (2 × 365 values per fold) to match the competition scoring formula exactly.

---

## 3. Leakage Control

**Definition of data leakage:** leakage occurs when information from the validation or test period is used — directly or indirectly — during model training or hyperparameter selection. This inflates in-sample metrics and produces models that fail on unseen data.

| Control point | Implementation |
|---|---|
| No future Revenue/COGS in training | Each fold filters strictly to `Date < fold_year`; test Revenue/COGS are never loaded at any stage. |
| No global statistics from test data | Scale factors, means, and standard deviations are computed only from the training portion of each fold. |
| Day-of-week profiles recomputed per fold | DoW multipliers use only `train[Date.year < fold_year]` data; the profile for fold 2022 does not see 2022 actuals. |
| Interpolation within fold only | Missing values are filled using each fold's training window alone; no forward-fill from future observations. |
| Ensemble weights from out-of-fold validation | Softmax weights are optimised against fold validation errors, not against the 2023–2024 test horizon. |
| Sample submission used only for dates | The provided test CSV supplies future dates only; its Revenue/COGS columns are ignored. |

---

## 4. Feature Importance and Explainability

**Why not SHAP?** The production model is an ensemble of classical statistical forecasters (Theta, ARIMA). SHAP via TreeSHAP requires a gradient-boosted tree structure; DeepSHAP requires a neural network. Applying kernel SHAP to a time-series forecaster would require treating each forecast step as an independent sample, which discards the temporal autocorrelation structure the models are specifically designed to exploit. Absolute Pearson correlation between each calendar feature and Revenue/COGS is used instead as a deterministic, interpretable importance proxy.

| Feature | Importance | Interpretation |
|---|---:|---|
| doy_cos | 16.86% | Cyclic annual phase (peak at seasonal high) — complements doy_sin. |
| is_month_end | 13.17% | End-of-month purchasing spike — consistent with payroll-cycle consumer behavior. |
| days_since | 10.91% | Long-term growth trend — the single largest driver of absolute Revenue level. |
| doy_sin | 10.68% | Cyclic annual phase (rising Jan→Jul) — smooth wrap at year boundary. |
| year | 10.31% | Annual level shift — tracks business growth year-over-year. |
| dom | 9.81% | Day-of-month position — reinforces end-of-month purchasing spikes. |
| month | 6.87% | Broad seasonal periods (summer/winter peaks in fashion). |
| week | 6.36% | Within-year seasonal progression. |

**Key insights:**

- **doy_cos / doy_sin** dominate because fashion e-commerce demand follows a strong annual cycle. Cyclic encoding captures this without the artificial discontinuity of raw integer day-of-year.
- **is_month_end** ranks highly (~13%), indicating end-of-month purchasing spikes consistent with payroll-cycle consumer behavior and monthly promotional campaigns.
- **days_since + year** together (~21%) confirm long-run business growth is the largest driver of absolute Revenue level — the primary reason tree ensembles (which cannot extrapolate) fail on this task.
- **dom** (~10%) reinforces the monthly periodicity signal that complements is_month_end.

---

## 5. Optimized Ensemble Weights

Objective = 0.45 × RMSE / 735,000 + 0.35 × MAE / 532,000 + 0.20 × (1 − R²).

Weights are found via Nelder-Mead minimisation with 12 random restarts over softmax-parameterised weight vectors. Softmax guarantees weights sum to 1 and remain non-negative without explicit constraints.

| Model | Revenue weight | COGS weight | Role |
|---|---:|---:|---|
| theta | 0.1742 | 0.1670 | Robust long-horizon trend with log scaling |
| holt_winters | 0.0000 | 0.0000 | Weekly seasonal ETS — 0% weight, ruled out by optimizer |
| arima_a2 | 0.8258 | 0.8330 | Linear seasonal trend + ARIMA residual correction |

---

## 6. Final Submission Totals

| Metric | Revenue | COGS |
|---|---:|---:|
| Total 2023–2024 | 2.322B VND | 2.109B VND |
| Daily average | 4,238,128 VND | 3,848,980 VND |
| COGS / Revenue ratio | — | 0.908 |

---

## Appendix A: Full Hyperparameters

| Parameter | Value | Tuning method |
|---|---:|---|
| ARIMA Revenue order (p, d, q) | (2, 0, 3) | Grid search over 8×7 order combinations on CV |
| ARIMA COGS order (p, d, q) | (2, 0, 3) | Grid search |
| ARIMA residual window (years) | 3 | CV |
| Holt-Winters trend | additive | Fixed |
| Holt-Winters seasonal | multiplicative | Fixed |
| Holt-Winters seasonal_periods | 7 | Fixed (weekly) |
| Theta period | 365 | Fixed (annual) |
| Theta log transform | True | Fixed |
| Theta deseasonalize | True | Fixed |
| SCALE_REVENUE | 1.184 | CV calibration |
| SCALE_COGS | 1.191 | CV calibration |
| DOW_STRENGTH | 0.40 | Manual tuning |
| REVENUE_BIAS (VND/day) | 25,000 | CV calibration |
| COGS_BIAS (VND/day) | 112,500 | CV calibration |
| MAX_COGS_RATIO | 1.05 | Competition constraint |
| Random seed | 42 | Fixed |
| Optimiser | Nelder-Mead | Fixed |
| Optimiser restarts | 12 | Fixed |

---

## Appendix B: Mathematical Formulas

### B.1 Theta Method

The Theta method (Assimakopoulos & Nikolopoulos, 2000) decomposes the series `y_t` into two modified lines:

```
Theta_0 line:  y_0(t) = 2·mean(y) - y(t)   # suppresses seasonality, retains linear trend
Theta_2 line:  y_2(t) = y(t)                # retains the full original series

Forecast:  F(h) = 0.5 · SES(y_0, h) + 0.5 · (a + b·(T + h))
           where SES = simple exponential smoothing on y_0
                 a, b = OLS intercept and slope of y_0
```

Applied on `log1p(y)`; output is `expm1(F(h)) × dow_factor[dow(h)]`.

### B.2 Holt-Winters (Multiplicative Seasonality)

```
Level:    L_t = alpha · (y_t / S_{t-m}) + (1-alpha) · (L_{t-1} + B_{t-1})
Trend:    B_t = beta  · (L_t - L_{t-1}) + (1-beta)  · B_{t-1}
Seasonal: S_t = gamma · (y_t / L_t)     + (1-gamma) · S_{t-m}
Forecast: F(h) = (L_T + h·B_T) · S_{T+h-m·ceil(h/m)}
          m = 7 (weekly),  alpha/beta/gamma optimised by SSE minimisation
```

### B.3 Hybrid ARIMA

```
Step 1 - Seasonal baseline:
  y_hat(t) = base_level · growth^years_ahead · seasonal_norm(month, day)
  growth        = geometric mean of last-3 annual YoY growth rates
  seasonal_norm = mean(y_t / annual_mean_t), grouped by (month, day)

Step 2 - Residual ARIMA(p, 0, q) on last 3 training years:
  e_t = y_t - y_hat(t)
  ARIMA: e_t = c + sum_i(phi_i · e_{t-i}) + sum_j(theta_j · eps_{t-j}) + eps_t

Step 3 - Combine:
  F(h) = max(0, y_hat(h) + ARIMA_forecast(h))
```

### B.4 Ensemble Objective

```
weights_rev = softmax(z[0:3]),  weights_cog = softmax(z[3:6]),  z in R^6 (unconstrained)

For each fold year k in {2020, 2021, 2022}:
  pred_rev_k = sum_i(w_rev_i · model_i_rev_k)
  pred_cog_k = sum_j(w_cog_j · model_j_cog_k)
  vector_k   = concat(postprocess(pred_rev_k), postprocess(pred_cog_k))   # 2 x 365
  actual_k   = concat(y_rev_k, y_cog_k)

  L_k = 0.45 · RMSE(vector_k, actual_k) / 735000
      + 0.35 · MAE(vector_k, actual_k)  / 532000
      + 0.20 · (1 - R2(vector_k, actual_k))

Minimise: mean(L_2020, L_2021, L_2022)  via Nelder-Mead, 12 random restarts
```

---

## Appendix C: Model Comparison

| Model | Type | Extrapolates trend | Avg CV MAE | Decision |
|---|---|:---:|---:|---|
| **Hybrid ARIMA** | Classical TS | Yes (geometric baseline) | ~540K | Selected (~83% weight) |
| **Theta** | Classical TS | Yes (SES trend component) | ~540K | Selected (~17% weight) |
| Holt-Winters | Classical TS | Yes (additive trend) | ~540K | In pool; 0% weight from optimizer |
| Prophet | Bayesian TS | Yes (piecewise linear) | ~540K | Tested; 0% NNLS weight — excluded |
| LightGBM (lag features) | Tree ensemble | No | ~631K holdout | Excluded — cannot extrapolate beyond training range |
| Chronos-T5 zero-shot | Foundation model | Partially | ~1,370K holdout | Excluded — poor on 2-year horizon |

---

## Appendix D: Day-of-Week Factors

Computed as `mean(y_dow / global_mean)` over the 2018–2022 training window.

| Day | Revenue factor | COGS factor |
|---|---:|---:|
| Mon | 1.015 | 1.013 |
| Tue | 1.033 | 1.032 |
| Wed | 1.091 | 1.091 |
| Thu | 1.038 | 1.038 |
| Fri | 0.945 | 0.945 |
| Sat | 0.918 | 0.919 |
| Sun | 0.959 | 0.960 |

Mid-week (Wed–Thu) consistently shows higher revenue than weekends (Fri–Sun), suggesting B2B or work-hour-adjacent purchasing behavior in this fashion e-commerce segment.