# Forecasting Drivers — What the Model Learned

## 1. Per-Model Contribution (NNLS Weights)

Mỗi model đóng góp bao nhiêu phần trăm vào dự báo cuối cùng.

| Model | Revenue Weight | COGS Weight | Vai trò kinh doanh |
|---|---:|---:|---|
| prophet | 0.086 | 0.062 | Bắt trend dài hạn + seasonality năm |
| linear_last3 | 0.841 | 0.829 | Ngoại suy tăng trưởng tuyến tính từ 3 năm gần nhất |
| theta | 0.055 | 0.060 | M3/M4 method — robust với dữ liệu nhiễu |
| holtwinters | 0.018 | 0.048 | Triple Exponential Smoothing — trend + weekly seasonality |

## 2. Day-of-Week Effect

Strength applied: **0.50** (0=tắt, 1=full).

| DoW | Revenue factor | COGS factor |
|---|---:|---:|
| Thứ 2 | 1.015 | 1.013 |
| Thứ 3 | 1.033 | 1.032 |
| Thứ 4 | 1.091 | 1.091 |
| Thứ 5 | 1.038 | 1.038 |
| Thứ 6 | 0.945 | 0.945 |
| Thứ 7 | 0.918 | 0.919 |
| Chủ Nhật | 0.959 | 0.960 |

## 3. Year-over-Year Growth

Revenue scale (test vs holdout): **×1.200**
COGS scale: **×1.220**

YoY growth from training: Rev 12.15%, COGS 8.42%

## 4. Validation Metrics (2022 Holdout)

- **MAE**:         562,159
- **RMSE**:        755,798
- **R²**:           0.7721

## 5. Business Interpretation

- **Trend tăng trưởng tuyến tính dominant** (84% weight): doanh thu tiếp tục tăng đều mỗi năm. Mô hình tin tưởng vào việc ngoại suy xu hướng quá khứ.
- **Theta 5%** giúp robustness với outlier.
- **Ngày bán cao nhất**: Thứ 4 (×1.09). **Ngày thấp nhất**: Thứ 7 (×0.92).
- **Holdout R² = 0.772** nghĩa là mô hình giải thích được 77.2% phương sai của doanh thu thực tế.