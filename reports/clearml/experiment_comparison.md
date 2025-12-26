# Experiment Comparison Report

**Project**: customer-churn-prediction

**Total Experiments**: 6

## Results

| task_id                          | task_name                 | created                          | model_type        | experiment   |   Metrics/accuracy |   Metrics/f1_score |   Metrics/precision |   Metrics/recall |   Metrics/roc_auc |   learn/Logloss |
|:---------------------------------|:--------------------------|:---------------------------------|:------------------|:-------------|-------------------:|-------------------:|--------------------:|-----------------:|------------------:|----------------:|
| 5818041653f64f63a738b23b0837894b | CatBoost_baseline         | 2025-12-22 11:22:57.587000+00:00 | catboost          | baseline     |           0.806246 |           0.58193  |            0.678571 |         0.509383 |          0.860071 |        0.412665 |
| e539585e514543c1bcc56087d43df841 | XGBoost_baseline          | 2025-12-22 11:22:33.848000+00:00 | xgboost           | baseline     |           0.804116 |           0.577982 |            0.672598 |         0.506702 |          0.852997 |      nan        |
| c5b04909c1924bba93155c0870897167 | GradientBoosting_baseline | 2025-12-22 11:23:17.750000+00:00 | gradient_boosting | baseline     |           0.799858 |           0.575301 |            0.656357 |         0.512064 |          0.852455 |      nan        |
| 2ff82563f31a415a808003c69695c6be | LightGBM_baseline         | 2025-12-22 11:19:41.185000+00:00 | lightgbm          | baseline     |           0.803407 |           0.575804 |            0.671429 |         0.504021 |          0.851562 |      nan        |
| b87f9ea4a8db4330a7abce75ce50af23 | lightgbm_baseline         | 2025-12-26 12:46:36.193000+00:00 | lightgbm          | baseline     |           0.803407 |           0.575804 |            0.671429 |         0.504021 |          0.851562 |      nan        |
| 6024fdf71f3a4dbaac6194bec9663a9d | lightgbm_baseline         | 2025-12-26 13:26:18.882000+00:00 | lightgbm          | baseline     |           0.803407 |           0.575804 |            0.671429 |         0.504021 |          0.851562 |      nan        |

## Best Model

- **Task ID**: 5818041653f64f63a738b23b0837894b
- **Model**: catboost
- **ROC-AUC**: 0.8601
