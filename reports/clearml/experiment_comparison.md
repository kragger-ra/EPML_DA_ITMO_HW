# Experiment Comparison Report

**Project**: customer-churn-prediction
**Total Experiments**: 4

## Best Model

- **Model**: CatBoost
- **Type**: catboost
- **Metrics/f1_score**: 0.5819
- **Task ID**: 5818041653f64f63a738b23b0837894b

## All Experiments

| task_name                 | model_name       | model_type        |   Metrics/accuracy |   Metrics/f1_score |   Metrics/precision |   Metrics/recall |   Metrics/roc_auc | learn/Logloss   |
|:--------------------------|:-----------------|:------------------|-------------------:|-------------------:|--------------------:|-----------------:|------------------:|:----------------|
| CatBoost_baseline         | CatBoost         | catboost          |             0.8062 |             0.5819 |              0.6786 |           0.5094 |            0.8601 | 0.4127          |
| XGBoost_baseline          | XGBoost          | xgboost           |             0.8041 |             0.578  |              0.6726 |           0.5067 |            0.853  | N/A             |
| GradientBoosting_baseline | GradientBoosting | gradient_boosting |             0.7999 |             0.5753 |              0.6564 |           0.5121 |            0.8525 | N/A             |
| LightGBM_baseline         | LightGBM         | lightgbm          |             0.8034 |             0.5758 |              0.6714 |           0.504  |            0.8516 | N/A             |

## Performance by Model Type

### catboost

- **Count**: 1
- **Avg Metrics/f1_score**: 0.5819
- **Max Metrics/f1_score**: 0.5819
- **Avg Metrics/roc_auc**: 0.8601
- **Max Metrics/roc_auc**: 0.8601

### xgboost

- **Count**: 1
- **Avg Metrics/f1_score**: 0.5780
- **Max Metrics/f1_score**: 0.5780
- **Avg Metrics/roc_auc**: 0.8530
- **Max Metrics/roc_auc**: 0.8530

### gradient_boosting

- **Count**: 1
- **Avg Metrics/f1_score**: 0.5753
- **Max Metrics/f1_score**: 0.5753
- **Avg Metrics/roc_auc**: 0.8525
- **Max Metrics/roc_auc**: 0.8525

### lightgbm

- **Count**: 1
- **Avg Metrics/f1_score**: 0.5758
- **Max Metrics/f1_score**: 0.5758
- **Avg Metrics/roc_auc**: 0.8516
- **Max Metrics/roc_auc**: 0.8516
