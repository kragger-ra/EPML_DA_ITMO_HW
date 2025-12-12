"""Experiment configurations for different models and hyperparameters."""

EXPERIMENTS = [
    # LightGBM experiments
    {
        "name": "lightgbm_baseline",
        "model_type": "lightgbm",
        "params": {
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 100,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        },
    },
    {
        "name": "lightgbm_high_lr",
        "model_type": "lightgbm",
        "params": {
            "num_leaves": 31,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        },
    },
    {
        "name": "lightgbm_deep",
        "model_type": "lightgbm",
        "params": {
            "num_leaves": 63,
            "learning_rate": 0.05,
            "n_estimators": 150,
            "max_depth": 10,
            "min_child_samples": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        },
    },
    {
        "name": "lightgbm_regularized",
        "model_type": "lightgbm",
        "params": {
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 100,
            "max_depth": -1,
            "min_child_samples": 30,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
        },
    },
    # XGBoost experiments
    {
        "name": "xgboost_baseline",
        "model_type": "xgboost",
        "params": {
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        },
    },
    {
        "name": "xgboost_shallow",
        "model_type": "xgboost",
        "params": {
            "max_depth": 3,
            "learning_rate": 0.1,
            "n_estimators": 150,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        },
    },
    {
        "name": "xgboost_deep",
        "model_type": "xgboost",
        "params": {
            "max_depth": 10,
            "learning_rate": 0.03,
            "n_estimators": 200,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        },
    },
    # CatBoost experiments
    {
        "name": "catboost_baseline",
        "model_type": "catboost",
        "params": {
            "depth": 6,
            "learning_rate": 0.05,
            "iterations": 100,
            "l2_leaf_reg": 3,
            "random_state": 42,
        },
    },
    {
        "name": "catboost_aggressive",
        "model_type": "catboost",
        "params": {
            "depth": 8,
            "learning_rate": 0.1,
            "iterations": 150,
            "l2_leaf_reg": 1,
            "random_state": 42,
        },
    },
    # Random Forest experiments
    {
        "name": "random_forest_baseline",
        "model_type": "random_forest",
        "params": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
        },
    },
    {
        "name": "random_forest_deep",
        "model_type": "random_forest",
        "params": {
            "n_estimators": 200,
            "max_depth": 20,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42,
        },
    },
    {
        "name": "random_forest_shallow",
        "model_type": "random_forest",
        "params": {
            "n_estimators": 100,
            "max_depth": 5,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "random_state": 42,
        },
    },
    # Gradient Boosting experiments
    {
        "name": "gradient_boosting_baseline",
        "model_type": "gradient_boosting",
        "params": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 0.8,
            "random_state": 42,
        },
    },
    {
        "name": "gradient_boosting_conservative",
        "model_type": "gradient_boosting",
        "params": {
            "n_estimators": 150,
            "learning_rate": 0.05,
            "max_depth": 3,
            "subsample": 0.8,
            "random_state": 42,
        },
    },
    # Logistic Regression experiments
    {
        "name": "logistic_regression_l2",
        "model_type": "logistic_regression",
        "params": {
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
            "random_state": 42,
        },
    },
    {
        "name": "logistic_regression_strong_reg",
        "model_type": "logistic_regression",
        "params": {
            "C": 0.1,
            "penalty": "l2",
            "solver": "lbfgs",
            "random_state": 42,
        },
    },
]
