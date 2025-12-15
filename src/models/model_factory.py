"""Model factory for creating different ML models."""

from typing import Any

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def create_model(model_type: str, params: dict[str, Any] | None = None) -> Any:
    """
    Create a machine learning model based on type.

    Args:
        model_type: Type of model to create
        params: Model parameters

    Returns:
        Initialized model instance

    Raises:
        ValueError: If model_type is not supported
    """
    if params is None:
        params = {}

    models = {
        "lightgbm": lambda: lgb.LGBMClassifier(**params),
        "xgboost": lambda: xgb.XGBClassifier(**params),
        "catboost": lambda: CatBoostClassifier(**params),
        "random_forest": lambda: RandomForestClassifier(**params),
        "gradient_boosting": lambda: GradientBoostingClassifier(**params),
        "logistic_regression": lambda: LogisticRegression(**params),
        "decision_tree": lambda: DecisionTreeClassifier(**params),
    }

    if model_type not in models:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available types: {', '.join(models.keys())}"
        )

    return models[model_type]()


def get_default_params(model_type: str) -> dict[str, Any]:
    """
    Get default parameters for a model type.

    Args:
        model_type: Type of model

    Returns:
        Dictionary of default parameters
    """
    default_params = {
        "lightgbm": {
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 100,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        },
        "xgboost": {
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        },
        "catboost": {
            "depth": 6,
            "learning_rate": 0.05,
            "iterations": 100,
            "l2_leaf_reg": 3,
            "random_state": 42,
        },
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
        },
        "gradient_boosting": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 0.8,
            "random_state": 42,
        },
        "logistic_regression": {
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
            "random_state": 42,
        },
        "decision_tree": {
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
        },
    }

    result = default_params.get(model_type, {})
    return result  # type: ignore[return-value]
