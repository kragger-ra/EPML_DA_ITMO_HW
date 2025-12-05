"""Train models with MLflow tracking."""

import argparse
from pathlib import Path
from typing import Any, cast

import catboost as cb
import lightgbm as lgb
import mlflow
import mlflow.catboost
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier


def load_params() -> dict[str, Any]:
    """Load parameters from params.yaml."""
    params_path = Path("params.yaml")
    with open(params_path) as f:
        return cast(dict[str, Any], yaml.safe_load(f))


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load processed data."""
    processed_dir = Path("data/processed")
    X_train = pd.read_csv(processed_dir / "X_train.csv")
    X_test = pd.read_csv(processed_dir / "X_test.csv")
    y_train = pd.read_csv(processed_dir / "y_train.csv").values.ravel()
    y_test = pd.read_csv(processed_dir / "y_test.csv").values.ravel()
    return X_train, X_test, y_train, y_test


def evaluate_model(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
    }


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict[str, Any],
) -> None:
    """Train Logistic Regression with MLflow tracking."""
    with mlflow.start_run(run_name="LogisticRegression"):
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        print(f"LogisticRegression - Metrics: {metrics}")


def train_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict[str, Any],
) -> None:
    """Train Decision Tree with MLflow tracking."""
    with mlflow.start_run(run_name="DecisionTree"):
        model = DecisionTreeClassifier(**params)
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        print(f"DecisionTree - Metrics: {metrics}")


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict[str, Any],
) -> None:
    """Train Random Forest with MLflow tracking."""
    with mlflow.start_run(run_name="RandomForest"):
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        print(f"RandomForest - Metrics: {metrics}")


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict[str, Any],
) -> None:
    """Train LightGBM with MLflow tracking."""
    with mlflow.start_run(run_name="LightGBM"):
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.lightgbm.log_model(model, "model")

        print(f"LightGBM - Metrics: {metrics}")


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict[str, Any],
) -> None:
    """Train XGBoost with MLflow tracking."""
    with mlflow.start_run(run_name="XGBoost"):
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, "model")

        print(f"XGBoost - Metrics: {metrics}")


def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict[str, Any],
) -> None:
    """Train CatBoost with MLflow tracking."""
    with mlflow.start_run(run_name="CatBoost"):
        model = cb.CatBoostClassifier(**params, verbose=False)
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.catboost.log_model(model, "model")

        print(f"CatBoost - Metrics: {metrics}")


def main() -> None:
    """Main function to run experiments."""
    parser = argparse.ArgumentParser(description="Train models with MLflow")
    parser.add_argument(
        "--experiment",
        type=str,
        default="churn-prediction",
        help="MLflow experiment name",
    )
    args = parser.parse_args()

    mlflow.set_experiment(args.experiment)
    mlflow.set_tracking_uri("file:./mlruns")

    X_train, X_test, y_train, y_test = load_data()

    print("Starting experiments...")
    print("=" * 80)

    # Logistic Regression experiments
    train_logistic_regression(
        X_train, y_train, X_test, y_test, {"max_iter": 1000, "random_state": 42}
    )
    train_logistic_regression(
        X_train,
        y_train,
        X_test,
        y_test,
        {"max_iter": 1000, "C": 0.1, "random_state": 42},
    )
    train_logistic_regression(
        X_train,
        y_train,
        X_test,
        y_test,
        {"max_iter": 1000, "C": 10.0, "random_state": 42},
    )

    # Decision Tree experiments
    train_decision_tree(
        X_train, y_train, X_test, y_test, {"max_depth": 5, "random_state": 42}
    )
    train_decision_tree(
        X_train, y_train, X_test, y_test, {"max_depth": 10, "random_state": 42}
    )
    train_decision_tree(
        X_train, y_train, X_test, y_test, {"max_depth": 20, "random_state": 42}
    )

    # Random Forest experiments
    train_random_forest(
        X_train,
        y_train,
        X_test,
        y_test,
        {"n_estimators": 50, "max_depth": 10, "random_state": 42},
    )
    train_random_forest(
        X_train,
        y_train,
        X_test,
        y_test,
        {"n_estimators": 100, "max_depth": 15, "random_state": 42},
    )
    train_random_forest(
        X_train,
        y_train,
        X_test,
        y_test,
        {"n_estimators": 200, "max_depth": 20, "random_state": 42},
    )

    # LightGBM experiments
    train_lightgbm(
        X_train,
        y_train,
        X_test,
        y_test,
        {
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 100,
            "random_state": 42,
        },
    )
    train_lightgbm(
        X_train,
        y_train,
        X_test,
        y_test,
        {
            "num_leaves": 50,
            "learning_rate": 0.1,
            "n_estimators": 150,
            "random_state": 42,
        },
    )
    train_lightgbm(
        X_train,
        y_train,
        X_test,
        y_test,
        {
            "num_leaves": 20,
            "learning_rate": 0.01,
            "n_estimators": 200,
            "random_state": 42,
        },
    )

    # XGBoost experiments
    train_xgboost(
        X_train,
        y_train,
        X_test,
        y_test,
        {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 100, "random_state": 42},
    )
    train_xgboost(
        X_train,
        y_train,
        X_test,
        y_test,
        {
            "max_depth": 8,
            "learning_rate": 0.05,
            "n_estimators": 150,
            "random_state": 42,
        },
    )
    train_xgboost(
        X_train,
        y_train,
        X_test,
        y_test,
        {"max_depth": 4, "learning_rate": 0.2, "n_estimators": 50, "random_state": 42},
    )

    # CatBoost experiments
    train_catboost(
        X_train,
        y_train,
        X_test,
        y_test,
        {"depth": 6, "learning_rate": 0.1, "iterations": 100, "random_state": 42},
    )
    train_catboost(
        X_train,
        y_train,
        X_test,
        y_test,
        {"depth": 8, "learning_rate": 0.05, "iterations": 150, "random_state": 42},
    )
    train_catboost(
        X_train,
        y_train,
        X_test,
        y_test,
        {"depth": 4, "learning_rate": 0.2, "iterations": 50, "random_state": 42},
    )

    print("=" * 80)
    print("All experiments completed!")
    print("View results: mlflow ui --backend-store-uri file:./mlruns")


if __name__ == "__main__":
    main()
