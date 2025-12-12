"""Train models with MLflow tracking for customer churn prediction."""

import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.model_factory import create_model
from src.utils.mlflow_utils import (
    log_metrics_from_dict,
    log_params_from_dict,
    setup_mlflow,
)


def load_params(params_path: str = "params.yaml") -> dict[str, Any]:
    """Load parameters from params.yaml."""
    with open(params_path) as f:
        params: dict[str, Any] = yaml.safe_load(f)
        return params


def load_data(processed_dir: str = "data/processed") -> tuple:
    """Load processed training and test data."""
    processed_path = Path(processed_dir)
    X_train = pd.read_csv(processed_path / "X_train.csv")
    X_test = pd.read_csv(processed_path / "X_test.csv")
    y_train = pd.read_csv(processed_path / "y_train.csv").values.ravel()
    y_test = pd.read_csv(processed_path / "y_test.csv").values.ravel()
    return X_train, X_test, y_train, y_test


def calculate_metrics(y_true, y_pred, y_pred_proba) -> dict[str, float]:
    """Calculate classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_pred_proba)),
    }


def train_model_with_mlflow(
    model_type: str = "lightgbm",
    run_name: str | None = None,
    params_path: str = "params.yaml",
    log_model_artifact: bool = True,
) -> tuple[Any, dict[str, float]]:
    """
    Train model with MLflow tracking.

    Args:
        model_type: Type of model to train (lightgbm, xgboost, random_forest, etc.)
        run_name: Name for the MLflow run
        params_path: Path to parameters file
        log_model_artifact: Whether to log model as MLflow artifact

    Returns:
        Tuple of (model, metrics)
    """
    setup_mlflow()

    params = load_params(params_path)
    X_train, X_test, y_train, y_test = load_data()

    with mlflow.start_run(run_name=run_name or f"{model_type}_experiment"):
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])

        model_params = params.get("model", {}).get("params", {})
        log_params_from_dict(model_params, prefix="model.")

        model = create_model(model_type, model_params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        log_metrics_from_dict(metrics)

        if log_model_artifact:
            mlflow.sklearn.log_model(model, "model")

            models_dir = Path("models")
            models_dir.mkdir(parents=True, exist_ok=True)
            model_path = models_dir / f"{model_type}_model.pkl"
            joblib.dump(model, model_path)
            mlflow.log_artifact(str(model_path))

        metrics_path = Path("metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        mlflow.log_artifact(str(metrics_path))

        print(f"\n{'='*50}")
        print(f"Model Type: {model_type}")
        print(f"Run Name: {run_name or f'{model_type}_experiment'}")
        print(f"{'='*50}")
        print("Metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
        print(f"{'='*50}\n")

        return model, metrics


if __name__ == "__main__":
    import sys

    model_type = sys.argv[1] if len(sys.argv) > 1 else "lightgbm"
    train_model_with_mlflow(model_type=model_type)
