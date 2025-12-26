"""Train models with Hydra configuration management."""

import json
from pathlib import Path
from typing import Any

import hydra
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.model_factory import create_model


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
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_pred_proba)),
    }


def setup_mlflow(cfg: DictConfig) -> None:
    """Setup MLflow tracking."""
    if cfg.mlflow.enabled:
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.mlflow.experiment_name)
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        print(f"MLflow experiment: {cfg.mlflow.experiment_name}")


def log_params_recursive(params: dict[str, Any], prefix: str = "") -> None:
    """Recursively log parameters to MLflow."""
    for key, value in params.items():
        if isinstance(value, dict):
            log_params_recursive(value, prefix=f"{prefix}{key}.")
        else:
            mlflow.log_param(f"{prefix}{key}", value)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def train(cfg: DictConfig) -> dict[str, float]:
    """
    Train model with Hydra configuration.

    Args:
        cfg: Hydra configuration object

    Returns:
        Dictionary of metrics
    """
    print("=" * 80)
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    setup_mlflow(cfg)

    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    model_type = cfg.model.type
    model_params = OmegaConf.to_container(cfg.model.params, resolve=True)

    run_name = f"{cfg.model.name}_{cfg.experiment.name}"

    if cfg.mlflow.enabled:
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("model_name", cfg.model.name)
            mlflow.log_param("experiment", cfg.experiment.name)
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])

            log_params_recursive(model_params, prefix="model.")

            if "tags" in cfg.experiment:
                for tag in cfg.experiment.tags:
                    mlflow.set_tag(tag, "true")

            print(f"\nTraining {cfg.model.name}...")
            model = create_model(model_type, model_params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            metrics = calculate_metrics(y_test, y_pred, y_pred_proba)

            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            if cfg.training.save_model:
                models_dir = Path(cfg.paths.models)
                models_dir.mkdir(parents=True, exist_ok=True)
                model_path = models_dir / f"{model_type}_model.pkl"
                joblib.dump(model, model_path)
                print(f"Model saved to: {model_path}")

                if cfg.mlflow.log_models:
                    mlflow.sklearn.log_model(model, "model")
                if cfg.mlflow.log_artifacts:
                    mlflow.log_artifact(str(model_path))

            metrics_path = Path(cfg.paths.metrics)
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)

            if cfg.mlflow.log_artifacts:
                mlflow.log_artifact(str(metrics_path))

            if cfg.training.save_predictions:
                predictions_dir = Path(cfg.paths.reports) / "predictions"
                predictions_dir.mkdir(parents=True, exist_ok=True)
                pred_path = predictions_dir / f"{model_type}_predictions.csv"
                pd.DataFrame(
                    {
                        "y_true": y_test,
                        "y_pred": y_pred,
                        "y_pred_proba": y_pred_proba,
                    }
                ).to_csv(pred_path, index=False)

                if cfg.mlflow.log_artifacts:
                    mlflow.log_artifact(str(pred_path))

            print("\n" + "=" * 80)
            print(f"Model: {cfg.model.name}")
            print(f"Experiment: {cfg.experiment.name}")
            print("=" * 80)
            print("Metrics:")
            for metric_name, metric_value in metrics.items():
                print(f"  {metric_name}: {metric_value:.4f}")
            print("=" * 80)

            return metrics
    else:
        print(f"\nTraining {cfg.model.name} (MLflow disabled)...")
        model = create_model(model_type, model_params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)

        if cfg.training.save_model:
            models_dir = Path(cfg.paths.models)
            models_dir.mkdir(parents=True, exist_ok=True)
            model_path = models_dir / f"{model_type}_model.pkl"
            joblib.dump(model, model_path)
            print(f"Model saved to: {model_path}")

        metrics_path = Path(cfg.paths.metrics)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        print("\n" + "=" * 80)
        print(f"Model: {cfg.model.name}")
        print("Metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
        print("=" * 80)

        return metrics


if __name__ == "__main__":
    train()
