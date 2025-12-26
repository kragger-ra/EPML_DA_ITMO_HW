"""Train models with ClearML tracking."""

import json
from pathlib import Path

import hydra
import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from clearml import Task
from src.models.model_factory import create_model
from src.utils.mlflow_utils import setup_mlflow


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    """Train model with ClearML tracking.

    Args:
        cfg: Hydra configuration
    """
    model_type = cfg.model.type
    experiment_name = cfg.experiment.name

    task = Task.init(
        project_name=cfg.clearml.project_name,
        task_name=f"{model_type}_{experiment_name}",
        task_type=Task.TaskTypes.training,
        auto_connect_frameworks={
            "scikit": True,
            "xgboost": True,
            "lightgbm": True,
            "catboost": True,
            "matplotlib": True,
        },
    )

    task.connect_configuration(
        configuration=OmegaConf.to_container(cfg, resolve=True), name="hydra_config"
    )

    logger = task.get_logger()

    setup_mlflow(
        tracking_uri=str(cfg.mlflow.tracking_uri), experiment_name=str(cfg.project.name)
    )

    print("Loading data...")
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

    print(f"Training {model_type} model...")
    model = create_model(model_type, cfg.model.params)

    with mlflow.start_run():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else y_pred
        )

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
        }

        print("\nMetrics:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")

            mlflow.log_metric(metric_name, metric_value)

            logger.report_scalar(
                title="Metrics", series=metric_name, value=metric_value, iteration=0
            )

        task.set_parameter("model/type", model_type)
        task.set_parameter("experiment/name", experiment_name)
        task.set_parameter("data/train_size", len(X_train))
        task.set_parameter("data/test_size", len(X_test))

        for key, value in cfg.model.params.items():
            task.set_parameter(f"model/params/{key}", value)
            mlflow.log_param(f"model/{key}", value)

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {model_type}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")

        cm_path = Path(cfg.paths.models) / f"{model_type}_confusion_matrix.png"
        cm_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(cm_path)
        logger.report_matplotlib_figure(
            title="Confusion Matrix",
            series=model_type,
            figure=plt.gcf(),
            iteration=0,
        )
        plt.close()

        if cfg.training.save_model:
            model_path = Path(cfg.paths.models) / f"{model_type}_model.pkl"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_path)
            print(f"Model saved to {model_path}")

            mlflow.sklearn.log_model(model, "model")

            task.upload_artifact(name=f"{model_type}_model", artifact_object=model_path)

        metrics_path = Path(cfg.paths.metrics)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {metrics_path}")

        task.upload_artifact(name="metrics", artifact_object=metrics_path)

    print(f"\nClearML Task ID: {task.id}")
    print("Training completed!")


if __name__ == "__main__":
    train()
