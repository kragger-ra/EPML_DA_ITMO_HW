"""MLflow utilities for experiment tracking."""

import functools
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlflow
import yaml


def setup_mlflow(
    tracking_uri: str = "./mlruns",
    experiment_name: str = "customer-churn-prediction",
) -> None:
    """
    Setup MLflow tracking.

        Args:
        tracking_uri: URI for MLflow tracking server
        experiment_name: Name of the experiment
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"MLflow experiment: {experiment_name}")


def log_params_from_dict(params: dict[str, Any], prefix: str = "") -> None:
    """
    Log parameters from dictionary to MLflow.

    Args:
        params: Dictionary of parameters
        prefix: Prefix for parameter names
    """
    for key, value in params.items():
        if isinstance(value, dict):
            log_params_from_dict(value, prefix=f"{prefix}{key}.")
        else:
            mlflow.log_param(f"{prefix}{key}", value)


def log_metrics_from_dict(metrics: dict[str, float], step: int | None = None) -> None:
    """
    Log metrics from dictionary to MLflow.

    Args:
        metrics: Dictionary of metrics
        step: Step number for the metrics
    """
    for key, value in metrics.items():
        mlflow.log_metric(key, value, step=step)


def mlflow_run(
    run_name: str | None = None,
    log_params: bool = True,
    log_model: bool = False,
    artifact_path: str = "model",
) -> Callable:
    """
    Decorator for MLflow run context.

    Args:
        run_name: Name of the run
        log_params: Whether to log function parameters
        log_model: Whether to log the model
        artifact_path: Path for model artifact

    Returns:
        Decorated function

    Example:
        @mlflow_run(run_name="lightgbm_experiment")
        def train_model(params):
            model = train(params)
            return model, metrics
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with mlflow.start_run(run_name=run_name):
                if log_params:
                    for key, value in kwargs.items():
                        if not key.startswith("_"):
                            try:
                                mlflow.log_param(key, value)
                            except Exception:  # noqa: S110
                                # Silently skip params that can't be logged
                                pass

                result = func(*args, **kwargs)

                if log_model and result is not None:
                    if isinstance(result, tuple):
                        model = result[0]
                        mlflow.sklearn.log_model(model, artifact_path)
                    else:
                        mlflow.sklearn.log_model(result, artifact_path)

                return result

        return wrapper

    return decorator


def load_experiment_config(config_path: str) -> dict[str, Any]:
    """
    Load experiment configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary

    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file) as f:
        config: dict[str, Any] = yaml.safe_load(f)
        return config


def save_experiment_config(config: dict[str, Any], output_path: str) -> None:
    """
    Save experiment configuration to YAML file.

    Args:
        config: Configuration dictionary
        output_path: Path to save the configuration
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def get_best_run(
    experiment_name: str, metric: str = "roc_auc"
) -> mlflow.entities.Run | None:
    """
    Get the best run from an experiment based on a metric.

    Args:
        experiment_name: Name of the experiment
        metric: Metric to optimize

    Returns:
        Best run object or None
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )

    if runs.empty:
        return None

    run_id = runs.iloc[0]["run_id"]
    return mlflow.get_run(run_id)


def compare_runs(run_ids: list[str], metrics: list[str]) -> dict[str, dict[str, float]]:
    """
    Compare multiple runs based on specified metrics.

    Args:
        run_ids: List of run IDs to compare
        metrics: List of metric names to compare

    Returns:
        Dictionary mapping run IDs to their metrics
    """
    comparison = {}

    for run_id in run_ids:
        run = mlflow.get_run(run_id)
        comparison[run_id] = {
            "run_name": run.info.run_name,
            **{metric: run.data.metrics.get(metric) for metric in metrics},
        }

    return comparison
