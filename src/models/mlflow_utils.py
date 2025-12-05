"""Utilities for working with MLflow."""

from pathlib import Path
from typing import Any

import mlflow
import pandas as pd


def get_best_run(experiment_name: str, metric: str = "roc_auc") -> dict[str, Any]:
    """Get the best run from an experiment based on a metric."""
    mlflow.set_tracking_uri("file:./mlruns")
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )

    if runs.empty:
        raise ValueError(f"No runs found in experiment '{experiment_name}'")

    best_run = runs.iloc[0]
    return {
        "run_id": best_run["run_id"],
        "run_name": best_run.get("tags.mlflow.runName", "Unknown"),
        "metrics": {
            col.replace("metrics.", ""): best_run[col]
            for col in runs.columns
            if col.startswith("metrics.")
        },
        "params": {
            col.replace("params.", ""): best_run[col]
            for col in runs.columns
            if col.startswith("params.")
        },
    }


def compare_runs(
    experiment_name: str, metrics: list[str] | None = None
) -> pd.DataFrame:
    """Compare all runs in an experiment."""
    if metrics is None:
        metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]

    mlflow.set_tracking_uri("file:./mlruns")
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    if runs.empty:
        raise ValueError(f"No runs found in experiment '{experiment_name}'")

    result = pd.DataFrame()
    result["run_name"] = runs.get("tags.mlflow.runName", "Unknown")
    result["run_id"] = runs["run_id"]

    for metric in metrics:
        metric_col = f"metrics.{metric}"
        if metric_col in runs.columns:
            result[metric] = runs[metric_col]

    return result.sort_values(by=metrics[0], ascending=False)


def export_best_model(
    experiment_name: str, metric: str = "roc_auc", output_dir: Path | None = None
) -> Path:
    """Export the best model from an experiment."""
    if output_dir is None:
        output_dir = Path("models/mlflow_best")

    output_dir.mkdir(parents=True, exist_ok=True)

    best_run = get_best_run(experiment_name, metric)
    run_id = best_run["run_id"]

    mlflow.set_tracking_uri("file:./mlruns")
    model_uri = f"runs:/{run_id}/model"

    output_path = output_dir / f"best_model_{metric}"
    mlflow.sklearn.save_model(mlflow.sklearn.load_model(model_uri), output_path)

    print(f"Best model saved to: {output_path}")
    print(f"Run: {best_run['run_name']}")
    print(f"Metrics: {best_run['metrics']}")

    return output_path


def print_experiment_summary(experiment_name: str) -> None:
    """Print a summary of all runs in an experiment."""
    mlflow.set_tracking_uri("file:./mlruns")
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    print(f"Experiment: {experiment_name}")
    print(f"Total runs: {len(runs)}")
    print("=" * 80)

    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    comparison = compare_runs(experiment_name, metrics)

    print(comparison.to_string(index=False))
    print("=" * 80)

    for metric in metrics:
        if metric in comparison.columns:
            best_run = comparison.iloc[0]
            print(
                f"Best {metric}: {best_run[metric]:.4f} (Run: {best_run['run_name']})"
            )


if __name__ == "__main__":
    experiment_name = "churn-prediction"

    print("Experiment Summary:")
    print_experiment_summary(experiment_name)

    print("\nExporting best model...")
    export_best_model(experiment_name)
