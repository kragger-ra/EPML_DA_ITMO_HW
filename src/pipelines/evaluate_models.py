"""Evaluate all trained models and compare results."""

import json
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd


def load_metrics_from_mlflow(experiment_name: str) -> pd.DataFrame:
    """
    Load metrics from MLflow experiment.

    Args:
        experiment_name: Name of the MLflow experiment

    Returns:
        DataFrame with metrics from all runs
    """
    mlflow.set_tracking_uri("file:./mlruns")
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        print(f"Warning: Experiment '{experiment_name}' not found in MLflow")
        return pd.DataFrame()

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.roc_auc DESC"],
    )

    if runs.empty:
        print(f"Warning: No runs found in experiment '{experiment_name}'")
        return pd.DataFrame()

    return runs


def evaluate_all_models() -> dict[str, Any]:
    """
    Evaluate all trained models and generate comparison report.

    Returns:
        Dictionary with evaluation results
    """
    print("=" * 80)
    print("EVALUATING ALL MODELS")
    print("=" * 80)

    runs_df = load_metrics_from_mlflow("customer-churn-prediction")

    if runs_df.empty:
        print("No runs found in MLflow. Using fallback metrics...")
        metrics_path = Path("metrics.json")
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            print(f"\nLoaded metrics from {metrics_path}:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")
            return metrics  # type: ignore[no-any-return]
        else:
            print("Error: No metrics found!")
            return {}

    metric_columns = [col for col in runs_df.columns if col.startswith("metrics.")]
    param_columns = [col for col in runs_df.columns if col.startswith("params.")]

    results = runs_df[
        ["run_id", "start_time", "status"] + param_columns + metric_columns
    ].copy()

    results.columns = [
        col.replace("metrics.", "").replace("params.", "") for col in results.columns
    ]

    if "roc_auc" in results.columns:
        results = results.sort_values("roc_auc", ascending=False)

    print(f"\nTotal runs: {len(results)}")
    print("\nTop 10 models by ROC-AUC:")
    print(
        results[["model_type", "roc_auc", "accuracy", "f1_score"]]
        .head(10)
        .to_string(index=False)
    )

    evaluation_dir = Path("reports/evaluation")
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    results_path = evaluation_dir / "model_comparison.csv"
    results.to_csv(results_path, index=False)
    print(f"\nFull results saved to: {results_path}")

    if "model_type" in results.columns:
        summary = (
            results.groupby("model_type")[
                ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
            ]
            .agg(["mean", "std", "min", "max"])
            .round(4)
        )

        summary_path = evaluation_dir / "model_summary.csv"
        summary.to_csv(summary_path)
        print(f"Summary statistics saved to: {summary_path}")

        print("\nModel Summary (mean ± std):")
        for model in summary.index:
            roc_auc_mean = summary.loc[model, ("roc_auc", "mean")]
            roc_auc_std = summary.loc[model, ("roc_auc", "std")]
            print(f"  {model}: {roc_auc_mean:.4f} ± {roc_auc_std:.4f}")

    if not results.empty and "roc_auc" in results.columns:
        best_model = results.iloc[0]
        best_model_info = {
            "model_type": best_model.get("model_type", "unknown"),
            "run_id": best_model.get("run_id", "unknown"),
            "roc_auc": float(best_model.get("roc_auc", 0.0)),
            "accuracy": float(best_model.get("accuracy", 0.0)),
            "f1_score": float(best_model.get("f1_score", 0.0)),
        }

        print("\n" + "=" * 80)
        print("BEST MODEL:")
        print("=" * 80)
        for key, value in best_model_info.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        print("=" * 80)

        best_model_path = evaluation_dir / "best_model.json"
        with open(best_model_path, "w") as f:
            json.dump(best_model_info, f, indent=4)
        print(f"\nBest model info saved to: {best_model_path}")

        return best_model_info

    return {}


if __name__ == "__main__":
    evaluate_all_models()
