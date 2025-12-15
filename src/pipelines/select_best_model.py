"""Select and save the best model based on evaluation metrics."""

import json
import shutil
from pathlib import Path
from typing import Any

import joblib
import mlflow


def select_best_model(metric: str = "roc_auc") -> dict[str, Any]:
    """
    Select the best model based on specified metric.

    Args:
        metric: Metric to use for selection (default: roc_auc)

    Returns:
        Dictionary with best model information
    """
    print("=" * 80)
    print(f"SELECTING BEST MODEL (by {metric})")
    print("=" * 80)

    evaluation_dir = Path("reports/evaluation")
    best_model_path = evaluation_dir / "best_model.json"

    if not best_model_path.exists():
        print(f"Error: {best_model_path} not found!")
        print("Please run evaluate_models.py first.")
        return {}

    with open(best_model_path) as f:
        best_model_info: dict[str, Any] = json.load(f)

    print("\nBest model selected:")
    for key, value in best_model_info.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    model_type = best_model_info["model_type"]
    run_id = best_model_info["run_id"]

    try:
        mlflow.set_tracking_uri("file:./mlruns")

        try:
            model_uri = f"runs:/{run_id}/model"
            model = mlflow.sklearn.load_model(model_uri)
            print(f"\nModel loaded from MLflow: {model_uri}")
        except Exception as e:
            print(f"\nWarning: Could not load from MLflow: {e}")
            model_path = Path("models") / f"{model_type}_model.pkl"
            if model_path.exists():
                model = joblib.load(model_path)
                print(f"Model loaded from file: {model_path}")
            else:
                print(f"Error: Model file not found: {model_path}")
                return best_model_info

        best_model_dir = Path("models/best")
        best_model_dir.mkdir(parents=True, exist_ok=True)

        best_model_file = best_model_dir / "best_model.pkl"
        joblib.dump(model, best_model_file)
        print(f"\nBest model saved to: {best_model_file}")

        metadata: dict[str, Any] = {
            **best_model_info,
            "selection_metric": metric,
            "model_path": str(best_model_file),
        }

        metadata_path = best_model_dir / "model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"Model metadata saved to: {metadata_path}")

        production_model_path = Path("models/churn_model.pkl")
        shutil.copy(best_model_file, production_model_path)
        print(f"\nProduction model updated: {production_model_path}")

        print("\n" + "=" * 80)
        print("BEST MODEL SELECTION COMPLETE")
        print("=" * 80)

        return metadata

    except Exception as e:
        print(f"\nError during model selection: {e}")
        return best_model_info


if __name__ == "__main__":
    select_best_model()
