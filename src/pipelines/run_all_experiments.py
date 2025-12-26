"""Run all model experiments with Hydra configurations."""

import subprocess  # nosec B404
import sys
from pathlib import Path

import pandas as pd

MODELS = [
    "lightgbm",
    "xgboost",
    "catboost",
    "random_forest",
    "gradient_boosting",
    "logistic_regression",
]


def run_experiment(model: str) -> dict | None:
    """
    Run a single experiment with specified model.

    Args:
        model: Model type to train

    Returns:
        Dictionary with experiment results or None if failed
    """
    print(f"\n{'=' * 80}")
    print(f"Running experiment: {model}")
    print(f"{'=' * 80}\n")

    try:
        cmd = [
            sys.executable,
            "src/models/train_hydra.py",
            f"model={model}",
        ]

        subprocess.run(  # nosec B603
            cmd, check=True, capture_output=False, text=True, cwd=Path.cwd()
        )

        print(f"\n[OK] Experiment {model} completed successfully")
        return {"model": model, "status": "success"}

    except subprocess.CalledProcessError as e:
        print(f"\n[FAIL] Experiment {model} failed: {e}")
        return {"model": model, "status": "failed", "error": str(e)}
    except Exception as e:
        print(f"\n[ERROR] Unexpected error in {model}: {e}")
        return {"model": model, "status": "error", "error": str(e)}


def main():
    """Run all experiments sequentially."""
    print("\n" + "=" * 80)
    print("RUNNING ALL MODEL EXPERIMENTS")
    print("=" * 80)
    print(f"Total models to train: {len(MODELS)}")
    print(f"Models: {', '.join(MODELS)}")
    print("=" * 80 + "\n")

    results = []

    for i, model in enumerate(MODELS, 1):
        print(f"\n[{i}/{len(MODELS)}] Starting {model}...")
        result = run_experiment(model)
        if result:
            results.append(result)

    print("\n" + "=" * 80)
    print("EXPERIMENT SUITE COMPLETED")
    print("=" * 80)

    successful = sum(1 for r in results if r.get("status") == "success")
    failed = len(results) - successful

    print(f"\nTotal experiments: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if results:
        results_df = pd.DataFrame(results)
        print("\nResults Summary:")
        print(results_df.to_string(index=False))

        results_path = Path("experiments/pipeline_results.csv")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_path, index=False)
        print(f"\nResults saved to: {results_path}")

    print("\n" + "=" * 80)
    print("View detailed results: pixi run mlflow-ui")
    print("=" * 80 + "\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
