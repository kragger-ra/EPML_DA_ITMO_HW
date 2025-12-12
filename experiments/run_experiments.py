"""Run multiple experiments with different configurations."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from experiments.experiment_configs import EXPERIMENTS  # noqa: E402
from src.models.train_mlflow import train_model_with_mlflow  # noqa: E402
from src.utils.mlflow_utils import setup_mlflow  # noqa: E402


def create_params_file(
    model_params: dict, output_path: str = "params_temp.yaml"
) -> str:
    """Create temporary params file for experiment."""
    params_structure = {
        "model": {"type": model_params["model_type"], "params": model_params["params"]},
        "data": {"test_size": 0.2, "random_state": 42},
        "features": {
            "categorical": [
                "gender",
                "Partner",
                "Dependents",
                "PhoneService",
                "InternetService",
                "Contract",
            ],
            "numerical": ["tenure", "MonthlyCharges", "TotalCharges"],
        },
    }

    with open(output_path, "w") as f:
        yaml.dump(params_structure, f, default_flow_style=False)

    return output_path


def run_all_experiments() -> None:
    """Run all experiments defined in experiment_configs."""
    print("=" * 70)
    print("STARTING EXPERIMENT SUITE")
    print("=" * 70)
    print(f"Total experiments to run: {len(EXPERIMENTS)}\n")

    setup_mlflow()

    results = []

    for i, experiment in enumerate(EXPERIMENTS, 1):
        print(f"\n{'='*70}")
        print(f"Experiment {i}/{len(EXPERIMENTS)}: {experiment['name']}")
        print(f"{'='*70}")

        params_file = create_params_file(experiment)

        try:
            model, metrics = train_model_with_mlflow(
                model_type=experiment["model_type"],
                run_name=experiment["name"],
                params_path=params_file,
                log_model_artifact=True,
            )

            results.append(
                {
                    "experiment_name": experiment["name"],
                    "model_type": experiment["model_type"],
                    **metrics,
                }
            )

            Path(params_file).unlink()

        except Exception as e:
            print(f"ERROR in experiment {experiment['name']}: {str(e)}")
            continue

    print("\n" + "=" * 70)
    print("EXPERIMENT SUITE COMPLETED")
    print("=" * 70)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("roc_auc", ascending=False)

    print("\nResults Summary (sorted by ROC-AUC):")
    print(results_df.to_string(index=False))

    results_df.to_csv("experiments/experiment_results.csv", index=False)
    print("\nResults saved to: experiments/experiment_results.csv")

    print("\n" + "=" * 70)
    print("Best Model:")

    best = results_df.iloc[0]
    print(f"  Name: {best['experiment_name']}")
    print(f"  Model Type: {best['model_type']}")
    print(f"  ROC-AUC: {best['roc_auc']:.4f}")
    print(f"  Accuracy: {best['accuracy']:.4f}")
    print(f"  F1-Score: {best['f1_score']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    run_all_experiments()
