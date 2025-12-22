"""ClearML Pipeline for ML workflow automation."""

import argparse

from clearml import PipelineController
from clearml.automation.controller import PipelineDecorator


def create_ml_pipeline(project_name: str = "customer-churn-prediction"):
    """Create ClearML pipeline for ML workflow.

    Args:
        project_name: ClearML project name
    """
    pipe = PipelineController(
        name="ML Training Pipeline",
        project=project_name,
        version="1.0",
        add_pipeline_tags=True,
    )

    pipe.set_default_execution_queue("default")

    pipe.add_step(
        name="preprocess_data",
        base_task_project=project_name,
        base_task_name="preprocess",
        parameter_override={"data/test_size": 0.2, "data/random_state": 42},
    )

    pipe.add_step(
        name="train_lightgbm",
        parents=["preprocess_data"],
        base_task_project=project_name,
        base_task_name="train_lightgbm",
        parameter_override={"model": "lightgbm", "experiment": "pipeline"},
    )

    pipe.add_step(
        name="train_xgboost",
        parents=["preprocess_data"],
        base_task_project=project_name,
        base_task_name="train_xgboost",
        parameter_override={"model": "xgboost", "experiment": "pipeline"},
    )

    pipe.add_step(
        name="train_catboost",
        parents=["preprocess_data"],
        base_task_project=project_name,
        base_task_name="train_catboost",
        parameter_override={"model": "catboost", "experiment": "pipeline"},
    )

    pipe.add_step(
        name="evaluate_models",
        parents=["train_lightgbm", "train_xgboost", "train_catboost"],
        base_task_project=project_name,
        base_task_name="evaluate",
        parameter_override={},
    )

    pipe.add_step(
        name="select_best",
        parents=["evaluate_models"],
        base_task_project=project_name,
        base_task_name="select_best",
        parameter_override={},
    )

    pipe.add_step(
        name="generate_report",
        parents=["select_best"],
        base_task_project=project_name,
        base_task_name="generate_report",
        parameter_override={},
    )

    return pipe


@PipelineDecorator.component(
    return_values=["processed_data"], cache=True, task_type="data_processing"
)
def preprocess_data_component(test_size: float = 0.2, random_state: int = 42):
    """Preprocess data component.

    Args:
        test_size: Test set size
        random_state: Random seed

    Returns:
        Path to processed data
    """
    import subprocess  # nosec B404

    subprocess.run(
        ["python", "src/data/preprocess.py"], check=False  # nosec B603, B607
    )
    return "data/processed"


@PipelineDecorator.component(
    return_values=["model_metrics"], cache=True, task_type="training"
)
def train_model_component(processed_data: str, model_type: str):
    """Train model component.

    Args:
        processed_data: Path to processed data
        model_type: Model type to train

    Returns:
        Model metrics
    """
    import json
    import subprocess  # nosec B404

    subprocess.run(
        [  # nosec B603, B607
            "python",
            "src/models/train_clearml.py",
            f"model={model_type}",
        ],
        check=False,
    )

    with open("metrics.json") as f:
        metrics = json.load(f)

    return metrics


@PipelineDecorator.component(
    return_values=["best_model"], cache=True, task_type="evaluation"
)
def select_best_model_component(*model_metrics):
    """Select best model component.

    Args:
        *model_metrics: Metrics from all models

    Returns:
        Best model info
    """
    best_model = None
    best_score = -1

    for metrics in model_metrics:
        if metrics.get("roc_auc", 0) > best_score:
            best_score = metrics["roc_auc"]
            best_model = metrics

    return best_model


@PipelineDecorator.pipeline(
    name="ML Pipeline (Decorated)", project="customer-churn-prediction", version="1.0"
)
def run_decorated_pipeline(
    test_size: float = 0.2, random_state: int = 42, models: list[str] | None = None
):
    """Run pipeline using decorators.

    Args:
        test_size: Test set size
        random_state: Random seed
        models: List of models to train
    """
    if models is None:
        models = ["lightgbm", "xgboost", "catboost"]

    processed = preprocess_data_component(test_size, random_state)

    model_results = []
    for model_type in models:
        metrics = train_model_component(processed, model_type)
        model_results.append(metrics)

    best = select_best_model_component(*model_results)

    print(f"Best model: {best}")

    return best


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="ClearML Pipeline")
    parser.add_argument("--project", type=str, default="customer-churn-prediction")
    parser.add_argument(
        "--mode", type=str, choices=["create", "run", "decorated"], default="create"
    )
    parser.add_argument("--queue", type=str, default="default")
    args = parser.parse_args()

    if args.mode == "create":
        pipe = create_ml_pipeline(args.project)

        print("Starting pipeline...")
        pipe.start_locally()

        print("\nPipeline created and running locally")
        print(f"Pipeline ID: {pipe.id}")

    elif args.mode == "run":
        pipe = create_ml_pipeline(args.project)

        print(f"Starting pipeline on queue '{args.queue}'...")
        pipe.start(queue=args.queue)

        print("Pipeline started remotely")
        print(f"Pipeline ID: {pipe.id}")

    elif args.mode == "decorated":
        print("Running decorated pipeline...")

        result = run_decorated_pipeline(
            test_size=0.2, random_state=42, models=["lightgbm", "xgboost", "catboost"]
        )

        print(f"Pipeline completed. Best model: {result}")


if __name__ == "__main__":
    main()
