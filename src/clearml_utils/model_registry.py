"""ClearML Model Registry management."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from clearml import InputModel, Model, Task


class ModelRegistry:
    """Manage models in ClearML."""

    def __init__(self, project_name: str):
        """Initialize model registry.

        Args:
            project_name: ClearML project name
        """
        self.project_name = project_name

    def register_model(
        self,
        task_id: str,
        model_name: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Register model from task.

        Args:
            task_id: ClearML task ID
            model_name: Model name
            tags: Model tags
            metadata: Additional metadata

        Returns:
            Model object
        """
        task = Task.get_task(task_id=task_id)

        model = Model(
            name=model_name,
            project=self.project_name,
            tags=tags or [],
            comment=f"Registered from task {task.name}",
        )

        task_models = task.models.get("output", [])
        if task_models:
            task_model = task_models[0]
            model.update_weights(
                weights_filename=task_model.url, auto_delete_file=False
            )
        else:
            print(f"Warning: No output model found for task {task_id}")

        metrics = self._extract_metrics(task)
        model.set_metadata("metrics", metrics)
        model.set_metadata("source_task_id", task_id)
        model.set_metadata("registered_at", datetime.now().isoformat())

        if metadata:
            for key, value in metadata.items():
                model.set_metadata(key, value)

        print(f"Model '{model_name}' registered successfully")
        print(f"Model ID: {model.id}")
        print(f"Metrics: {metrics}")
        print(f"Source task: {task_id}")

        return model

    def _extract_metrics(self, task: Any) -> dict[str, Any]:
        """Extract metrics from task.

        Args:
            task: ClearML task

        Returns:
            Dictionary of metrics
        """
        metrics = {}
        scalars = task.get_reported_scalars()

        for title, series_dict in scalars.items():
            for series, data in series_dict.items():
                if data and "y" in data:
                    last_value = data["y"][-1]
                    metrics[f"{title}/{series}"] = last_value

        return metrics

    def get_model(
        self, model_id: str | None = None, model_name: str | None = None
    ) -> Any:
        """Get model by ID or name.

        Args:
            model_id: Model ID
            model_name: Model name

        Returns:
            Model object or None
        """
        if model_id:
            return InputModel(model_id=model_id)
        elif model_name:
            models = Model.query_models(
                project_name=self.project_name, model_name=model_name
            )
            if models:
                return models[0]
        return None

    def list_models(self, tags: list[str] | None = None) -> list[Any]:
        """List all registered models.

        Args:
            tags: Filter by tags

        Returns:
            List of Model objects
        """
        models = Model.query_models(project_name=self.project_name, tags=tags)
        return models

    def compare_models(
        self, model_ids: list[str] | None = None, save_path: str | None = None
    ) -> dict[str, Any]:
        """Compare multiple models.

        Args:
            model_ids: List of model IDs to compare
            save_path: Path to save comparison

        Returns:
            Comparison dictionary
        """
        if not model_ids:
            models = self.list_models()
            model_ids = [m.id for m in models]

        comparison: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "models": [],
        }

        for model_id in model_ids:
            model = InputModel(model_id=model_id)
            metadata = model.get_metadata()

            model_info = {
                "model_id": model.id,
                "model_name": model.name,
                "project": model.project,
                "tags": model.tags,
                "created": model.created,
                "metadata": metadata,
            }

            comparison["models"].append(model_info)

        if comparison["models"]:
            for metric_key in [
                "metrics/Metrics/roc_auc",
                "metrics/roc_auc",
                "metrics/Metrics/f1_score",
            ]:
                models_list = comparison["models"]
                if any(metric_key in str(m.get("metadata", {})) for m in models_list):
                    models_list.sort(
                        key=lambda x: float(
                            str(x.get("metadata", {})).get(metric_key, 0)
                        ),
                        reverse=True,
                    )
                    comparison["sorted_by"] = metric_key
                    break

        if save_path:
            path_obj = Path(save_path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            with open(path_obj, "w") as f:
                json.dump(comparison, f, indent=2, default=str)
            print(f"Comparison saved to {path_obj}")

        return comparison

    def get_best_model(self, metric: str = "roc_auc") -> Any:
        """Get best model by metric.

        Args:
            metric: Metric name

        Returns:
            Best Model object or None
        """
        models = self.list_models()

        best_model = None
        best_value = -1

        for model in models:
            metadata = model.get_metadata()
            metrics = metadata.get("metrics", {})

            metric_value = None
            for key, value in metrics.items():
                if metric in key.lower():
                    metric_value = value
                    break

            if metric_value is not None and metric_value > best_value:
                best_value = metric_value
                best_model = model

        if best_model:
            print(f"Best model by {metric}: {best_model.name} ({best_value:.4f})")

        return best_model

    def create_model_version(
        self,
        base_model_id: str,
        task_id: str,
        version: str,
        comment: str | None = None,
    ) -> Any:
        """Create new version of existing model.

        Args:
            base_model_id: Base model ID
            task_id: New task ID
            version: Version string
            comment: Version comment

        Returns:
            New Model object
        """
        base_model = InputModel(model_id=base_model_id)
        task = Task.get_task(task_id=task_id)

        new_model = Model(
            name=f"{base_model.name}_v{version}",
            project=self.project_name,
            tags=base_model.tags + [f"version:{version}"],
            comment=comment or f"Version {version} from task {task.name}",
        )

        task_models = task.models.get("output", [])
        if task_models:
            task_model = task_models[0]
            new_model.update_weights(
                weights_filename=task_model.url, auto_delete_file=False
            )

        metadata = base_model.get_metadata()
        metadata["version"] = version
        metadata["base_model_id"] = base_model_id
        metadata["created_at"] = datetime.now().isoformat()

        new_metrics = self._extract_metrics(task)
        metadata["metrics"] = new_metrics

        for key, value in metadata.items():
            new_model.set_metadata(key, value)

        print(f"Created model version {version}")
        print(f"Model ID: {new_model.id}")

        return new_model


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="ClearML Model Registry")
    parser.add_argument("--project", type=str, default="customer-churn-prediction")
    parser.add_argument(
        "--action",
        type=str,
        choices=["register", "list", "compare", "best"],
        required=True,
    )
    parser.add_argument("--task-id", type=str, help="Task ID for registration")
    parser.add_argument("--model-name", type=str, help="Model name")
    parser.add_argument("--model-ids", nargs="+", help="Model IDs for comparison")
    parser.add_argument("--metric", type=str, default="roc_auc", help="Metric name")
    parser.add_argument("--output", type=str, help="Output path for comparison")

    args = parser.parse_args()

    registry = ModelRegistry(args.project)

    if args.action == "register":
        if not args.task_id or not args.model_name:
            print("Error: --task-id and --model-name required for registration")
            return

        registry.register_model(args.task_id, args.model_name)

    elif args.action == "list":
        models = registry.list_models()
        print(f"\nFound {len(models)} models:")
        for model in models:
            metadata = model.get_metadata()
            metrics = metadata.get("metrics", {})
            print(f"\n  ID: {model.id}")
            print(f"  Name: {model.name}")
            print(f"  Tags: {model.tags}")
            print(f"  Metrics: {metrics}")

    elif args.action == "compare":
        output_path = args.output or "reports/clearml/model_comparison.json"
        comparison = registry.compare_models(args.model_ids, output_path)
        print(f"\nCompared {len(comparison['models'])} models")

    elif args.action == "best":
        best = registry.get_best_model(args.metric)
        if best:
            print(f"\nBest model: {best.name}")
            print(f"ID: {best.id}")


if __name__ == "__main__":
    main()
