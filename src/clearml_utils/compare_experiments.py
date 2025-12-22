"""Compare ClearML experiments and generate reports."""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from clearml import Task


class ExperimentComparator:
    """Compare ClearML experiments."""

    def __init__(self, project_name: str):
        """Initialize comparator.

        Args:
            project_name: ClearML project name
        """
        self.project_name = project_name

    def get_all_tasks(self) -> list[Any]:
        """Get all completed tasks from project.

        Returns:
            List of Task objects
        """
        tasks = Task.get_tasks(
            project_name=self.project_name, task_filter={"status": ["completed"]}
        )
        return tasks  # type: ignore

    def extract_task_info(self, task: Any) -> dict[str, Any]:
        """Extract key information from task.

        Args:
            task: ClearML Task

        Returns:
            Dictionary with task information
        """
        scalars = task.get_reported_scalars()

        metrics = {}
        for title, series_dict in scalars.items():
            for series, data in series_dict.items():
                if data and "y" in data:
                    last_value = data["y"][-1]
                    metrics[f"{title}/{series}"] = last_value

        params = task.get_parameters()

        return {
            "task_id": task.id,
            "task_name": task.name,
            "created": task.created,
            "model_type": params.get("model/type", "unknown"),
            "experiment": params.get("experiment/name", "unknown"),
            **metrics,
        }

    def compare_tasks(self, save_path: str | None = None) -> pd.DataFrame:
        """Compare all tasks and create comparison table.

        Args:
            save_path: Path to save comparison CSV

        Returns:
            DataFrame with comparison
        """
        tasks = self.get_all_tasks()

        if not tasks:
            print("No completed tasks found")
            return pd.DataFrame()

        print(f"Found {len(tasks)} completed tasks")

        task_data = [self.extract_task_info(task) for task in tasks]

        df = pd.DataFrame(task_data)

        for sort_col in ["Metrics/roc_auc", "Metrics/f1_score", "Metrics/accuracy"]:
            if sort_col in df.columns:
                df = df.sort_values(sort_col, ascending=False)
                break

        if save_path:
            path_obj = Path(save_path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path_obj, index=False)
            print(f"Comparison saved to {path_obj}")

        return df

    def generate_report(self, df: pd.DataFrame, output_path: str):
        """Generate markdown report from comparison.

        Args:
            df: Comparison DataFrame
            output_path: Path to save markdown report
        """
        if df.empty:
            return

        path_obj = Path(output_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write("# Experiment Comparison Report\n\n")
            f.write(f"**Project**: {self.project_name}\n\n")
            f.write(f"**Total Experiments**: {len(df)}\n\n")

            f.write("## Results\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n\n")

            # Best model
            if "Metrics/roc_auc" in df.columns:
                best = df.iloc[0]
                f.write("## Best Model\n\n")
                f.write(f"- **Task ID**: {best['task_id']}\n")
                f.write(f"- **Model**: {best['model_type']}\n")
                f.write(f"- **ROC-AUC**: {best['Metrics/roc_auc']:.4f}\n")

        print(f"Report saved to {output_path}")

    def find_best_model(self, metric: str = "roc_auc") -> dict[str, Any]:
        """Find best model by metric.

        Args:
            metric: Metric to compare

        Returns:
            Best model info
        """
        df = self.compare_tasks()

        if df.empty:
            return {}

        metric_cols = [col for col in df.columns if metric in col.lower()]

        if not metric_cols:
            print(f"Metric '{metric}' not found")
            return {}

        metric_col = metric_cols[0]
        best = df.sort_values(metric_col, ascending=False).iloc[0]

        best_info = {
            "task_id": best["task_id"],
            "task_name": best["task_name"],
            "model_type": best.get("model_type", "unknown"),
            "model_name": best.get("model_type", "unknown"),
            "metric": metric_col,
            "metric_value": best[metric_col],
        }

        return best_info


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="ClearML Experiment Comparator")
    parser.add_argument("--project", type=str, default="customer-churn-prediction")
    parser.add_argument("--output", type=str, help="Output CSV path")
    parser.add_argument("--report", type=str, help="Output markdown report path")
    parser.add_argument("--metric", type=str, default="roc_auc", help="Metric name")

    args = parser.parse_args()

    comparator = ExperimentComparator(args.project)

    output_path = args.output or "reports/clearml/experiment_comparison.csv"
    df = comparator.compare_tasks(output_path)

    if args.report:
        comparator.generate_report(df, args.report)
    else:
        report_path = "reports/clearml/experiment_comparison.md"
        comparator.generate_report(df, report_path)

    best = comparator.find_best_model(args.metric)
    if best:
        print(f"\n{'=' * 80}")
        print("BEST MODEL")
        print("=" * 80)
        for key, value in best.items():
            print(f"{key}: {value}")
        print("=" * 80)

        best_path = Path("reports/clearml/best_model.json")
        best_path.parent.mkdir(parents=True, exist_ok=True)
        with open(best_path, "w") as f:
            json.dump(best, f, indent=2)
        print(f"Best model info saved to {best_path}")


if __name__ == "__main__":
    main()
