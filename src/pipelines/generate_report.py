"""Generate comprehensive pipeline execution report."""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def generate_report() -> None:
    """Generate comprehensive report of pipeline execution."""
    print("=" * 80)
    print("GENERATING PIPELINE REPORT")
    print("=" * 80)

    report_lines = []
    report_lines.append("# ML Pipeline Execution Report")
    report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("\n## Pipeline Status\n")

    required_files = {
        "Data (processed)": Path("data/processed/X_train.csv"),
        "Model": Path("models/churn_model.pkl"),
        "Metrics": Path("metrics.json"),
        "Evaluation": Path("reports/evaluation/best_model.json"),
    }

    all_present = True
    for name, path in required_files.items():
        status = "[OK]" if path.exists() else "[MISSING]"
        report_lines.append(f"- {status} {name}: `{path}`")
        if not path.exists():
            all_present = False

    if all_present:
        report_lines.append("\n**Status:** All pipeline stages completed successfully")
    else:
        report_lines.append(
            "\n**Status:** Some pipeline stages are incomplete or failed"
        )

    metrics_path = Path("metrics.json")
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

        report_lines.append("\n## Model Performance\n")
        report_lines.append("| Metric | Value |")
        report_lines.append("|--------|-------|")
        for metric, value in metrics.items():
            report_lines.append(f"| {metric} | {value:.4f} |")

    best_model_path = Path("reports/evaluation/best_model.json")
    if best_model_path.exists():
        with open(best_model_path) as f:
            best_model = json.load(f)

        report_lines.append("\n## Best Model\n")
        report_lines.append(f"- **Model Type:** {best_model.get('model_type', 'N/A')}")
        report_lines.append(f"- **ROC-AUC:** {best_model.get('roc_auc', 0):.4f}")
        report_lines.append(f"- **Accuracy:** {best_model.get('accuracy', 0):.4f}")
        report_lines.append(f"- **F1-Score:** {best_model.get('f1_score', 0):.4f}")
        report_lines.append(f"- **Run ID:** `{best_model.get('run_id', 'N/A')}`")

    comparison_path = Path("reports/evaluation/model_comparison.csv")
    if comparison_path.exists():
        try:
            comparison = pd.read_csv(comparison_path)
            if "model_type" in comparison.columns and "roc_auc" in comparison.columns:
                report_lines.append("\n## Model Comparison (Top 5)\n")
                report_lines.append("| Rank | Model | ROC-AUC | Accuracy | F1-Score |")
                report_lines.append("|------|-------|---------|----------|----------|")

                top_models = (
                    comparison.nsmallest(5, "roc_auc")
                    if "roc_auc" in comparison.columns
                    else comparison.head(5)
                )

                for idx, row in enumerate(top_models.itertuples(), 1):
                    model_type = getattr(row, "model_type", "N/A")
                    roc_auc = getattr(row, "roc_auc", 0)
                    accuracy = getattr(row, "accuracy", 0)
                    f1 = getattr(row, "f1_score", 0)
                    report_lines.append(
                        f"| {idx} | {model_type} | {roc_auc:.4f} | "
                        f"{accuracy:.4f} | {f1:.4f} |"
                    )
        except Exception as e:
            report_lines.append(f"\n*Error loading model comparison: {e}*")

    report_lines.append("\n## Pipeline Configuration\n")
    report_lines.append("- **Orchestration:** DVC Pipelines")
    report_lines.append("- **Configuration Management:** Hydra")
    report_lines.append("- **Experiment Tracking:** MLflow")
    report_lines.append("- **Version Control:** Git + DVC")

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_path = reports_dir / "pipeline_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nReport generated: {report_path}")
    print("\nReport preview:")
    print("=" * 80)
    print("\n".join(report_lines[:20]))
    if len(report_lines) > 20:
        print(f"\n... ({len(report_lines) - 20} more lines)")
    print("=" * 80)


if __name__ == "__main__":
    generate_report()
