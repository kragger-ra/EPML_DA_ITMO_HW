"""Send notifications about pipeline execution results."""

import json
import sys
from datetime import datetime
from pathlib import Path


def send_notification(
    status: str = "success", message: str = "", details: dict | None = None
) -> None:
    """
    Send notification about pipeline execution.

    Args:
        status: Status of the pipeline (success/failure/warning)
        message: Custom message
        details: Additional details to include
    """
    notification = {
        "timestamp": datetime.now().isoformat(),
        "status": status,
        "message": message or f"Pipeline completed with status: {status}",
        "details": details or {},
    }

    notifications_dir = Path("reports/notifications")
    notifications_dir.mkdir(parents=True, exist_ok=True)

    notification_file = (
        notifications_dir
        / f"notification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    with open(notification_file, "w") as f:
        json.dump(notification, f, indent=4)

    status_symbol = {
        "success": "[OK]",
        "failure": "[FAIL]",
        "warning": "[WARN]",
        "info": "[INFO]",
    }

    print("\n" + "=" * 80)
    print(f"{status_symbol.get(status, '[-]')} PIPELINE NOTIFICATION")
    print("=" * 80)
    print(f"Status: {status.upper()}")
    print(f"Time: {notification['timestamp']}")
    print(f"Message: {notification['message']}")

    if details:
        print("\nDetails:")
        for key, value in details.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    print("=" * 80)
    print(f"Notification saved to: {notification_file}")
    print("=" * 80 + "\n")


def check_pipeline_status() -> dict:
    """
    Check pipeline status and generate notification.

    Returns:
        Dictionary with status information
    """
    status_info = {
        "data_processed": Path("data/processed/X_train.csv").exists(),
        "model_trained": Path("models/churn_model.pkl").exists(),
        "metrics_available": Path("metrics.json").exists(),
        "evaluation_done": Path("reports/evaluation/best_model.json").exists(),
    }

    all_ok = all(status_info.values())
    status = "success" if all_ok else "warning"

    details = {}
    if status_info["metrics_available"]:
        try:
            with open("metrics.json") as f:
                metrics = json.load(f)
            details["metrics"] = metrics
        except Exception as e:
            details["metrics_error"] = str(e)

    if status_info["evaluation_done"]:
        try:
            with open("reports/evaluation/best_model.json") as f:
                best_model = json.load(f)
            details["best_model"] = best_model.get("model_type", "unknown")
            details["best_roc_auc"] = best_model.get("roc_auc", 0.0)
        except Exception as e:
            details["best_model_error"] = str(e)

    message = (
        "Pipeline completed successfully!"
        if all_ok
        else f"Pipeline completed with issues: {status_info}"
    )

    send_notification(status=status, message=message, details=details)

    return {"status": status, "details": status_info}


if __name__ == "__main__":
    result = check_pipeline_status()
    sys.exit(0 if result["status"] == "success" else 1)
