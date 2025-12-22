"""Automatic launch and monitoring of ClearML pipelines."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from clearml import Task
from clearml.automation import TaskScheduler, TriggerScheduler


class PipelineScheduler:
    """Scheduler for automatic pipeline launch."""

    def __init__(self, project_name: str):
        """Initialize scheduler.

        Args:
            project_name: ClearML project name
        """
        self.project_name = project_name

    def schedule_daily(self, pipeline_task_id: str, hour: int = 2, minute: int = 0):
        """Schedule daily pipeline execution.

        Args:
            pipeline_task_id: Pipeline task ID
            hour: Hour to run (0-23)
            minute: Minute to run (0-59)
        """
        scheduler = TaskScheduler()

        scheduler.add_task(
            schedule_task_id=pipeline_task_id,
            schedule_function=lambda: True,
            hour=hour,
            minute=minute,
        )

        print(f"Pipeline {pipeline_task_id} scheduled daily at {hour:02d}:{minute:02d}")

    def schedule_weekly(
        self,
        pipeline_task_id: str,
        day_of_week: int = 0,
        hour: int = 2,
        minute: int = 0,
    ):
        """Schedule weekly pipeline execution.

        Args:
            pipeline_task_id: Pipeline task ID
            day_of_week: Day of week (0=Monday, 6=Sunday)
            hour: Hour to run
            minute: Minute to run
        """
        scheduler = TaskScheduler()

        scheduler.add_task(
            schedule_task_id=pipeline_task_id,
            schedule_function=lambda: datetime.now().weekday() == day_of_week,
            hour=hour,
            minute=minute,
        )

        days = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        print(
            f"Pipeline {pipeline_task_id} scheduled weekly on {days[day_of_week]} at {hour:02d}:{minute:02d}"
        )

    def create_trigger(self, source_task_id: str, pipeline_task_id: str):
        """Create trigger to run pipeline when source task completes.

        Args:
            source_task_id: Task ID that triggers the pipeline
            pipeline_task_id: Pipeline task ID to run
        """
        scheduler = TriggerScheduler()

        def trigger_callback(task_id: str):
            """Callback to run when source task completes."""
            pipeline_task = Task.get_task(task_id=pipeline_task_id)
            cloned = Task.clone(source_task=pipeline_task)
            Task.enqueue(cloned, queue_name="default")
            print(f"Triggered pipeline {cloned.id} from task {task_id}")

        scheduler.add_task(
            task_id=source_task_id,
            trigger_on_status=["completed"],
            trigger_callback=trigger_callback,
        )

        print(f"Trigger created: {source_task_id} -> {pipeline_task_id}")

    def monitor_pipeline(self, pipeline_task_id: str, interval: int = 60):
        """Monitor pipeline execution.

        Args:
            pipeline_task_id: Pipeline task ID
            interval: Check interval in seconds
        """
        task = Task.get_task(task_id=pipeline_task_id)

        print(f"Monitoring pipeline {pipeline_task_id}...")

        start_time = time.time()
        status_log = []

        while True:
            task.reload()
            status = task.status

            elapsed = time.time() - start_time
            status_log.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "status": status,
                    "elapsed_seconds": elapsed,
                }
            )

            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"Status: {status} (elapsed: {elapsed:.0f}s)"
            )

            if status in ["completed", "failed", "stopped"]:
                break

            time.sleep(interval)

        log_dir = Path("reports/clearml/monitoring")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"pipeline_{pipeline_task_id}.json"

        with open(log_path, "w") as f:
            json.dump(
                {
                    "pipeline_id": pipeline_task_id,
                    "final_status": status,
                    "total_elapsed": time.time() - start_time,
                    "status_log": status_log,
                },
                f,
                indent=2,
            )

        print(f"\nMonitoring completed. Status: {status}")
        print(f"Log saved to {log_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="ClearML Pipeline Scheduler")
    parser.add_argument("--project", type=str, default="customer-churn-prediction")
    parser.add_argument(
        "--action",
        type=str,
        choices=["schedule", "trigger", "monitor"],
        required=True,
    )
    parser.add_argument("--pipeline-id", type=str, help="Pipeline task ID")
    parser.add_argument("--source-id", type=str, help="Source task ID for trigger")
    parser.add_argument("--schedule", type=str, choices=["daily", "weekly"])
    parser.add_argument("--hour", type=int, default=2, help="Hour to run (0-23)")
    parser.add_argument("--minute", type=int, default=0, help="Minute to run (0-59)")
    parser.add_argument(
        "--day", type=int, default=0, help="Day of week for weekly (0-6)"
    )
    parser.add_argument(
        "--interval", type=int, default=60, help="Monitor interval in seconds"
    )

    args = parser.parse_args()

    scheduler = PipelineScheduler(args.project)

    if args.action == "schedule":
        if not args.pipeline_id or not args.schedule:
            print("Error: --pipeline-id and --schedule required for scheduling")
            return

        if args.schedule == "daily":
            scheduler.schedule_daily(args.pipeline_id, args.hour, args.minute)
        elif args.schedule == "weekly":
            scheduler.schedule_weekly(
                args.pipeline_id, args.day, args.hour, args.minute
            )

    elif args.action == "trigger":
        if not args.source_id or not args.pipeline_id:
            print("Error: --source-id and --pipeline-id required for trigger")
            return

        scheduler.create_trigger(args.source_id, args.pipeline_id)

    elif args.action == "monitor":
        if not args.pipeline_id:
            print("Error: --pipeline-id required for monitoring")
            return

        scheduler.monitor_pipeline(args.pipeline_id, args.interval)


if __name__ == "__main__":
    main()
