"""Quick connection test — run after clearml-init."""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from clearml import Task

print("Connecting to ClearML...")
task = Task.init(
    project_name="MedVision/Test",
    task_name="connection-check",
    reuse_last_task_id=False,
)
task.get_logger().report_text("ClearML connection OK")
task.close()
print("\nConnection OK")
print(f"Task ID: {task.id}")
print("View at: https://app.clear.ml/projects")
