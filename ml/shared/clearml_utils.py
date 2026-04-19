"""
ClearML helpers shared across all training scripts.
Reads credentials from environment / ~/.clearml.conf.
"""

import os
from pathlib import Path
from typing import Any

from clearml import Dataset, Task
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")


def init_task(
    project: str,
    name: str,
    task_type: Task.TaskTypes = Task.TaskTypes.training,
    tags: list[str] | None = None,
    config: dict[str, Any] | None = None,
    reuse_last_task_id: bool = False,
) -> Task:
    """
    Initialise a ClearML task and connect hyperparameters.

    Args:
        project:  ClearML project name, e.g. "MedVision/MRI-Segmentation"
        name:     Task name, e.g. "train-yolo-v1"
        task_type: training | testing | inference | ...
        tags:     Optional list of tags shown in the UI
        config:   Dict of hyperparameters to log (auto-connected)
        reuse_last_task_id: Set True only in interactive/notebook sessions

    Returns:
        Initialised ClearML Task
    """
    task = Task.init(
        project_name=project,
        task_name=name,
        task_type=task_type,
        tags=tags or [],
        reuse_last_task_id=reuse_last_task_id,
        auto_connect_frameworks={
            "pytorch": True,
            "tensorboard": False,
            "matplotlib": True,
        },
    )

    if config:
        task.connect(config, name="Hyperparameters")

    return task


def get_or_create_dataset(
    dataset_name: str,
    project: str,
    local_path: str | Path,
    tags: list[str] | None = None,
) -> Dataset:
    """
    Upload a local dataset folder to ClearML if it doesn't exist yet.
    Returns the Dataset object (use .get_local_copy() to get the local path).
    """
    try:
        dataset = Dataset.get(
            dataset_name=dataset_name,
            dataset_project=project,
            auto_create=False,
        )
        print(f"[ClearML] Dataset '{dataset_name}' already exists — reusing.")
        return dataset
    except Exception:
        pass

    print(f"[ClearML] Creating dataset '{dataset_name}' from {local_path} ...")
    dataset = Dataset.create(
        dataset_name=dataset_name,
        dataset_project=project,
        dataset_tags=tags or [],
    )
    dataset.add_local_files(str(local_path))
    dataset.upload()
    dataset.finalize()
    print(f"[ClearML] Dataset uploaded — ID: {dataset.id}")
    return dataset


def log_metrics(task: Task, metrics: dict[str, float], epoch: int) -> None:
    """Log a dict of scalar metrics for the current epoch."""
    logger = task.get_logger()
    for name, value in metrics.items():
        series, title = (name.split("/", 1) + [name])[:2]
        logger.report_scalar(title=title, series=series, value=value, iteration=epoch)


def upload_model(task: Task, model_path: str | Path, name: str) -> None:
    """Upload a model file as a Task artifact."""
    task.upload_artifact(name=name, artifact_object=str(model_path))
    print(f"[ClearML] Model '{name}' uploaded from {model_path}")
