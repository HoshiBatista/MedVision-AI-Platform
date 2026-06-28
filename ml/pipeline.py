"""
ClearML PipelineController — full train → eval → export → deploy flow.

Orchestrates the existing per-task scripts (no logic is duplicated here):

    ml/<task>/train.py        → trains YOLOv11, writes best.pt
    ml/<task>/evaluate.py     → validates on the test split, writes eval_results.json
    ml/<task>/export_onnx.py  → exports best.pt → ONNX, copies into triton_models/
    (deploy)                  → quality gate on mAP50, confirms the ONNX is in place

Each stage is a ClearML pipeline step (its own task); artifacts flow between
steps via return values, referenced with ``${<step>.<return>}``. Project, run
directory and ClearML names are read from each task's ``config.yaml`` so this
stays in sync with the standalone scripts.

Usage:
    # Run the whole DAG locally (every step on this machine — no agent needed):
    python ml/pipeline.py --task mri_segmentation

    # Tune the gate / hyperparams:
    python ml/pipeline.py --task pneumonia_detection --epochs 50 --min-map 0.5

    # Enqueue on a ClearML agent queue instead of running locally:
    python ml/pipeline.py --task skin_classification --remote --queue default

Note: each stage runs the task's own script, so the training device (mps/cuda/cpu)
is governed by ``ml/<task>/train.py`` / ``evaluate.py``, not by this controller.
"""

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent  # project root
sys.path.insert(0, str(ROOT / "ml"))

TASKS = ("mri_segmentation", "pneumonia_detection", "skin_classification")


# ── Pipeline steps ────────────────────────────────────────────────────────────
# Each step is a standalone function (ClearML pickles it and runs it as its own
# task), so every step re-imports what it needs and takes only plain arguments.


def step_train(
    repo_root: str,
    task: str,
    epochs: int,
    batch: int,
    model: str,
    nested_clearml: bool,
) -> str:
    """Train via ml/<task>/train.py; return the resolved best.pt path."""
    import subprocess

    import yaml as _yaml

    root = Path(repo_root)
    cfg = _yaml.safe_load((root / "ml" / task / "config.yaml").read_text())

    cmd = [
        sys.executable,
        str(root / "ml" / task / "train.py"),
        "--epochs", str(epochs),
        "--batch", str(batch),
    ]
    if model:
        cmd += ["--model", model]
    if not nested_clearml:
        # The pipeline step is itself a ClearML task capturing this run, so by
        # default we don't have train.py spin up a second nested Task.
        cmd.append("--no-clearml")

    subprocess.run(cmd, check=True, cwd=str(root))

    # Ultralytics auto-increments run dirs (train, train2, ...); pick the newest.
    runs = sorted(
        (root / cfg["project"]).glob(f"{cfg['name']}*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        raise FileNotFoundError(f"no best.pt under {root / cfg['project']}")
    return str(runs[0])


def step_evaluate(repo_root: str, task: str, weights: str) -> dict:
    """Validate via ml/<task>/evaluate.py; return the parsed metrics dict."""
    import json
    import subprocess

    root = Path(repo_root)
    subprocess.run(
        [
            sys.executable,
            str(root / "ml" / task / "evaluate.py"),
            "--weights", weights,
            "--split", "test",
            "--save-json",
        ],
        check=True,
        cwd=str(root),
    )
    # evaluate.py --save-json writes eval_results.json next to the weights.
    results = Path(weights).parent / "eval_results.json"
    metrics: dict = json.loads(results.read_text())
    return metrics


def step_export(repo_root: str, task: str, weights: str) -> str:
    """Export to ONNX + copy into triton_models/; return the deployed ONNX path."""
    import subprocess

    root = Path(repo_root)
    subprocess.run(
        [
            sys.executable,
            str(root / "ml" / task / "export_onnx.py"),
            "--weights", weights,
        ],
        check=True,
        cwd=str(root),
    )
    # export_onnx.py copies the graph into triton_models/<task>/1/model.onnx.
    dest = root / "triton_models" / task / "1" / "model.onnx"
    if not dest.exists():
        raise FileNotFoundError(f"export did not produce {dest}")
    return str(dest)


def step_deploy(
    repo_root: str,
    task: str,
    metrics: dict,
    onnx_path: str,
    min_map: float,
) -> dict:
    """Quality gate: deploy only if mAP50 ≥ min_map and the ONNX is in place."""
    map50 = float(metrics.get("map50", 0.0))
    if map50 < float(min_map):
        raise RuntimeError(
            f"quality gate failed for {task}: mAP50 {map50:.4f} < {min_map}"
        )
    if not Path(onnx_path).exists():
        raise FileNotFoundError(f"served ONNX missing: {onnx_path}")

    print(f"[deploy] {task}: mAP50={map50:.4f} ≥ {min_map} → live at {onnx_path}")
    return {"deployed": True, "task": task, "map50": map50, "onnx": onnx_path}


# ── Controller ────────────────────────────────────────────────────────────────


def build_pipeline(
    task: str,
    epochs: int,
    batch: int,
    model: str,
    min_map: float,
    nested_clearml: bool,
):
    """Wire the four steps into a ClearML PipelineController for one task."""
    from clearml import PipelineController, Task

    cfg = yaml.safe_load((ROOT / "ml" / task / "config.yaml").read_text())

    pipe = PipelineController(
        name=f"pipeline-{task}",
        project=cfg["clearml_project"],
        version="1.0.0",
        add_pipeline_tags=True,
    )
    # Tunable from the ClearML UI on re-run.
    pipe.add_parameter("epochs", epochs)
    pipe.add_parameter("batch", batch)
    pipe.add_parameter("model", model)
    pipe.add_parameter("min_map", min_map)

    common = {"repo_root": str(ROOT), "task": task}

    pipe.add_function_step(
        name="train",
        function=step_train,
        function_kwargs={
            **common,
            "epochs": "${pipeline.epochs}",
            "batch": "${pipeline.batch}",
            "model": "${pipeline.model}",
            "nested_clearml": nested_clearml,
        },
        function_return=["weights"],
        task_type=Task.TaskTypes.training,
        cache_executed_step=False,
    )
    pipe.add_function_step(
        name="evaluate",
        function=step_evaluate,
        function_kwargs={**common, "weights": "${train.weights}"},
        function_return=["metrics"],
        task_type=Task.TaskTypes.testing,
    )
    pipe.add_function_step(
        name="export",
        function=step_export,
        function_kwargs={**common, "weights": "${train.weights}"},
        function_return=["onnx_path"],
        task_type=Task.TaskTypes.custom,
    )
    pipe.add_function_step(
        name="deploy",
        function=step_deploy,
        function_kwargs={
            **common,
            "metrics": "${evaluate.metrics}",
            "onnx_path": "${export.onnx_path}",
            "min_map": "${pipeline.min_map}",
        },
        function_return=["deployed"],
        task_type=Task.TaskTypes.custom,
    )
    return pipe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MedVision train→eval→export→deploy pipeline")
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--model", default="", help="Override base model (else task config.yaml)")
    p.add_argument("--min-map", type=float, default=0.4,
                   help="Quality gate: minimum test mAP50 required to deploy")
    p.add_argument("--remote", action="store_true",
                   help="Enqueue steps on a ClearML agent instead of running locally")
    p.add_argument("--queue", default="default", help="Agent queue (with --remote)")
    p.add_argument("--nested-clearml", action="store_true",
                   help="Let train.py create its own nested ClearML task too")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pipe = build_pipeline(
        task=args.task,
        epochs=args.epochs,
        batch=args.batch,
        model=args.model,
        min_map=args.min_map,
        nested_clearml=args.nested_clearml,
    )

    if args.remote:
        print(f"Enqueuing pipeline '{args.task}' on queue '{args.queue}' ...")
        pipe.start(queue=args.queue)
    else:
        print(f"Running pipeline '{args.task}' locally (all steps on this host) ...")
        pipe.start_locally(run_pipeline_steps_locally=True)


if __name__ == "__main__":
    main()
