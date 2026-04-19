"""
MRI Brain Tumor Segmentation — Training script.
Model: YOLOv11-seg  |  Task: instance segmentation
Classes: GLIOMA, MENINGIOMA, NOTUMOR, PITUITARY

Usage:
    python train.py
    python train.py --epochs 50 --batch 8 --model yolo11m-seg.pt
    python train.py --resume runs/mri_segmentation/train/weights/last.pt
"""

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent   # project root
sys.path.insert(0, str(ROOT / "ml"))

from shared.clearml_utils import init_task
from ultralytics import YOLO

DATA_YAML = ROOT / "ml" / "data" / "mri_segmentation" / "data.yaml"
CONFIG     = Path(__file__).parent / "config.yaml"


def parse_args() -> argparse.Namespace:
    cfg = yaml.safe_load(CONFIG.read_text())
    p = argparse.ArgumentParser(description="Train MRI brain tumor segmentation")
    p.add_argument("--model",   default=cfg["model"])
    p.add_argument("--epochs",  type=int,   default=cfg["epochs"])
    p.add_argument("--imgsz",   type=int,   default=cfg["imgsz"])
    p.add_argument("--batch",   type=int,   default=cfg["batch"])
    p.add_argument("--lr0",     type=float, default=cfg["lr0"])
    p.add_argument("--patience",type=int,   default=cfg["patience"])
    p.add_argument("--resume",  default=None, help="Resume from checkpoint path")
    p.add_argument("--no-clearml", action="store_true", help="Disable ClearML logging")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = yaml.safe_load(CONFIG.read_text())

    if not args.no_clearml:
        task = init_task(
            project=cfg["clearml_project"],
            name=cfg["clearml_task"],
            tags=["yolo11-seg", "mri", "segmentation"],
            config=vars(args),
        )

    model = YOLO(args.resume if args.resume else args.model)

    results = model.train(
        data=str(DATA_YAML),
        task="segment",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        lrf=cfg["lrf"],
        momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"],
        warmup_epochs=cfg["warmup_epochs"],
        patience=args.patience,
        dropout=cfg["dropout"],
        hsv_h=cfg["hsv_h"],
        hsv_s=cfg["hsv_s"],
        hsv_v=cfg["hsv_v"],
        degrees=cfg["degrees"],
        translate=cfg["translate"],
        scale=cfg["scale"],
        flipud=cfg["flipud"],
        fliplr=cfg["fliplr"],
        mosaic=cfg["mosaic"],
        copy_paste=cfg["copy_paste"],
        project=str(ROOT / cfg["project"]),
        name=cfg["name"],
        save=True,
        save_period=cfg["save_period"],
        resume=bool(args.resume),
        device="mps",           # Apple Silicon; use "cuda" on GPU server, "cpu" as fallback
        plots=True,
        verbose=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nBest model: {best}")
    print(f"Results:    {results.save_dir}")

    if not args.no_clearml:
        from shared.clearml_utils import upload_model
        upload_model(task, best, "mri-seg-best")
        task.close()


if __name__ == "__main__":
    main()
