"""
Pneumonia Detection — Training script.
Model: YOLOv11  |  Task: object detection
Classes: Atypical, Indeterminate, Typical

Usage:
    python train.py
    python train.py --epochs 50 --batch 16 --model yolo11m.pt
    python train.py --resume runs/pneumonia_detection/train/weights/last.pt
"""

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "ml"))

from shared.clearml_utils import init_task
from ultralytics import YOLO

DATA_YAML = ROOT / "ml" / "data" / "pneumonia_detection" / "data.yaml"
CONFIG     = Path(__file__).parent / "config.yaml"


def parse_args() -> argparse.Namespace:
    cfg = yaml.safe_load(CONFIG.read_text())
    p = argparse.ArgumentParser(description="Train pneumonia detection")
    p.add_argument("--model",    default=cfg["model"])
    p.add_argument("--epochs",   type=int,   default=cfg["epochs"])
    p.add_argument("--imgsz",    type=int,   default=cfg["imgsz"])
    p.add_argument("--batch",    type=int,   default=cfg["batch"])
    p.add_argument("--lr0",      type=float, default=cfg["lr0"])
    p.add_argument("--patience", type=int,   default=cfg["patience"])
    p.add_argument("--resume",   default=None)
    p.add_argument("--no-clearml", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = yaml.safe_load(CONFIG.read_text())

    if not args.no_clearml:
        task = init_task(
            project=cfg["clearml_project"],
            name=cfg["clearml_task"],
            tags=["yolo11", "pneumonia", "detection", "cxr"],
            config=vars(args),
        )

    model = YOLO(args.resume if args.resume else args.model)

    results = model.train(
        data=str(DATA_YAML),
        task="detect",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        lrf=cfg["lrf"],
        momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"],
        warmup_epochs=cfg["warmup_epochs"],
        patience=args.patience,
        hsv_h=cfg["hsv_h"],
        hsv_s=cfg["hsv_s"],
        hsv_v=cfg["hsv_v"],
        degrees=cfg["degrees"],
        translate=cfg["translate"],
        scale=cfg["scale"],
        flipud=cfg["flipud"],
        fliplr=cfg["fliplr"],
        mosaic=cfg["mosaic"],
        project=str(ROOT / cfg["project"]),
        name=cfg["name"],
        save=True,
        save_period=cfg["save_period"],
        resume=bool(args.resume),
        device="mps",
        plots=True,
        verbose=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nBest model: {best}")
    print(f"Results:    {results.save_dir}")

    if not args.no_clearml:
        from shared.clearml_utils import upload_model
        upload_model(task, best, "pneumonia-det-best")
        task.close()


if __name__ == "__main__":
    main()
