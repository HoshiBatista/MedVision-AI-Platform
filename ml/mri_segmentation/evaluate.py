"""
MRI Segmentation — Evaluation on test split.

Usage:
    python evaluate.py --weights runs/mri_segmentation/train/weights/best.pt
    python evaluate.py --weights best.pt --split val
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "ml"))

from ultralytics import YOLO

DATA_YAML = ROOT / "ml" / "data" / "mri_segmentation" / "data.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="Path to best.pt")
    p.add_argument("--split",   default="test", choices=["train", "val", "test"])
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--batch",   type=int, default=16)
    p.add_argument("--conf",    type=float, default=0.25)
    p.add_argument("--iou",     type=float, default=0.6)
    p.add_argument("--save-json", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)

    metrics = model.val(
        data=str(DATA_YAML),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device="mps",
        plots=True,
        save_json=args.save_json,
        verbose=True,
    )

    print("\n─── MRI Segmentation Results ───────────────────────────")
    print(f"  mAP50      : {metrics.seg.map50:.4f}")
    print(f"  mAP50-95   : {metrics.seg.map:.4f}")
    print(f"  Precision  : {metrics.seg.mp:.4f}")
    print(f"  Recall     : {metrics.seg.mr:.4f}")
    print(f"  Box mAP50  : {metrics.box.map50:.4f}")
    print("────────────────────────────────────────────────────────")

    print("\nPer-class (seg mAP50):")
    names = ["GLIOMA", "MENINGIOMA", "NOTUMOR", "PITUITARY"]
    for i, (name, ap) in enumerate(zip(names, metrics.seg.maps)):
        print(f"  {name:<14}: {ap:.4f}")

    if args.save_json:
        out = Path(args.weights).parent / "eval_results.json"
        out.write_text(json.dumps({
            "map50": metrics.seg.map50,
            "map":   metrics.seg.map,
            "mp":    metrics.seg.mp,
            "mr":    metrics.seg.mr,
        }, indent=2))
        print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
