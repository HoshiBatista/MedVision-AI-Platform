"""
Skin Lesion Detection — Export trained model to ONNX (opset 17).

Usage:
    python export_onnx.py --weights runs/skin_classification/train/weights/best.pt
    python export_onnx.py --weights best.pt --imgsz 640 --batch 1
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "ml"))

TRITON_DEST = ROOT / "triton_models" / "skin_classification" / "1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights",   required=True)
    p.add_argument("--imgsz",     type=int, default=640)
    p.add_argument("--batch",     type=int, default=1)
    p.add_argument("--opset",     type=int, default=17)
    p.add_argument("--simplify",  action="store_true", default=True)
    p.add_argument("--no-triton", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)

    print(f"Exporting {args.weights} → ONNX opset {args.opset}  (batch={args.batch}, imgsz={args.imgsz})")
    onnx_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        batch=args.batch,
        opset=args.opset,
        simplify=args.simplify,
        dynamic=False,
    )

    onnx_file = Path(onnx_path)
    print(f"\nExported: {onnx_file}  ({onnx_file.stat().st_size / 1e6:.1f} MB)")

    if not args.no_triton:
        TRITON_DEST.mkdir(parents=True, exist_ok=True)
        dest = TRITON_DEST / "model.onnx"
        shutil.copy2(onnx_file, dest)
        print(f"Copied to Triton repo: {dest}")

    print("\nDone. Verify with:")
    print(f"  python -c \"import onnx; m=onnx.load('{onnx_file}'); onnx.checker.check_model(m); print('OK')\"")


if __name__ == "__main__":
    main()
