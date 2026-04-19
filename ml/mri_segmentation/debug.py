"""
MRI Segmentation — Debug: run inference and visualise predictions on sample images.

Usage:
    python debug.py --weights best.pt                    # random test images
    python debug.py --weights best.pt --source path/img  # specific file or folder
    python debug.py --weights best.pt --n 20 --conf 0.3
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "ml"))

from ultralytics import YOLO

TEST_IMAGES = ROOT / "ml" / "data" / "mri_segmentation" / "test" / "images"
TEST_LABELS = ROOT / "ml" / "data" / "mri_segmentation" / "test" / "labels"

CLASSES = ["GLIOMA", "MENINGIOMA", "NOTUMOR", "PITUITARY"]
COLORS  = [(220, 50, 50), (50, 200, 50), (50, 50, 220), (220, 180, 50)]   # BGR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--source",  default=None, help="Image file or folder (default: random test images)")
    p.add_argument("--n",       type=int, default=9,   help="Number of images to show")
    p.add_argument("--conf",    type=float, default=0.25)
    p.add_argument("--iou",     type=float, default=0.6)
    p.add_argument("--save",    action="store_true", help="Save grid to debug_output/")
    return p.parse_args()


def load_gt_polygons(label_path: Path, img_w: int, img_h: int) -> list[tuple]:
    """Parse YOLO segmentation label → list of (class_id, polygon_pts)."""
    polys = []
    if not label_path.exists():
        return polys
    for line in label_path.read_text().strip().splitlines():
        vals = list(map(float, line.split()))
        cls  = int(vals[0])
        pts  = np.array(vals[1:]).reshape(-1, 2)
        pts[:, 0] *= img_w
        pts[:, 1] *= img_h
        polys.append((cls, pts.astype(int)))
    return polys


def draw_prediction(img: np.ndarray, result, conf_thr: float) -> np.ndarray:
    out = img.copy()
    if result.masks is None:
        return out
    for mask, box in zip(result.masks.xy, result.boxes):
        conf = float(box.conf)
        cls  = int(box.cls)
        if conf < conf_thr:
            continue
        color = COLORS[cls % len(COLORS)]
        pts   = mask.astype(np.int32).reshape(-1, 1, 2)
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)
        cv2.polylines(out, [pts], True, color, 2)
        x1, y1 = int(box.xyxy[0][0]), int(box.xyxy[0][1])
        label  = f"{CLASSES[cls]} {conf:.2f}"
        cv2.rectangle(out, (x1, y1 - 18), (x1 + len(label) * 9, y1), color, -1)
        cv2.putText(out, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return out


def draw_gt(img: np.ndarray, label_path: Path) -> np.ndarray:
    out = img.copy()
    h, w = img.shape[:2]
    for cls, pts in load_gt_polygons(label_path, w, h):
        color   = COLORS[cls % len(COLORS)]
        pts_cv  = pts.reshape(-1, 1, 2)
        cv2.polylines(out, [pts_cv], True, color, 2)
        if len(pts):
            x, y = pts[0]
            cv2.putText(out, CLASSES[cls], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out


def main() -> None:
    args   = parse_args()
    model  = YOLO(args.weights)

    if args.source:
        src   = Path(args.source)
        imgs  = [src] if src.is_file() else sorted(src.glob("*.jpg")) + sorted(src.glob("*.png"))
        imgs  = imgs[:args.n]
    else:
        all_imgs = sorted(TEST_IMAGES.glob("*.jpg")) + sorted(TEST_IMAGES.glob("*.png"))
        imgs     = random.sample(all_imgs, min(args.n, len(all_imgs)))

    print(f"Running inference on {len(imgs)} images...")
    results = model.predict(imgs, conf=args.conf, iou=args.iou, device="mps", verbose=False)

    cols = 3
    rows = (len(imgs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 8, rows * 4))
    fig.suptitle("MRI Brain Tumor Segmentation — Debug\n(left: GT  |  right: Prediction)", fontsize=14)
    axes = axes.flatten()

    for i, (img_path, result) in enumerate(zip(imgs, results)):
        img  = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        lbl  = TEST_LABELS / (Path(img_path).stem + ".txt")

        gt_img   = cv2.cvtColor(draw_gt(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), lbl), cv2.COLOR_BGR2RGB)
        pred_img = cv2.cvtColor(draw_prediction(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), result, args.conf), cv2.COLOR_BGR2RGB)

        ax_gt   = axes[i * 2]
        ax_pred = axes[i * 2 + 1]
        ax_gt.imshow(gt_img);   ax_gt.set_title(f"GT: {Path(img_path).stem[:25]}", fontsize=7)
        ax_pred.imshow(pred_img); ax_pred.set_title("Prediction", fontsize=7)
        ax_gt.axis("off"); ax_pred.axis("off")

    # Hide unused axes
    for j in range(len(imgs) * 2, len(axes)):
        axes[j].axis("off")

    # Legend
    patches = [mpatches.Patch(color=[c/255 for c in COLORS[i]], label=CLASSES[i]) for i in range(len(CLASSES))]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=9)
    plt.tight_layout()

    if args.save:
        out_dir = ROOT / "ml" / "runs" / "mri_segmentation" / "debug"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "debug_grid.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
