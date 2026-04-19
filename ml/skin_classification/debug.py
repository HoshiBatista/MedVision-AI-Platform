"""
Skin Lesion Detection — Debug: run inference and visualise predictions.

Usage:
    python debug.py --weights best.pt
    python debug.py --weights best.pt --source path/to/image.jpg
    python debug.py --weights best.pt --n 12 --conf 0.3 --save
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

TEST_IMAGES = ROOT / "ml" / "data" / "skin_classification" / "test" / "images"
TEST_LABELS = ROOT / "ml" / "data" / "skin_classification" / "test" / "labels"

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
COLORS  = [
    (220, 50,  50),   # akiec — red
    (50,  200, 50),   # bcc   — green
    (50,  50,  220),  # bkl   — blue
    (220, 180, 50),   # df    — yellow
    (180, 50,  220),  # mel   — purple
    (50,  200, 200),  # nv    — cyan
    (220, 120, 50),   # vasc  — orange
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--source",  default=None)
    p.add_argument("--n",       type=int,   default=9)
    p.add_argument("--conf",    type=float, default=0.25)
    p.add_argument("--iou",     type=float, default=0.6)
    p.add_argument("--save",    action="store_true")
    return p.parse_args()


def load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> list[tuple]:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().splitlines():
        cls, cx, cy, bw, bh = map(float, line.split())
        x1 = int((cx - bw / 2) * img_w)
        y1 = int((cy - bh / 2) * img_h)
        x2 = int((cx + bw / 2) * img_w)
        y2 = int((cy + bh / 2) * img_h)
        boxes.append((int(cls), x1, y1, x2, y2))
    return boxes


def draw_boxes(img: np.ndarray, boxes: list, show_conf: bool = False) -> np.ndarray:
    out = img.copy()
    for item in boxes:
        if show_conf:
            cls, x1, y1, x2, y2, conf = item
            text = f"{CLASSES[cls]} {conf:.2f}"
        else:
            cls, x1, y1, x2, y2 = item
            text = CLASSES[cls]
        color = COLORS[cls % len(COLORS)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(out, (x1, y1 - 18), (x1 + len(text) * 9, y1), color, -1)
        cv2.putText(out, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return out


def main() -> None:
    args  = parse_args()
    model = YOLO(args.weights)

    if args.source:
        src  = Path(args.source)
        imgs = [src] if src.is_file() else sorted(src.glob("*.jpg")) + sorted(src.glob("*.png"))
        imgs = imgs[:args.n]
    else:
        all_imgs = sorted(TEST_IMAGES.glob("*.jpg")) + sorted(TEST_IMAGES.glob("*.png"))
        imgs     = random.sample(all_imgs, min(args.n, len(all_imgs)))

    print(f"Running inference on {len(imgs)} images...")
    results = model.predict(imgs, conf=args.conf, iou=args.iou, device="mps", verbose=False)

    cols = 3
    rows = (len(imgs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 8, rows * 4))
    fig.suptitle("Skin Lesion Detection — Debug\n(left: GT  |  right: Prediction)", fontsize=14)
    axes = axes.flatten()

    for i, (img_path, result) in enumerate(zip(imgs, results)):
        img  = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        lbl  = TEST_LABELS / (Path(img_path).stem + ".txt")

        gt_boxes   = load_gt_boxes(lbl, w, h)
        pred_boxes = []
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            pred_boxes.append((int(box.cls), x1, y1, x2, y2, float(box.conf)))

        gt_bgr   = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        gt_img   = cv2.cvtColor(draw_boxes(gt_bgr, gt_boxes), cv2.COLOR_BGR2RGB)
        pred_img = cv2.cvtColor(draw_boxes(gt_bgr.copy(), pred_boxes, show_conf=True), cv2.COLOR_BGR2RGB)

        ax_gt   = axes[i * 2]
        ax_pred = axes[i * 2 + 1]
        ax_gt.imshow(gt_img);    ax_gt.set_title(f"GT: {Path(img_path).stem[:25]}", fontsize=7)
        ax_pred.imshow(pred_img); ax_pred.set_title("Prediction", fontsize=7)
        ax_gt.axis("off"); ax_pred.axis("off")

    for j in range(len(imgs) * 2, len(axes)):
        axes[j].axis("off")

    patches = [mpatches.Patch(color=[c/255 for c in COLORS[i]], label=CLASSES[i]) for i in range(len(CLASSES))]
    fig.legend(handles=patches, loc="lower center", ncol=7, fontsize=8)
    plt.tight_layout()

    if args.save:
        out_dir = ROOT / "ml" / "runs" / "skin_classification" / "debug"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "debug_grid.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
