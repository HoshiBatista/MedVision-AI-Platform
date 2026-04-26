# ML Pipeline

All training runs on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) and is tracked in ClearML. Trained weights are exported to ONNX opset 17 and deployed to Triton.

---

## Models

### MRI Brain Tumor Segmentation

| Property | Value |
|---|---|
| Architecture | YOLOv11-seg (instance segmentation) |
| Classes | GLIOMA, MENINGIOMA, NOTUMOR, PITUITARY |
| Input size | 640×640 |
| ClearML project | `MedVision/MRI-Segmentation` |
| Triton model name | `mri_segmentation` |
| Dataset source | Roboflow — `brain-tumor-segmentation-r4in2 v2` |

**Training config** (`ml/mri_segmentation/config.yaml`):

| Param | Value |
|---|---|
| epochs | 100 |
| batch | 16 |
| lr0 | 0.01 |
| imgsz | 640 |
| patience | 20 |
| augmentation | HSV jitter, flips, mosaic, copy-paste |

**Evaluation metrics** (reported by `evaluate.py`):
- mAP50 (segmentation masks)
- mAP50-95
- Precision / Recall
- Per-class seg mAP50

---

### Pneumonia Detection (CXR)

| Property | Value |
|---|---|
| Architecture | YOLOv11-det (object detection) |
| Classes | Atypical, Indeterminate, Typical |
| Input size | 640×640 |
| ClearML project | `MedVision/Pneumonia-Detection` |
| Triton model name | `pneumonia_detection` |
| Dataset source | Roboflow — `pneumonia-detection-ssibx v3` (SIIM-CXR) |

**Training config** (`ml/pneumonia_detection/config.yaml`):

| Param | Value |
|---|---|
| epochs | 100 |
| batch | 32 |
| lr0 | 0.01 |
| imgsz | 640 |
| flipud | 0.0 (anatomical constraint — chest X-rays are not flipped vertically) |

---

### Skin Lesion Detection (HAM10000)

| Property | Value |
|---|---|
| Architecture | YOLOv11-det (object detection) |
| Classes | akiec, bcc, bkl, df, mel, nv, vasc |
| Input size | 640×640 |
| ClearML project | `MedVision/Skin-Classification` |
| Triton model name | `skin_classification` |
| Dataset source | Roboflow — `skin-lesion-jxjgm v8` (HAM10000-style) |

**Training config** (`ml/skin_classification/config.yaml`):

| Param | Value |
|---|---|
| epochs | 100 |
| batch | 16 |
| lr0 | 0.001 (lower — fine-tuning regime) |
| warmup_epochs | 5 |
| patience | 25 |
| flipud | 0.5 (dermoscopic images are orientation-agnostic) |

---

## Dataset Download

Datasets are hosted on Roboflow and downloaded in YOLOv11 format:

```bash
cd ml

# All three datasets
python download_datasets.py

# Single dataset
python download_datasets.py --task mri
python download_datasets.py --task pneumonia
python download_datasets.py --task skin
```

Requires `ROBOFLOW_API_KEY` in `.env`. Downloads to `ml/data/{task}/`.

Dataset layout after download:
```
ml/data/
├── mri_segmentation/
│   ├── data.yaml
│   ├── train/images/  train/labels/
│   ├── valid/images/  valid/labels/
│   └── test/images/   test/labels/
├── pneumonia_detection/   (same structure)
└── skin_classification/   (same structure)
```

---

## Training

```bash
# Default config
python ml/mri_segmentation/train.py

# Override params
python ml/mri_segmentation/train.py --epochs 50 --batch 8 --model yolo11m-seg.pt

# Resume from checkpoint
python ml/mri_segmentation/train.py --resume runs/mri_segmentation/train/weights/last.pt

# Skip ClearML (local-only run)
python ml/mri_segmentation/train.py --no-clearml
```

Same interface for `pneumonia_detection/train.py` and `skin_classification/train.py`.

Training output is saved to `runs/{task}/train/`:
```
runs/mri_segmentation/train/
├── weights/
│   ├── best.pt
│   └── last.pt
├── results.csv
├── confusion_matrix.png
└── ...
```

Device used: `mps` (Apple Silicon). Change to `cuda` for GPU server, `cpu` as fallback.

---

## Evaluation

```bash
python ml/mri_segmentation/evaluate.py --weights runs/mri_segmentation/train/weights/best.pt
python ml/mri_segmentation/evaluate.py --weights best.pt --split val --conf 0.3 --save-json
```

Prints per-class and aggregate metrics to stdout. Pass `--save-json` to write `eval_results.json` next to the weights file.

---

## ONNX Export

```bash
python ml/mri_segmentation/export_onnx.py --weights runs/mri_segmentation/train/weights/best.pt
```

This:
1. Exports to ONNX opset 17 with `simplify=True`
2. Copies `model.onnx` to `triton_models/mri_segmentation/1/model.onnx`

Options:
```
--imgsz 640      Static input size
--batch 1        Static batch size (Triton expects fixed batch for ONNX Runtime)
--opset 17
--no-triton      Skip copying to triton_models/
```

Verify the exported model:
```bash
python -c "import onnx; m=onnx.load('model.onnx'); onnx.checker.check_model(m); print('OK')"
```

---

## Benchmark

Measure latency and throughput before deploying:

```bash
# PyTorch (best.pt)
python ml/mri_segmentation/benchmark.py --weights best.pt --runs 200

# ONNX Runtime
python ml/mri_segmentation/benchmark.py --weights model.onnx --onnx
```

Output includes mean, median, P95, P99 latency and throughput (img/s).

---

## Debug Visualisation

Visually compare ground-truth masks vs model predictions on test images:

```bash
python ml/mri_segmentation/debug.py --weights best.pt
python ml/mri_segmentation/debug.py --weights best.pt --n 20 --conf 0.3
python ml/mri_segmentation/debug.py --weights best.pt --source path/to/image.jpg --save
```

Opens a matplotlib grid showing GT (left) vs prediction (right) for each image. Saved to `ml/runs/mri_segmentation/debug/debug_grid.png` with `--save`.

---

## ClearML Integration

All training scripts initialise a ClearML task via `ml/shared/clearml_utils.py`:

```python
from shared.clearml_utils import init_task, upload_model

task = init_task(
    project="MedVision/MRI-Segmentation",
    name="train-yolo11-seg",
    tags=["yolo11-seg", "mri", "segmentation"],
    config=vars(args),      # logs all hyperparameters
)
# ... train ...
upload_model(task, best_pt_path, "mri-seg-best")
task.close()
```

ClearML projects:
- `MedVision/MRI-Segmentation`
- `MedVision/Pneumonia-Detection`
- `MedVision/Skin-Classification`

Credentials are read from `CLEARML_API_ACCESS_KEY` / `CLEARML_API_SECRET_KEY` in `.env`.

Check connectivity:
```bash
python ml/check_clearml.py
```

---

## Triton Model Repository

After export, each model lives at:

```
triton_models/
├── mri_segmentation/
│   ├── config.pbtxt
│   └── 1/model.onnx
├── pneumonia_detection/
│   ├── config.pbtxt
│   └── 1/model.onnx
└── skin_classification/
    ├── config.pbtxt
    └── 1/model.onnx
```

`config.pbtxt` declares backend (`onnxruntime`), input/output tensor names and shapes, and dynamic batching settings. See `triton_models/*/config.pbtxt` for the exact tensor specs per model.

Key Triton constraint: always send tensors with shape `[batch, C, H, W]` — even for batch=1 Triton rejects wrong rank.
