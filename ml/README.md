# MedVision ML

All training code for the three YOLOv11 (Ultralytics) tasks, tracked via ClearML.

| Task directory          | Model        | Output                         |
|-------------------------|--------------|--------------------------------|
| `mri_segmentation/`     | YOLOv11-seg  | instance masks + class + conf  |
| `pneumonia_detection/`  | YOLOv11-det  | bbox + class + conf            |
| `skin_classification/`  | YOLOv11-det  | bbox + class + conf            |

Each task directory has the same layout — `config.yaml` (model, data path,
hyperparameters, ClearML names) plus standalone scripts:

| Script            | Role                                                              |
|-------------------|------------------------------------------------------------------|
| `train.py`        | Ultralytics training + ClearML logging → writes `best.pt`        |
| `evaluate.py`     | Validate on a split; `--save-json` writes `eval_results.json`    |
| `export_onnx.py`  | Export `best.pt` → ONNX and copy into `triton_models/<task>/1/`  |
| `benchmark.py`    | Latency comparison (PyTorch vs ONNX Runtime)                     |
| `debug.py`        | Visual sanity check of predictions                               |

Shared helpers live in `shared/` (`clearml_utils.py`, `metrics.py`, `transforms.py`);
datasets in YOLO format live under `data/<task>/`.

---

## Pipeline (`pipeline.py`)

`pipeline.py` wires the per-task scripts into a single **ClearML
`PipelineController`** for the full **train → eval → export → deploy** flow. It
orchestrates the existing scripts — no training/export logic is duplicated.

```
 ┌─────────┐   weights   ┌──────────┐   metrics   ┌────────┐  onnx_path  ┌────────┐
 │  train  ├────────────▶│ evaluate ├────────────▶│ export ├────────────▶│ deploy │
 └─────────┘             └──────────┘             └────────┘             └────────┘
  train.py                evaluate.py              export_onnx.py    quality gate:
  → best.pt               → eval_results.json      → triton_models/  mAP50 ≥ --min-map
```

Each stage is its own ClearML step (task); artifacts flow between steps via
return values (`${train.weights}`, `${evaluate.metrics}`, `${export.onnx_path}`).
Project, run directory and ClearML names are read from each task's `config.yaml`,
so the pipeline stays in sync with the standalone scripts.

The **deploy** step is a quality gate: it only confirms the freshly exported ONNX
in `triton_models/<task>/1/` when the test `mAP50` clears `--min-map`; otherwise
it fails the run.

### Run it

```bash
# Whole DAG locally — every step on this machine, no ClearML agent needed:
python ml/pipeline.py --task mri_segmentation

# Tune hyperparameters / the deploy gate:
python ml/pipeline.py --task pneumonia_detection --epochs 50 --batch 8 --min-map 0.5

# Enqueue on a ClearML agent queue instead of running locally:
python ml/pipeline.py --task skin_classification --remote --queue default

# Or via the Makefile (ARGS is forwarded verbatim):
make pipeline TASK=pneumonia_detection ARGS="--min-map 0.5"
```

| Flag               | Default          | Meaning                                                  |
|--------------------|------------------|----------------------------------------------------------|
| `--task`           | *(required)*     | `mri_segmentation` \| `pneumonia_detection` \| `skin_classification` |
| `--epochs`         | `100`            | Training epochs                                          |
| `--batch`          | `16`             | Batch size                                              |
| `--model`          | task `config.yaml` | Override the base model                              |
| `--min-map`        | `0.4`            | Minimum test `mAP50` required to deploy                 |
| `--remote`         | off              | Enqueue steps on a ClearML agent instead of locally     |
| `--queue`          | `default`        | Agent queue (with `--remote`)                           |
| `--nested-clearml` | off              | Let `train.py` create its own nested ClearML task too   |

> The training device (`mps`/`cuda`/`cpu`) is governed by each task's
> `train.py`/`evaluate.py`, not by the pipeline. ClearML credentials are read
> from the environment / `~/.clearml.conf` (see `check_clearml.py`).
