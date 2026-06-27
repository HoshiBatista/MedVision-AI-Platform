# MedVision AI Platform

[![CI](https://github.com/HoshiBatista/MedVision-AI-Platform/actions/workflows/ci.yml/badge.svg)](https://github.com/HoshiBatista/MedVision-AI-Platform/actions/workflows/ci.yml)
[![CodeQL](https://img.shields.io/badge/CodeQL-enabled-2088FF?logo=github&logoColor=white)](https://github.com/HoshiBatista/MedVision-AI-Platform/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11--3.13-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-async-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?logo=typescript&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

Production-grade medical imaging analysis platform. Ingests DICOM / PNG / JPEG images, runs deep-learning inference on three clinical tasks, overlays explainability heatmaps, and generates AI-assisted radiology reports via a local LLM (Ollama / OpenBioLLM-8B). Inference runs on CPU out of the box (local ONNX Runtime) or on GPU via NVIDIA Triton — all on-prem, no external API.

---

## Clinical Tasks

| Task | Model | Classes | Input |
|---|---|---|---|
| MRI Brain Tumor Segmentation | YOLOv11-seg | GLIOMA, MENINGIOMA, NOTUMOR, PITUITARY | 640×640 |
| Pneumonia Detection (CXR) | YOLOv11-det | Atypical, Indeterminate, Typical | 640×640 |
| Skin Lesion Detection | YOLOv11-det | akiec, bcc, bkl, df, mel, nv, vasc | 640×640 |

All models are exported to ONNX (opset 17) and served either by a local ONNX Runtime (CPU, default) or NVIDIA Triton (GPU) — see [Inference backends](#inference-backends).

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Training | Ultralytics YOLO + ClearML |
| Model Serving | Local ONNX Runtime (CPU, default) **or** NVIDIA Triton (GPU) — via `INFERENCE_BACKEND` |
| Backend Services | FastAPI (async) |
| Task Queue | Celery + Redis |
| Storage | Local filesystem (`/data/studies`, `/data/heatmaps`) + PostgreSQL (metadata) |
| Gateway | Nginx + JWT |
| Observability | Prometheus + Grafana (OpenTelemetry/Jaeger planned) |
| Report Generation | Local LLM via Ollama (default OpenBioLLM-8B) — no external API |
| Frontend | React + TypeScript + Vite |
| Containers | Docker Compose (dev); Helm/K8s scaffolding planned |

---

## Quickstart

```bash
# 1. Copy and fill environment variables
cp .env.example .env

# 2. Download datasets from Roboflow
cd ml && python download_datasets.py

# 3. Train a model (starts a ClearML task)
make train TASK=mri_segmentation

# 4. Export to ONNX and copy to Triton repo
make export TASK=mri_segmentation VERSION=1

# 5. Start the full platform
make up        # CPU inference (local ONNX Runtime) — no GPU required
# or
make up-gpu    # GPU inference via NVIDIA Triton (needs the NVIDIA Container Toolkit)

# 6. Open the UI
open http://localhost:3000
```

### Inference backends

Two interchangeable backends, selected by `INFERENCE_BACKEND`:

| Variant | Command | Backend | Hardware |
|---|---|---|---|
| CPU (default) | `make up` | local ONNX Runtime in the analysis worker | any |
| GPU | `make up-gpu` | NVIDIA Triton Inference Server (`--profile triton`) | NVIDIA GPU + Container Toolkit |

Both serve the same exported ONNX models from `triton_models/<model>/1/`. The GPU
variant layers `docker-compose.gpu.yml` on top of the base stack.

---

## Repository Layout

```
.
├── ml/                    Training, evaluation, ONNX export
│   ├── shared/            ClearML helpers, metrics, transforms
│   ├── mri_segmentation/  YOLOv11-seg — brain tumor segmentation
│   ├── pneumonia_detection/ YOLOv11-det — chest X-ray
│   └── skin_classification/ YOLOv11-det — skin lesions (HAM10000)
├── services/
│   ├── gateway/           Nginx + JWT validation
│   ├── upload_service/    DICOM ingestion → local disk
│   ├── analysis_service/  Job orchestration; ONNX Runtime (CPU) / Triton (GPU) client
│   ├── report_service/    LLM report generation (thin Ollama client)
│   ├── auth_service/      JWT issuance, user management
│   └── gradcam_service/   EigenCAM explainability (ONNX, CPU)
├── triton_models/         ONNX model repository (served by ONNX Runtime or Triton)
├── frontend/              React UI
├── infra/                 Prometheus/Grafana config (Helm/Terraform: planned, empty)
└── tests/                 Integration + E2E test suites
```

---

## Documentation

| Document | Description |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture, service contracts, data flow |
| [ML_PIPELINE.md](docs/ML_PIPELINE.md) | Model specs, training configs, datasets, evaluation |
| [API.md](docs/API.md) | Full REST API reference for all services |
| [DEVELOPMENT.md](docs/DEVELOPMENT.md) | Local dev setup, tooling, code standards |
| [DEPLOYMENT.md](docs/DEPLOYMENT.md) | Docker Compose, Kubernetes/Helm, environment config |

---

## License

[MIT](LICENSE)
