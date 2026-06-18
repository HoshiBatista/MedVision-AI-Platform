# MedVision AI Platform

[![Python Quality](https://github.com/HoshiBatista/MedVision-AI-Platform/actions/workflows/python-quality.yml/badge.svg)](https://github.com/HoshiBatista/MedVision-AI-Platform/actions/workflows/python-quality.yml)
[![Python Tests](https://github.com/HoshiBatista/MedVision-AI-Platform/actions/workflows/python-tests.yml/badge.svg)](https://github.com/HoshiBatista/MedVision-AI-Platform/actions/workflows/python-tests.yml)
[![Frontend](https://github.com/HoshiBatista/MedVision-AI-Platform/actions/workflows/frontend.yml/badge.svg)](https://github.com/HoshiBatista/MedVision-AI-Platform/actions/workflows/frontend.yml)
[![Docker](https://github.com/HoshiBatista/MedVision-AI-Platform/actions/workflows/docker.yml/badge.svg)](https://github.com/HoshiBatista/MedVision-AI-Platform/actions/workflows/docker.yml)
[![CodeQL](https://github.com/HoshiBatista/MedVision-AI-Platform/actions/workflows/codeql.yml/badge.svg)](https://github.com/HoshiBatista/MedVision-AI-Platform/actions/workflows/codeql.yml)
[![Repo Hygiene](https://github.com/HoshiBatista/MedVision-AI-Platform/actions/workflows/meta.yml/badge.svg)](https://github.com/HoshiBatista/MedVision-AI-Platform/actions/workflows/meta.yml)

Production-grade medical imaging analysis platform. Ingests DICOM / PNG / JPEG images, runs deep-learning inference on three clinical tasks, overlays explainability heatmaps, and generates AI-assisted radiology reports via a local BioGPT model.

---

## Clinical Tasks

| Task | Model | Classes | Input |
|---|---|---|---|
| MRI Brain Tumor Segmentation | YOLOv11-seg | GLIOMA, MENINGIOMA, NOTUMOR, PITUITARY | 640×640 |
| Pneumonia Detection (CXR) | YOLOv11-det | Atypical, Indeterminate, Typical | 640×640 |
| Skin Lesion Detection | YOLOv11-det | akiec, bcc, bkl, df, mel, nv, vasc | 640×640 |

All models are exported to ONNX (opset 17) and served via NVIDIA Triton Inference Server.

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Training | Ultralytics YOLO + ClearML |
| Model Serving | NVIDIA Triton (ONNX Runtime backend) |
| Backend Services | FastAPI (async) |
| Task Queue | Celery + Redis |
| Storage | MinIO (images/results), PostgreSQL (metadata) |
| Gateway | Nginx + JWT |
| Observability | Prometheus + Grafana + Jaeger (OpenTelemetry) |
| Report Generation | Anthropic API (claude-sonnet-4) |
| Frontend | React + TypeScript + Vite |
| Containers | Docker Compose (dev) / Helm + Kubernetes (prod) |

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
make up

# 6. Open the UI
open http://localhost:3000
```

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
│   ├── upload_service/    DICOM ingestion MinIO
│   ├── analysis_service/  Job orchestration, Triton gRPC client
│   ├── report_service/    LLM report generation (Anthropic)
│   ├── auth_service/      JWT issuance, user management
│   └── gradcam_service/   GradCAM / GradCAM++ explainability
├── triton_models/         Triton model repository (config + ONNX)
├── frontend/              React UI
├── infra/                 Helm charts, Terraform, Prometheus/Grafana
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
