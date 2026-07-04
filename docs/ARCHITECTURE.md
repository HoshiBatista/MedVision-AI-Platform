# System Architecture

## Overview

MedVision AI is a microservices platform for medical image analysis. A clinician uploads an image through the frontend; the platform runs deep-learning inference, generates explainability heatmaps, and produces an AI-assisted radiology report — all asynchronously via a job queue. Everything runs on-prem: no object storage and no external LLM API.

---

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Browser / Client                                                       │
│  React + TypeScript + Vite                                             │
│  • DicomViewer  • HeatmapOverlay  • ResultsPanel / ReportViewer        │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │ HTTP
┌──────────────────────────────▼──────────────────────────────────────────┐
│  Gateway  (Nginx)                                                       │
│  • Reverse proxy → each service   • Rate limiting                       │
│  • Serves /static/{studies,heatmaps} from the shared volumes           │
└──┬────────┬─────────┬──────────┬──────────┬──────────────┬──────────────┘
   ▼        ▼         ▼          ▼          ▼              ▼
 Auth     Upload   Analysis   GradCAM    Report        Frontend
 Service  Service  Service    Service    Service        (static)
 (JWT)    (DICOM   (orchestr. (EigenCAM) (Ollama
          → disk)   + Celery)            client)
   │        │         │                     │
   └────┬───┘         ▼                     ▼
        │        Celery + Redis          Ollama
   PostgreSQL    (async jobs)            (OpenBioLLM-8B,
   (metadata,        │                    CPU or GPU)
    results)         ▼
        │     Inference backend (INFERENCE_BACKEND)
        │       • onnx   — local ONNX Runtime in the worker (CPU, default)
        │       • triton — NVIDIA Triton Inference Server (GPU, profile)
        ▼
  Local volumes:  study_data (/data/studies),  heatmap_data (/data/heatmaps)
```

---

## Services

### gateway
- **Role**: Nginx reverse proxy and routing boundary
- **Responsibilities**: route to upstream services, rate limiting, serve study/heatmap files under `/static/`
- **Port**: 80

### auth_service
- **Role**: Identity and access management
- **Responsibilities**: user registration/login/logout, JWT access-token issuance (HS256), password hashing, admin user management (RBAC)
- **Stack**: FastAPI, PostgreSQL, `python-jose`, `passlib[bcrypt]`
- **Port**: 8001

### upload_service
- **Role**: Ingestion boundary for medical images
- **Responsibilities**: validate DICOM/PNG/JPEG, apply DICOM pixel rescaling (`RescaleSlope`/`RescaleIntercept`), store on the local `study_data` volume, write the study record to PostgreSQL
- **Stack**: FastAPI, `pydicom`, Pillow
- **Port**: 8002

### analysis_service
- **Role**: Job orchestrator
- **Responsibilities**: receive analysis requests, push jobs to Celery, poll/return results
- **Celery worker**: read image from disk → preprocess (640×640, normalize [0,1]) → run inference (ONNX Runtime on CPU, or Triton over HTTP on GPU) → post-process YOLO output → request a heatmap from gradcam_service (non-fatal) → persist results
- **Stack**: FastAPI, Celery, `onnxruntime` (CPU) / `tritonclient[http]` (GPU)
- **Port**: 8003

### gradcam_service
- **Role**: Standalone explainability service
- **Responsibilities**: compute **EigenCAM** (gradient-free) on the ONNX model, render a jet-colormap overlay PNG, write it to the `heatmap_data` volume
- **Stack**: FastAPI, `onnx`, `onnxruntime`, Pillow (CPU only)
- **Port**: 8004

### report_service
- **Role**: AI-assisted radiology report generation
- **Responsibilities**: assemble structured findings, render Jinja2 prompt templates, call a local **Ollama** server for text generation, append a fixed disclaimer, store the report
- **Stack**: FastAPI, `httpx` (thin Ollama client — no torch/transformers), Jinja2
- **Port**: 8005

### ollama
- **Role**: Local LLM inference server for report generation
- **Responsibilities**: serve a quantised GGUF medical model (default **OpenBioLLM-8B Q8**) via llama.cpp; same weights run on CPU (default) or GPU (gpu overlay)
- **Port**: 11434

---

## Data Flow

```
1. Client  → POST /api/v1/upload  (multipart: file, ?modality=)
              upload_service validates, stores under /data/studies/{study_id}/
              inserts study record in PostgreSQL → returns the study (201, id)

2. Client  → POST /api/v1/analyze/  { study_id, task }
              analysis_service creates a job, queues a Celery task → returns job_id (202)

3. Celery worker:
   a. Read the study file from /data/studies
   b. Preprocess: resize to 640×640, normalize to [0,1] → [1,3,640,640] FP32
   c. Inference via INFERENCE_BACKEND (onnx: local ONNX Runtime | triton: HTTP)
   d. Post-process: detection → bboxes/conf; segmentation → boxes + mask coeffs
   e. POST /api/v1/explain → gradcam_service → heatmap PNG on /data/heatmaps (non-fatal)
   f. Update the job in PostgreSQL (status=completed, results JSON incl. heatmap_path)

4. Client  → GET /api/v1/results/{job_id}
              Returns structured results; files are served at /static/heatmaps/...

5. Client  → POST /api/v1/reports/generate  { study_id, modality, job_ids, findings }
              report_service builds a Jinja2 prompt, calls Ollama, stores the report (202)

6. Client  → GET /api/v1/reports/{report_id}
              Returns the report (status, modality, content)
```

---

## Storage Layout

Local Docker volumes (no object storage):

```
study_data    → /data/studies/{study_id}/<original_filename>
heatmap_data  → /data/heatmaps/heatmap_<digest>.png
PostgreSQL    → study metadata, analysis jobs/results, reports, users
ollama_data   → pulled LLM weights (/root/.ollama)
```

The gateway mounts `study_data` and `heatmap_data` read-only and serves them under `/static/studies/` and `/static/heatmaps/`.

---

## Authentication

All API calls (except `/health`, `/ready`, `/metrics`, `/docs`) require a Bearer JWT:

```
Authorization: Bearer <access_token>
```

Tokens are issued by `auth_service` and validated by `upload_service` / `analysis_service`
using the shared `JWT_SECRET_KEY` (HS256). Access-token lifetime is configurable via
`ACCESS_TOKEN_EXPIRE_MINUTES` (default 30 min). Refresh tokens are not implemented yet.

---

## Observability

Every FastAPI service exposes:

| Surface | Endpoint | Purpose |
|---|---|---|
| Health | `GET /health` | Liveness: `{"status": "ok"}` |
| Ready | `GET /ready` | Readiness: checks dependencies (DB, Redis, inference backend, model) |
| Metrics | `GET /metrics` | Prometheus exposition |

Prometheus + Grafana are wired (enable with `--profile monitoring`); dashboards live in
`infra/monitoring/grafana/dashboards/`, alert rules in `infra/monitoring/alerts.yml`.
Distributed tracing uses OpenTelemetry OTLP export to Jaeger (same monitoring profile;
`make up-monitoring` sets `OTEL_TRACES_ENABLED=true`). Spans cover HTTP handlers,
SQLAlchemy queries, outbound httpx calls, Triton/ONNX inference, and Celery tasks.

---

## Security Notes

- Secrets only via environment variables — never in code or committed files
- `.env` is gitignored; use `.env.example` as the template
- DICOM pixel rescaling is always applied before inference (skipping it yields wrong intensities)
- All Python services run as non-root (`appuser`, uid 10001) with `no-new-privileges`
- `DOCS_ENABLED=false` in production disables `/docs`
