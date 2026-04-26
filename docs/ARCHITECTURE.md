# System Architecture

## Overview

MedVision AI is a microservices platform for medical image analysis. A clinician uploads an image through the frontend; the platform runs deep-learning inference, generates explainability heatmaps, and produces an AI-assisted radiology report — all asynchronously via a job queue.

---

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Browser / Client                                                       │
│  React + TypeScript + Vite                                              │
│  • DicomViewer (cornerstone.js)                                         │
│  • HeatmapOverlay (GradCAM)                                             │
│  • ResultsPanel / ReportViewer                                          │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │ HTTPS
┌──────────────────────────────▼──────────────────────────────────────────┐
│  Gateway  (Nginx)                                                       │
│  • JWT validation                                                       │
│  • Reverse proxy → each service                                         │
│  • Rate limiting / TLS termination                                      │
└──┬───────────┬───────────┬────────────┬───────────────┬─────────────────┘
   │           │           │            │               │
   ▼           ▼           ▼            ▼               ▼
Auth      Upload       Analysis     Report         GradCAM
Service   Service      Service      Service        Service
(JWT)     (DICOM)      (Triton      (Anthropic     (CAM
          → MinIO       gRPC)        API)           heatmaps)
   │           │           │
   └──────┬────┘           │
          │                │
     PostgreSQL         Celery + Redis
     (metadata,         (async job queue)
      results)
          │
        MinIO
        (DICOM, images,
         masks, heatmaps)
          │
      Triton Inference Server
      (ONNX Runtime — GPU)
      • mri_segmentation
      • pneumonia_detection
      • skin_classification
```

---

## Services

### gateway
- **Role**: Nginx reverse proxy and security boundary
- **Responsibilities**: TLS termination, JWT token validation, rate limiting, routing to upstream services
- **Port**: 80 / 443

### auth_service
- **Role**: Identity and access management
- **Responsibilities**: User registration/login, JWT issuance (access + refresh tokens), password hashing
- **Stack**: FastAPI, PostgreSQL, `python-jose`, `passlib`
- **Port**: 8001

### upload_service
- **Role**: Ingestion boundary for medical images
- **Responsibilities**: Validate DICOM/PNG/JPEG, apply DICOM pixel rescaling (`RescaleSlope` / `RescaleIntercept`), store in MinIO, write study record to PostgreSQL
- **Stack**: FastAPI, `pydicom`, `boto3`/MinIO client
- **Port**: 8002

### analysis_service
- **Role**: Job orchestrator
- **Responsibilities**: Receive analysis requests, push jobs to Celery, poll/return results
- **Celery workers**: fetch image from MinIO → preprocess → gRPC call to Triton → post-process → store results → call GradCAM service
- **Stack**: FastAPI, Celery, `tritonclient[grpc]`
- **Port**: 8003

### gradcam_service
- **Role**: Standalone explainability service
- **Responsibilities**: Compute GradCAM / GradCAM++ / ScoreCAM heatmaps on CPU, render jet-colormap overlays, store in MinIO
- **Stack**: FastAPI, PyTorch (CPU only), OpenCV
- **Port**: 8004
- **Note**: Runs on CPU — GPU is reserved exclusively for Triton inference

### report_service
- **Role**: AI-assisted radiology report generation
- **Responsibilities**: Assemble structured findings, render Jinja2 prompt templates, call Anthropic claude-sonnet-4, store report as Markdown/PDF
- **Stack**: FastAPI, `anthropic` SDK, Jinja2
- **Port**: 8005
- **Model**: `claude-sonnet-4-20250514`, max 2048 tokens

---

## Data Flow

```
1. Client  → POST /api/v1/upload
              Upload service validates image, stores in MinIO
              Inserts study record in PostgreSQL → returns study_id

2. Client  → POST /api/v1/analyze  { study_id, task }
              Analysis service creates job record
              Pushes Celery task → returns job_id

3. Celery Worker:
   a. Fetch image bytes from MinIO
   b. Preprocess: resize to 640×640, normalize to [0,1]
   c. gRPC request to Triton → raw prediction tensor
   d. Post-process:
      - Segmentation: decode polygons, compute area
      - Detection:    decode bboxes, confidence scores
   e. POST /api/v1/explain → GradCAM service → heatmap stored in MinIO
   f. Update job record in PostgreSQL (status=completed, results JSON)

4. Client  → GET /api/v1/results/{job_id}
              Returns structured results + presigned MinIO URLs (1h TTL)

5. Client  → POST /api/v1/reports/generate  { study_id, job_ids }
              Report service builds Jinja2 prompt
              Calls Anthropic API → stores report
              Returns report_id

6. Client  → GET /api/v1/reports/{report_id}
              Returns Markdown/PDF report
```

---

## Storage Layout

```
MinIO bucket: medvision
├── studies/{study_id}/
│   └── original.{dcm|png|jpg}
├── results/{job_id}/
│   ├── mask.png            (segmentation)
│   ├── gradcam.png         (heatmap overlay)
│   └── results.json
└── reports/{report_id}/
    └── report.md
```

---

## Authentication

All API calls (except `/health`, `/ready`, `/docs`) require a Bearer JWT:

```
Authorization: Bearer <access_token>
```

Tokens are issued by `auth_service`. The gateway validates the signature using `JWT_SECRET_KEY` before forwarding.

Token lifetimes:
- Access token: 30 minutes (configurable via `ACCESS_TOKEN_EXPIRE_MINUTES`)
- Refresh token: 7 days (configurable via `REFRESH_TOKEN_EXPIRE_DAYS`)

---

## Observability

Every service exposes three observability surfaces:

| Surface | Endpoint | Purpose |
|---|---|---|
| Health | `GET /health` | Liveness: returns `{"status": "ok"}` |
| Ready | `GET /ready` | Readiness: checks DB, Redis, Triton |
| Metrics | `GET /metrics` | Prometheus exposition (request rate, latency, error rate) |

Distributed tracing uses OpenTelemetry → Jaeger. Every Triton gRPC call and every database query carries a span.

Alerting rules (defined in `infra/monitoring/alerts.yml`) cover:
- Service down (no scrape for 2 min)
- High error rate (5xx > 5%)
- Inference latency P99 > 2s
- Celery queue depth > 50

---

## Security Notes

- Secrets only via environment variables — never in code or committed files
- `.env` is gitignored; use `.env.example` as the template
- DICOM pixel rescaling is always applied before inference (footgun: skipping this produces wrong intensity values)
- MinIO presigned URLs expire in 1h — never cache them beyond that
- `DOCS_ENABLED=false` in production (disables `/docs` Swagger UI)
