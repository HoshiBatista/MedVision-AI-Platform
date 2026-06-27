# API Reference

All services sit behind the Nginx gateway. In development the base URL is `http://localhost`.

Most endpoints (except `/health`, `/ready`, `/metrics`) require:
```
Authorization: Bearer <access_token>
```

On error services return:
```json
{ "detail": "human-readable message", "error_code": "SNAKE_CASE_CODE" }
```

---

## Auth Service

### POST /api/v1/auth/register
Register a new user.

**Request**
```json
{ "email": "user@hospital.org", "password": "s3cret", "full_name": "Dr Who" }
```

**Response** `201`
```json
{ "id": 1, "email": "user@hospital.org", "full_name": "Dr Who", "role": "user", "is_active": true }
```

---

### POST /api/v1/auth/login
Obtain an access token (form-encoded, OAuth2 password flow).

**Request** — `application/x-www-form-urlencoded`
```
username=user@hospital.org&password=s3cret
```

**Response** `200`
```json
{ "access_token": "eyJ...", "token_type": "bearer", "expires_in": 1800 }
```

> Refresh tokens are not implemented yet — clients re-login when the access token expires.

### POST /api/v1/auth/logout
Stateless acknowledgement (`204`). Also: `GET /api/v1/users/me`, `PATCH /api/v1/users/me`, and admin routes under `/api/v1/admin/users`.

---

## Upload Service

### POST /api/v1/upload
Upload a medical image. The modality is passed as a query parameter.

**Request** — `multipart/form-data`, `?modality=MRI|CXR|DERM`
```
file: <DICOM | PNG | JPEG binary>
```

**Response** `201`
```json
{
  "id": "uuid",
  "user_id": 1,
  "modality": "DERM",
  "original_filename": "lesion.png",
  "file_path": "/data/studies/{id}/lesion.png",
  "file_size_bytes": 12345,
  "status": "uploaded",
  "created_at": "2026-06-27T12:00:00Z"
}
```

**Error codes**: `422` (unsupported type / invalid modality / invalid DICOM), `413` (exceeds `MAX_UPLOAD_SIZE_MB`).

---

## Analysis Service

### POST /api/v1/analyze/
Submit an analysis job (note the trailing slash).

**Request**
```json
{ "study_id": "uuid", "task": "segmentation" | "detection" | "classification", "config": {} }
```

**Response** `202`
```json
{ "job_id": "uuid", "status": "queued" }
```

---

### GET /api/v1/results/{job_id}
Poll the result of an analysis job.

**Response** `200`
```json
{
  "job_id": "uuid",
  "study_id": "uuid",
  "task": "classification",
  "status": "queued" | "running" | "completed" | "failed",
  "results": {
    "model": "skin_classification",
    "findings": [ { "bbox": [x1,y1,x2,y2], "confidence": 0.94, "class_id": 4, "area_px2": 3842.0 } ],
    "num_detections": 1,
    "top_confidence": 0.94,
    "orig_size": [w, h],
    "heatmap_path": "/data/heatmaps/heatmap_abc123.png",
    "processing_time_s": 0.42
  },
  "error": null
}
```

The heatmap is served by the gateway at `/static/heatmaps/<filename>` (basename of `heatmap_path`).

---

## GradCAM Service

### POST /api/v1/explain
Compute an EigenCAM heatmap for a model + image. Called by the analysis worker; failures are non-fatal there.

**Request**
```json
{
  "model_name": "skin_classification" | "pneumonia_detection" | "mri_segmentation",
  "image_path": "/data/studies/{study_id}/lesion.png",
  "target_class": 4
}
```

**Response** `200`
```json
{
  "heatmap_path": "/data/heatmaps/heatmap_abc123.png",
  "cam_shape": [40, 40],
  "top_regions": [ { "quadrant": 0, "mean_activation": 0.61, "max_activation": 1.0 } ],
  "num_detections": 1
}
```

---

## Report Service

### POST /api/v1/reports/generate
Generate an AI report from structured findings. Requires the LLM model to be loaded
(`/ready` → `model: true`), otherwise returns `503`.

**Request**
```json
{
  "study_id": "uuid",
  "modality": "MRI" | "CXR" | "DERM",
  "job_ids": ["uuid"],
  "findings": { "confidence": 0.94, "labels": ["lesion"], "bbox_count": 1, "raw": {} },
  "patient_context": { "age": 45, "sex": "F", "clinical_indication": "Headache" }
}
```

**Response** `202`
```json
{ "report_id": "uuid", "status": "pending", "modality": "MRI" }
```

---

### GET /api/v1/reports/{report_id}
Retrieve a generated report. Also: `GET /api/v1/reports/study/{study_id}`.

**Response** `200`
```json
{
  "report_id": "uuid",
  "status": "completed" | "generating" | "pending" | "failed",
  "modality": "MRI",
  "content": "...generated report...\n\nDISCLAIMER: ...",
  "created_at": "2026-06-27T12:00:00Z"
}
```

Report sections: clinical indication, technique (from modality), findings (LLM expansion),
impression (LLM summary), and an always-appended AI-assistance disclaimer.

---

## Health Endpoints

All services expose these (no auth):

```
GET /health   → { "status": "ok" }
GET /ready    → { "status": "ok"|"degraded", "checks": { ... } }
GET /metrics  → Prometheus text format
```

Readiness checks per service:
- `upload_service`: PostgreSQL
- `analysis_service`: PostgreSQL, Redis, and (only when `INFERENCE_BACKEND=triton`) Triton
- `gradcam_service`: process liveness
- `report_service`: PostgreSQL, Ollama model loaded (`model: true`)
