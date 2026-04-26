# API Reference

All services sit behind the Nginx gateway. In development the base URL is `http://localhost`.

Every endpoint (except `/health`, `/ready`) requires:
```
Authorization: Bearer <access_token>
Content-Type: application/json
```

On error every service returns:
```json
{ "detail": "human-readable message", "error_code": "SNAKE_CASE_CODE" }
```

---

## Auth Service

### POST /api/v1/auth/register
Register a new user.

**Request**
```json
{ "email": "user@hospital.org", "password": "s3cret" }
```

**Response** `201`
```json
{ "id": "uuid", "email": "user@hospital.org" }
```

---

### POST /api/v1/auth/login
Obtain access + refresh tokens.

**Request** (form data)
```
username=user@hospital.org&password=s3cret
```

**Response** `200`
```json
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer"
}
```

---

### POST /api/v1/auth/refresh
Exchange a refresh token for a new access token.

**Request**
```json
{ "refresh_token": "eyJ..." }
```

**Response** `200`
```json
{ "access_token": "eyJ...", "token_type": "bearer" }
```

---

## Upload Service

### POST /api/v1/upload
Upload a medical image and queue it for ingestion.

**Request** — `multipart/form-data`
```
file:     <DICOM | PNG | JPEG binary>
modality: "MRI" | "CXR" | "DERM"
```

**Response** `202`
```json
{
  "study_id": "uuid",
  "status": "queued",
  "storage_path": "s3://medvision/studies/{study_id}/original.dcm"
}
```

**Error codes**
| Code | Meaning |
|---|---|
| `INVALID_DICOM` | File fails DICOM validation |
| `UNSUPPORTED_FORMAT` | File type not accepted |
| `FILE_TOO_LARGE` | Exceeds `MAX_UPLOAD_SIZE_MB` (default 512 MB) |
| `MINIO_UNAVAILABLE` | Storage backend unreachable |

---

## Analysis Service

### POST /api/v1/analyze
Submit an analysis job.

**Request**
```json
{
  "study_id": "uuid",
  "task": "segmentation" | "detection" | "classification",
  "config": {}
}
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
  "status": "queued" | "running" | "completed" | "failed",
  "task": "segmentation",
  "results": {
    "mask_url": "https://…/results/{job_id}/mask.png?X-Amz-Expires=3600",
    "gradcam_url": "https://…/results/{job_id}/gradcam.png?X-Amz-Expires=3600",
    "confidence": 0.94,
    "findings": [
      { "label": "GLIOMA", "bbox": [x1, y1, x2, y2], "area_px": 3842 }
    ]
  }
}
```

Note: presigned URLs expire in 1 hour. Do not cache them.

---

## GradCAM Service

### POST /api/v1/explain
Generate a GradCAM heatmap for a model output.

**Request**
```json
{
  "model_name": "skin_classification" | "pneumonia_detection" | "mri_segmentation",
  "image_url": "s3://medvision/studies/{study_id}/original.jpg",
  "target_class": 4
}
```

**Response** `200`
```json
{
  "heatmap_url": "https://…/results/{job_id}/gradcam.png?X-Amz-Expires=3600",
  "top_regions": [
    { "bbox": [x1, y1, x2, y2], "score": 0.87 }
  ]
}
```

---

## Report Service

### POST /api/v1/reports/generate
Trigger AI report generation from completed analysis jobs.

**Request**
```json
{
  "study_id": "uuid",
  "job_ids": ["uuid", "uuid"],
  "patient_context": {
    "age": 45,
    "sex": "F",
    "clinical_indication": "Headache, vision changes"
  }
}
```

**Response** `202`
```json
{ "report_id": "uuid", "status": "generating" }
```

---

### GET /api/v1/reports/{report_id}
Retrieve a generated report.

**Response** `200`
```json
{
  "report_id": "uuid",
  "status": "completed" | "generating" | "failed",
  "modality": "MRI",
  "content": "## MRI Brain Report\n\n**Clinical Indication** ...",
  "created_at": "2026-04-26T12:00:00Z"
}
```

Report sections:
1. Clinical indication (from `patient_context`)
2. Technique (auto-filled from modality)
3. Findings (LLM expansion of structured ML output)
4. Impression (LLM summary)
5. Confidence disclaimer (always appended — AI-assisted, requires radiologist review)

---

## Health Endpoints

All services expose identical health endpoints (no auth required):

```
GET /health   → { "status": "ok" }
GET /ready    → { "status": "ok", "checks": { "db": true, "redis": true } }
GET /metrics  → Prometheus text format
```

Readiness checks differ per service:
- `upload_service`: MinIO reachable
- `analysis_service`: PostgreSQL, Redis, Triton gRPC reachable
- `report_service`: PostgreSQL, Anthropic API key configured
- `gradcam_service`: MinIO reachable
