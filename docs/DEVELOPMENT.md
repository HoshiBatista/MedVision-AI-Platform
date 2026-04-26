# Development Guide

## Prerequisites

| Tool | Version |
|---|---|
| Python | 3.13+ |
| Docker + Docker Compose | latest |
| Node.js | 20+ |
| Git | any |

Optional but recommended:
- `make` — all dev commands are wrapped in the Makefile
- `uv` — fast Python package installer (or plain `pip`)

---

## Initial Setup

### 1. Clone and configure environment

```bash
git clone https://github.com/your-org/medvision-ai-platform
cd medvision-ai-platform
cp .env.example .env
# Edit .env — fill in API keys, change default passwords
```

### 2. Python environment (ML code)

```bash
cd ml
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # if present, else install per-task deps

# Verify ClearML connectivity
python check_clearml.py
```

### 3. Download datasets

```bash
# Requires ROBOFLOW_API_KEY in .env
python download_datasets.py         # all three
python download_datasets.py --task mri   # single dataset
```

### 4. Start the platform

```bash
make up        # starts all containers (gateway, services, redis, postgres, minio, triton)
make logs SERVICE=analysis_service   # tail a specific service
make down      # stop and remove containers
```

---

## Development Workflow

### Run a single service with hot-reload

```bash
make dev SERVICE=analysis_service
```

This mounts the service directory into the container and runs `uvicorn --reload`.

### Train a model locally

```bash
# Uses Apple Silicon MPS. Change device="cuda" in train.py for NVIDIA GPU.
python ml/mri_segmentation/train.py --no-clearml

# With ClearML tracking
python ml/mri_segmentation/train.py
```

### Export and deploy a model

```bash
python ml/mri_segmentation/export_onnx.py \
  --weights runs/mri_segmentation/train/weights/best.pt

# Then restart Triton to pick up the new model
docker compose restart triton
```

### Run benchmark

```bash
python ml/mri_segmentation/benchmark.py --weights best.pt --runs 200
python ml/mri_segmentation/benchmark.py --weights model.onnx --onnx
```

### Debug predictions visually

```bash
python ml/mri_segmentation/debug.py --weights best.pt --n 9 --save
```

---

## Code Standards

### Python

- **Type hints everywhere** — `mypy --strict` must pass
- **No `print()`** — use `structlog.get_logger()`
- **No secrets in code** — all config via environment variables
- **Async all the way** — FastAPI endpoints must be `async def`
- **Test coverage ≥ 80%** on business logic (not on model/training code)

### Pre-commit hooks

```bash
pip install pre-commit
pre-commit install
```

Hooks run: `ruff` (lint), `black` (format), `mypy` (types), `hadolint` (Dockerfile lint).

### FastAPI services

Every service must implement:
- `GET /health` — always `{"status": "ok"}` (liveness)
- `GET /ready` — checks downstream dependencies (readiness)
- `GET /metrics` — Prometheus format via `prometheus-fastapi-instrumentator`

Use the dependency injection pattern — no global state:

```python
# core/deps.py
async def get_db() -> AsyncGenerator[AsyncSession, None]: ...
async def get_redis() -> Redis: ...
async def get_current_user(token: str = Depends(oauth2_scheme)) -> User: ...
```

Error responses must use the standard envelope:
```python
raise HTTPException(
    status_code=503,
    detail={"detail": "Triton unreachable", "error_code": "TRITON_UNAVAILABLE"}
)
```

### ML code

- Config in `config.yaml`, loaded via `argparse` — never hardcoded hyperparameters
- All ClearML task setup through `ml/shared/clearml_utils.init_task()`
- Never call `Task.init()` inside Celery workers — use `Task.get_task()` or offline task
- Always apply DICOM pixel rescaling (`RescaleSlope` / `RescaleIntercept`) before inference

---

## Running Tests

```bash
# Integration tests (requires running Docker stack)
make test-integration

# E2E tests
make test-e2e

# Specific test file
pytest tests/integration/test_upload_pipeline.py -v
```

Integration tests hit real PostgreSQL, Redis, and MinIO — no mocking of infrastructure.

---

## Useful Make Targets

```bash
make up                         # start full stack
make down                       # stop and remove containers
make dev SERVICE=<name>         # hot-reload dev mode for one service
make train TASK=mri_segmentation
make export TASK=mri_segmentation VERSION=1
make test-integration
make lint                       # ruff + mypy across all services
make logs SERVICE=<name>
```

---

## IDE Setup (VS Code)

Recommended extensions:
- Python (Microsoft)
- Pylance
- Ruff
- Docker
- YAML

`.vscode/settings.json`:
```json
{
  "python.defaultInterpreterPath": "ml/.venv/bin/python",
  "editor.formatOnSave": true,
  "[python]": { "editor.defaultFormatter": "charliermarsh.ruff" }
}
```

---

## Common Footguns

| Issue | Fix |
|---|---|
| Triton rejects tensor | Always send `[batch, C, H, W]` even for batch=1 — wrong rank is rejected |
| Wrong MRI intensities | Apply `RescaleSlope`/`RescaleIntercept` from DICOM header before inference |
| ClearML in Celery workers | Use `Task.get_task()` not `Task.init()` inside workers |
| GradCAM on ViT | Standard GradCAM doesn't work on attention layers — use attention rollout or target the final LayerNorm |
| Stale MinIO URLs | Presigned URLs expire in 1h — never cache them past that |
| Celery result loss | Use Redis as result backend, not RPC — RPC breaks on worker restart |
| SSL on Python 3.13/macOS | Set `SSL_CERT_FILE` to `certifi.where()` (see `download_datasets.py`) |
