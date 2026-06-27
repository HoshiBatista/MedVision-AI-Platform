# Deployment Guide

## Environments

| Environment | Stack | Notes |
|---|---|---|
| Local dev | Docker Compose | `make up` (CPU) or `make up-gpu` (GPU) |
| Staging | Docker Compose on a VM | same compose file, `.env` overrides |
| Production | Kubernetes + Helm | **planned** — `infra/helm/` / `infra/terraform/` are empty scaffolding |

---

## Environment Variables

Copy `.env.example` to `.env` and fill in every value. Never commit `.env`.

```bash
cp .env.example .env
```

Key variables:

| Variable | Default | Production action |
|---|---|---|
| `POSTGRES_PASSWORD` | `medvision` | Generate a strong random password |
| `JWT_SECRET_KEY` | placeholder | `openssl rand -hex 32` |
| `INFERENCE_BACKEND` | `onnx` | `triton` for GPU serving |
| `LLM_MODEL_NAME` | OpenBioLLM-8B Q8 (HF GGUF) | Verify the tag, or pick another Ollama model |
| `OLLAMA_URL` | `http://ollama:11434` | Point at an external Ollama if not in-compose |
| `CLEARML_API_ACCESS_KEY` | placeholder | From ClearML settings (training only) |
| `ENVIRONMENT` | `development` | `production` |
| `DOCS_ENABLED` | `true` | `false` |

There is no object-storage or external-LLM configuration — storage is local volumes and the LLM is the in-cluster Ollama.

---

## Docker Compose (Dev / Staging)

### Start the stack

```bash
make up        # CPU inference (local ONNX Runtime) — no GPU required
make up-gpu    # GPU inference via NVIDIA Triton (needs the NVIDIA Container Toolkit)
# or plain:    docker compose up -d
```

Services started (default / CPU):
- `gateway` — :80
- `auth_service` — :8001
- `upload_service` — :8002
- `analysis_service` + `analysis_worker` (Celery) — :8003
- `gradcam_service` — :8004
- `report_service` — :8005
- `ollama` — :11434 (report LLM)
- `frontend` — :3000
- `postgres` — :5432, `redis` — :6379

Optional profiles:
- `--profile triton` (via `make up-gpu`) → `triton` on :8010 (HTTP) / :8011 (gRPC) / :8012 (metrics)
- `--profile monitoring` (via `make up-monitoring`) → `prometheus` :9090, `grafana` :3001

### GPU requirement

Only the GPU variant needs hardware: an NVIDIA GPU + the NVIDIA Container Toolkit. The
default CPU stack runs anywhere (including Apple Silicon) — `analysis_worker` runs ONNX
Runtime on CPU and `ollama` runs a quantised model on CPU. The GPU overlay
(`docker-compose.gpu.yml`) reserves the GPU for both `triton` and `ollama`.

```bash
# Install the NVIDIA Container Toolkit (Ubuntu)
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Useful compose commands

```bash
docker compose logs -f analysis_service
docker compose exec postgres psql -U medvision medvision
docker compose exec ollama ollama list
```

---

## Model Deployment

Trained YOLOv11 weights are exported to ONNX and placed in the Triton-style repo:

```bash
python ml/mri_segmentation/export_onnx.py \
  --weights runs/mri_segmentation/train/weights/best.pt
# → writes the ONNX into triton_models/mri_segmentation/1/
```

Both inference backends read this same repo (`triton_models/<model>/1/*.onnx`, resolved by glob):
- **CPU (default)**: `analysis_worker` loads the ONNX directly via ONNX Runtime — restart the worker to pick up a new model (`docker compose restart analysis_worker`).
- **GPU**: Triton serves the repo — reload via the model-control API or `docker compose restart triton`.

The report LLM is pulled into Ollama automatically at `report_service` startup (`ensure_model`).

---

## Database Migrations

Schema is managed via Alembic per service; each service runs `alembic upgrade head` before
its app starts (compose `command`). The four DB services share one database but use separate
`alembic_version_<svc>` tables.

```bash
make migrate                      # all DB services
docker compose exec auth_service alembic upgrade head   # one service
```

---

## Kubernetes / Helm (planned)

`infra/helm/` and `infra/terraform/` are empty placeholders — there is no working chart yet.
A production K8s deployment would need: a chart per service (Deployment/Service/Ingress),
PVCs for the study/heatmap volumes, a GPU node pool for `triton`/`ollama`, an HPA on the
Celery worker, and secrets in K8s Secrets / a vault.

---

## Observability

- **Prometheus** (`--profile monitoring`): scrapes every `/metrics`. Config: `infra/monitoring/prometheus.yml`. UI: `http://localhost:9090`.
- **Grafana**: dashboards in `infra/monitoring/grafana/dashboards/`. UI: `http://localhost:3001` (default `admin`/`admin`).
- **Alerting**: rules in `infra/monitoring/alerts.yml` (service down, 5xx rate, inference latency, Celery queue depth).
- **Tracing (OpenTelemetry → Jaeger)**: planned, not yet wired.

---

## CI/CD

GitHub Actions: `.github/workflows/ci.yml`. Jobs: ruff + mypy (advisory), bandit, per-service
pytest with coverage, frontend typecheck/lint/build, hadolint, docker build of all images,
compose validation, **e2e** (boots the full stack and runs `tests/e2e`), CodeQL, repo hygiene
(actionlint/yamllint/gitleaks), aggregated by a `ci-success` gate.
