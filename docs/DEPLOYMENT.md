# Deployment Guide

## Environments

| Environment | Stack | Notes |
|---|---|---|
| Local dev | Docker Compose | `make up`, hot-reload available |
| Staging | Docker Compose on VM | same compose file, `.env` overrides |
| Production | Kubernetes + Helm | `infra/helm/` charts |

---

## Environment Variables

Copy `.env.example` to `.env` and fill in every value. Never commit `.env`.

```bash
cp .env.example .env
```

Key variables to change from defaults:

| Variable | Default | Production action |
|---|---|---|
| `POSTGRES_PASSWORD` | `change_me_in_prod` | Generate strong random password |
| `MINIO_SECRET_KEY` | `change_me_in_prod` | Generate strong random password |
| `JWT_SECRET_KEY` | placeholder | `openssl rand -hex 32` |
| `ANTHROPIC_API_KEY` | `sk-ant-...` | Real key from Anthropic console |
| `CLEARML_API_ACCESS_KEY` | placeholder | From ClearML settings |
| `ENVIRONMENT` | `development` | Set to `production` |
| `DOCS_ENABLED` | `true` | Set to `false` |
| `ALLOWED_ORIGINS` | `http://localhost:3000` | Production frontend URL |

---

## Docker Compose (Dev / Staging)

### Start full stack

```bash
make up
# or
docker compose up -d
```

Services started:
- `gateway` — Nginx on :80
- `auth_service` — :8001
- `upload_service` — :8002
- `analysis_service` + Celery worker — :8003
- `report_service` — :8005
- `gradcam_service` — :8004
- `postgres` — :5432
- `redis` — :6379
- `minio` — :9000 (API) / :9001 (console)
- `triton` — :8000 (HTTP) / :8001 (gRPC) / :8002 (metrics)
- `prometheus` — :9090
- `grafana` — :3001
- `jaeger` — :16686 (UI) / :14268 (collector)

### Triton GPU requirement

Triton requires an NVIDIA GPU with the NVIDIA Container Toolkit installed:

```bash
# Install NVIDIA Container Toolkit (Ubuntu)
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
  | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

On Apple Silicon (development only), Triton is not available natively — run models locally with `device="mps"` in the training scripts and skip Triton.

### Useful compose commands

```bash
docker compose logs -f analysis_service
docker compose restart triton
docker compose exec postgres psql -U medvision medvision
docker compose exec minio mc ls local/medvision
```

---

## Model Deployment to Triton

1. Train the model:
   ```bash
   python ml/mri_segmentation/train.py
   ```

2. Export to ONNX:
   ```bash
   python ml/mri_segmentation/export_onnx.py \
     --weights runs/mri_segmentation/train/weights/best.pt
   ```
   This copies `model.onnx` to `triton_models/mri_segmentation/1/model.onnx`.

3. Reload Triton (zero-downtime model reload):
   ```bash
   # HTTP model control API
   curl -X POST http://localhost:8000/v2/repository/models/mri_segmentation/load
   ```
   Or restart the container:
   ```bash
   docker compose restart triton
   ```

4. Verify model is ready:
   ```bash
   curl http://localhost:8000/v2/models/mri_segmentation/ready
   # → {"ready":true}
   ```

---

## Kubernetes / Helm (Production)

Helm charts are in `infra/helm/`. Each service has its own chart.

```bash
# Add values override for your environment
cp infra/helm/values.yaml infra/helm/values.prod.yaml

# Deploy
helm upgrade --install medvision-ai ./infra/helm \
  --values infra/helm/values.prod.yaml \
  --namespace medvision \
  --create-namespace
```

Production considerations:
- Use a managed PostgreSQL (e.g., RDS, Cloud SQL) instead of the in-cluster container
- Use managed Redis (ElastiCache, Memorystore) for Celery broker/backend
- Use real S3 (or GCS with S3-compatible API) instead of MinIO
- Set resource requests/limits for Triton (needs GPU node pool)
- Enable horizontal pod autoscaling on analysis_service Celery workers
- Store secrets in Kubernetes Secrets or an external vault — not in `values.yaml`

---

## Database Migrations

Database schema is managed via Alembic (configured per service in `services/*/app/`):

```bash
# Inside the service container
docker compose exec auth_service alembic upgrade head
docker compose exec analysis_service alembic upgrade head
```

Never run migrations against production without a backup.

---

## Observability

### Prometheus

Scrapes all services on their `/metrics` endpoint. Config: `infra/monitoring/prometheus.yml`.

Open Prometheus: `http://localhost:9090`

### Grafana

Pre-built dashboards in `infra/monitoring/grafana/dashboards/`.

Open Grafana: `http://localhost:3001` (default credentials: `admin` / `admin`, change on first login)

### Jaeger (Distributed Tracing)

Open Jaeger UI: `http://localhost:16686`

Each Triton gRPC call and database query emits a span. Use this to trace slow requests across services.

### Alerting

Alert rules defined in `infra/monitoring/alerts.yml`. Key alerts:
- Service instance down > 2 minutes
- HTTP 5xx error rate > 5%
- Inference latency P99 > 2s
- Celery queue depth > 50 tasks

Configure Alertmanager to route alerts to Slack/PagerDuty.

---

## CI/CD

GitHub Actions workflow: `.github/workflows/blank.yml` (extend as needed).

Recommended pipeline stages:
1. `ruff` + `mypy` lint across all services
2. Unit tests
3. Docker build (fail fast if any Dockerfile has errors)
4. Integration tests against `docker compose`
5. Push image to registry
6. `helm upgrade` to staging
7. Manual approval gate → production deploy
