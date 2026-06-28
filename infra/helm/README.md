# MedVision AI — Helm chart

Kubernetes deployment for the full MedVision stack. Mirrors `docker-compose.yml`:
5 FastAPI services + Celery worker, Postgres, Redis, Ollama, optional Triton (GPU),
an nginx gateway (JWT routing + rate-limiting + static file serving), and the
React frontend.

Chart lives in [`medvision/`](./medvision).

## Prerequisites

1. **Kubernetes** ≥ 1.25 and **Helm** ≥ 3.10.
2. **A ReadWriteMany StorageClass.** `study-data` and `heatmap-data` are shared
   across pods (upload + worker write, gradcam + gateway read), so they need RWX
   (NFS, CephFS, AWS EFS, Azure Files, GCP Filestore, …). Set it via
   `storage.studyData.storageClass` / `storage.heatmapData.storageClass`.
   On a single-node dev cluster you can also point both at a hostPath-backed RWX
   provisioner.
3. **Service images** built and pushed to a registry the cluster can pull from:
   `auth_service`, `upload_service`, `analysis_service`, `gradcam_service`,
   `report_service`, `frontend`, `gateway`. Set `image.repository`,
   `image.tag`, and `global.imageRegistry`.
4. **(GPU only)** NVIDIA device plugin installed for `triton.enabled=true` and/or
   `ollama.gpu=true`.

## Install

```bash
# 1. Lint / preview
helm lint infra/helm/medvision
helm template medv infra/helm/medvision | less

# 2. Install (CPU / ONNX inference — the default)
helm install medv infra/helm/medvision \
  --namespace medvision --create-namespace \
  --set global.imageRegistry=ghcr.io/hoshibatista/ \
  --set image.tag=0.1.0 \
  --set storage.studyData.storageClass=nfs-rwx \
  --set storage.heatmapData.storageClass=nfs-rwx \
  --set secrets.jwtSecretKey=$(openssl rand -hex 32) \
  --set secrets.postgresPassword=$(openssl rand -hex 16) \
  --set secrets.adminPassword=$(openssl rand -hex 12)
```

### Populate the model repo

The ONNX models (`triton_models/`) are too large for ConfigMaps and aren't baked
into images. Copy them into the `*-model-repo` PVC once after install (the worker
in `onnx` mode and Triton both read `/models`):

```bash
kubectl -n medvision cp triton_models/. <model-loader-pod>:/models/
```

See the post-install `NOTES` for a ready-made loader-pod snippet.

## Common configurations

| Goal | Flags |
|---|---|
| GPU inference via Triton | `--set triton.enabled=true --set inference.backend=triton` |
| GPU report generation | `--set ollama.gpu=true` |
| Expose via Ingress | `--set ingress.enabled=true --set ingress.className=nginx --set ingress.host=medvision.example.com` |
| External secrets | `--set secrets.existingSecret=my-secret` (keys: `JWT_SECRET_KEY`, `POSTGRES_PASSWORD`, `ADMIN_PASSWORD`) |
| Scale a service | `--set services.analysis_worker.replicas=3` |

## Access

- **With Ingress:** `http(s)://<ingress.host>/`
- **Without:** `kubectl -n medvision port-forward svc/medv-medvision-gateway 8080:80`
  then open `http://localhost:8080/`.

## Architecture notes

- **Migrations** run as a per-service `alembic upgrade head` initContainer. Each
  DB service uses its own `alembic_version_<svc>` table, so they're independent
  and safe to run in parallel against the shared `medvision` database.
- **Gateway** ships a templated `nginx.conf` (ConfigMap) because the baked one
  references compose service names with underscores, which are invalid K8s DNS
  labels. Upstreams point at `<release>-<service>` Services.
- **Postgres/Redis** are single-replica StatefulSets with RWO PVCs — fine for
  dev/staging. For production prefer a managed Postgres and an HA Redis; point
  the services at them via `secrets.existingSecret` + a custom ConfigMap, or
  disable the bundled ones (`postgres.enabled=false`, `redis.enabled=false`)
  and override the URLs.
- **Security:** application pods run as non-root (uid 10001, `fsGroup` 10001 for
  shared-volume writes). The frontend (`bare`) and gateway run nginx as-is.

## Uninstall

```bash
helm uninstall medv -n medvision
# PVCs are retained by default; delete explicitly if you want the data gone:
kubectl -n medvision delete pvc -l app.kubernetes.io/instance=medv
```
