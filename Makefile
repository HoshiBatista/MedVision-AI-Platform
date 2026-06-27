.PHONY: help up up-cpu down build logs ps shell restart \
        up-monitoring up-triton up-gpu up-all \
        lint test-integration e2e clean migrate makemigration

DC      := docker compose
# GPU/Triton variant: base stack + GPU overlay + triton profile, with inference
# routed to Triton instead of the default local ONNX Runtime (CPU) backend.
GPU_DC  := INFERENCE_BACKEND=triton $(DC) -f docker-compose.yml -f docker-compose.gpu.yml --profile triton
SERVICE ?= auth_service

help:
	@echo "MedVision AI — dev commands"
	@echo ""
	@echo "  make up              Start core stack — CPU inference (local ONNX Runtime)"
	@echo "  make up-gpu          Core stack — GPU inference via Triton (needs NVIDIA toolkit)"
	@echo "  make up-monitoring   Core + Prometheus + Grafana"
	@echo "  make up-all          Everything (GPU/Triton + monitoring)"
	@echo "  make down            Stop and remove containers"
	@echo "  make build           Rebuild all images"
	@echo "  make logs            Tail logs (SERVICE=<name> for one service)"
	@echo "  make ps              Show running containers"
	@echo "  make shell           Open a shell in SERVICE container"
	@echo "  make restart         Restart SERVICE"
	@echo "  make lint            Run ruff + mypy on all services"
	@echo "  make migrate         Apply Alembic migrations (all DB services)"
	@echo "  make makemigration   Autogenerate a revision (SERVICE=<svc> MSG=\"...\")"
	@echo "  make test-integration Run integration tests"
	@echo "  make e2e             Bring up the stack and run e2e tests (then tear down)"
	@echo "  make clean           Remove volumes (DATA LOSS)"

# ── Ensure .env exists ────────────────────────────────────────────────────────
.env:
	@echo ".env not found — copying from .env.example"
	cp .env.example .env
	@echo "Edit .env and set secrets, then re-run make up"
	@exit 1

# ── Stack management ──────────────────────────────────────────────────────────
# Default: CPU inference (local ONNX Runtime) — no GPU required.
up: .env
	$(DC) up -d --remove-orphans

up-cpu: up   # alias

# GPU inference via Triton Inference Server (requires an NVIDIA GPU + toolkit).
up-gpu: .env
	$(GPU_DC) up -d --remove-orphans

# Back-compat alias for up-gpu.
up-triton: up-gpu

up-monitoring: .env
	$(DC) --profile monitoring up -d --remove-orphans

up-all: .env
	$(GPU_DC) --profile monitoring up -d --remove-orphans

down:
	$(DC) down

build:
	$(DC) build --parallel

# ── Operational ───────────────────────────────────────────────────────────────
logs:
	$(DC) logs -f $(SERVICE)

ps:
	$(DC) ps

shell:
	$(DC) exec $(SERVICE) /bin/sh

restart:
	$(DC) restart $(SERVICE)

# ── Quality ───────────────────────────────────────────────────────────────────
lint:
	@for svc in auth_service upload_service analysis_service gradcam_service report_service; do \
	  echo "── $$svc ──"; \
	  $(DC) run --rm --no-deps $$svc sh -c "pip install ruff mypy -q && ruff check app/ && mypy app/ --ignore-missing-imports" || true; \
	done

test-integration:
	$(DC) run --rm --no-deps -e ENVIRONMENT=test auth_service \
	  sh -c "pip install pytest httpx -q && pytest /app/tests/integration/ -v"

# Full-stack e2e: build + start the core stack (no triton/monitoring profiles),
# run the e2e suite against the gateway, then tear everything down.
# Requires pytest + httpx on the host (pip install pytest httpx).
e2e: .env
	$(DC) up -d --build
	@pytest tests/e2e -v; rc=$$?; \
	  $(DC) logs --no-color --tail=200 > e2e-compose.log 2>&1 || true; \
	  $(DC) down -v; \
	  exit $$rc

# ── Database migrations (Alembic) ─────────────────────────────────────────────
# On `make up` each DB service runs `alembic upgrade head` before its app starts;
# these targets are for manual/dev use.
migrate:
	@for svc in auth_service upload_service analysis_service report_service; do \
	  echo "── migrate $$svc ──"; \
	  $(DC) run --rm $$svc alembic upgrade head; \
	done

# Autogenerate a revision for one service, e.g.:
#   make makemigration SERVICE=auth_service MSG="add last_login"
makemigration:
	$(DC) run --rm $(SERVICE) alembic revision --autogenerate -m "$(MSG)"

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	@echo "WARNING: This will delete all volumes (postgres data, heatmaps, model cache)."
	@read -p "Continue? [y/N] " ans && [ "$$ans" = "y" ]
	$(DC) down -v
