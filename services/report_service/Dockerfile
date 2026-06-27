# ── Build stage: CPU-only torch + deps into an isolated venv ─────────────────
FROM python:3.11-slim@sha256:cdbd05fb6f457ca275ff51ce00d93d865ca0b6a25f5ffb08262d94f6835771e5 AS builder
ENV PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app
# CPU-only torch first (~800MB vs 3GB for CUDA), then the rest (skip torch).
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN grep -v "^torch" requirements.txt | pip install -r /dev/stdin

# ── Runtime stage: libgomp1 is required by torch ─────────────────────────────
FROM python:3.11-slim@sha256:cdbd05fb6f457ca275ff51ce00d93d865ca0b6a25f5ffb08262d94f6835771e5 AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/data/hf_cache
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --gid 10001 appuser \
 && useradd --uid 10001 --gid 10001 --create-home appuser \
 && mkdir -p /data/hf_cache \
 && chown -R appuser:appuser /data
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY app/ ./app/
COPY templates/ ./templates/
COPY alembic.ini ./
COPY migrations/ ./migrations/
USER appuser
EXPOSE 8005
HEALTHCHECK --interval=20s --timeout=5s --start-period=60s --retries=6 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8005/health', timeout=3)"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8005"]
