# ── Build stage: install deps into an isolated venv ──────────────────────────
FROM python:3.12-slim@sha256:6c4dd321d176d61ea848dc8c73a4f7dbae8f70e0ee48bb411ea2f045b599fa8e AS builder
ENV PIP_NO_CACHE_DIR=1
# gcc as insurance for any sdist-only dependency (psycopg2-binary ships wheels).
RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# ── Runtime stage: no compilers; psycopg2-binary bundles libpq ───────────────
FROM python:3.12-slim@sha256:6c4dd321d176d61ea848dc8c73a4f7dbae8f70e0ee48bb411ea2f045b599fa8e AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"
RUN groupadd --gid 10001 appuser \
 && useradd --uid 10001 --gid 10001 --create-home appuser \
 && mkdir -p /data/studies \
 && chown -R appuser:appuser /data
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY app/ ./app/
COPY alembic.ini ./
COPY migrations/ ./migrations/
USER appuser
EXPOSE 8003
HEALTHCHECK --interval=15s --timeout=5s --start-period=25s --retries=5 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8003/health', timeout=3)"]
# Default: run the FastAPI service.
# To run the Celery worker instead, override the command:
#   celery -A app.workers.celery_app.celery worker -Q analysis -c 2 -l info
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]
